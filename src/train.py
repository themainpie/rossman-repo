import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import yaml
import logging

from src.preprocess import prepare_data, build_preprocessing_pipeline
from src.load_data import load_train, load_store
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, BaseCrossValidator, GridSearchCV
from sklearn.metrics import mean_squared_error

with open("config.yaml") as f:
    config = yaml.safe_load(f)

seed=config["seed"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train.log")
    ]
)


# Custom cross-validator to prevent time-series target leakage.
# Ensures that training data always precedes validation data in time,
# using unique calendar dates shared across multiple stores.
class GroupTimeSeriesCV(BaseCrossValidator):
    """
    Time series cross-validator that splits data based on unique groups/dates.

    Takes:
    - X: feature matrix (not used in splitting)
    - y: target array (optional)
    - groups: array of group labels (e.g., dates)

    Returns:
    - train_idx, test_idx : indices for training and testing for each split
    """
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):        
        unique_dates = np.unique(groups)
        n_dates = len(unique_dates)

        fold_size = n_dates // (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):
            train_dates = unique_dates[: fold_size * i]
            test_dates  = unique_dates[fold_size * i : fold_size * (i + 1)]

            train_idx = np.where(np.isin(groups, train_dates))[0]
            test_idx  = np.where(np.isin(groups, test_dates))[0]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def time_split(X, y, groups, test_size=0.2):
    """
    Splits features, target, and group labels into training and testing sets based on time.

    Parameters:
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    groups : pd.Series
        Grouping variable (e.g., dates) used to preserve temporal order.
    test_size : float, default=0.2
        Fraction of data to use as the test set.

    Returns:
    X_train, X_test : pd.DataFrame
        Split feature matrices for training and testing.
    y_train, y_test : pd.Series
        Split target variables for training and testing.
    groups_train, groups_test : pd.Series
        Split groups corresponding to training and testing sets.

    Notes:
    - Ensures that all rows with the same group label are kept in the same split.
    """
    unique_dates = np.unique(groups)
    split_point = int(len(unique_dates) * (1 - test_size))

    train_dates = unique_dates[:split_point]
    test_dates = unique_dates[split_point:]

    train_idx = np.where(np.isin(groups, train_dates))[0]
    test_idx = np.where(np.isin(groups, test_dates))[0]

    return (
        X.iloc[train_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx],
        groups.iloc[train_idx], groups.iloc[test_idx]
    )


def demo_train():
    model = xgb.XGBRegressor(random_state=seed)
    train = load_train()
    store = load_store()

    X, y = prepare_data(train, store, config, "Date")
    groups = X.pop("Date")
    X_train, X_test, y_train, y_test = time_split(X, y, groups)
    
    preprocessor = build_preprocessing_pipeline()
    model_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, y_pred))

    return score


def train():
    """
    Trains an XGBoost regression model using a preprocessing pipeline and hyperparameter tuning.

    Workflow:
    1. Loads training and store datasets.
    2. Prepares features and target using `prepare_data`.
    3. Splits data with a time-based strategy using `time_split`.
    4. Builds a preprocessing pipeline.
    5. Performs hyperparameter tuning with either RandomizedSearchCV or GridSearchCV.

    Requirements:
    - `config` dictionary with keys:
        - "tuning": {"method", "cv", "n_iter"}
        - "param_dist": hyperparameter distributions or grid
    - `seed` integer for reproducibility
    - Functions: load_train(), load_store(), prepare_data(), time_split(), build_preprocessing_pipeline()
    - GroupTimeSeriesCV class for CV splitting

    Returns:
    best_model : estimator
        The trained model with the best-found hyperparameters.
    """
    logging.info("Starting training process...")

    method = config["tuning"]["method"]

    logging.info("Loading train and store datasets...")
    train = load_train()
    store = load_store()

    logging.info("Preparing features and target...")
    X, y = prepare_data(train, store, config, "Date")
    groups = X.pop("Date")

    logging.info("Splitting data with time-based strategy...")
    X_train, X_test, y_train, y_test, groups_train, groups_test = time_split(X, y, groups)
    joblib.dump((X_train, X_test, y_train, y_test, groups_train, groups_test),
            "data/processed/split_data.pkl")
    logging.info("Saved train/test split to data/preprocessed")


    logging.info("Building preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline()

    full_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", xgb.XGBRegressor())
    ])

    logging.info(f"Starting hyperparameter tuning using {method}...")
    if method == "RandomSearchCV":
        search = RandomizedSearchCV(
            estimator=full_pipeline,
            param_distributions=config["param_dist"],
            n_iter=config["tuning"]["n_iter"],
            cv=GroupTimeSeriesCV(config["tuning"]["cv"]),
            n_jobs=-1,
            verbose=1,
            random_state=seed
        )
    
    elif method == "GridSearchCV":
        search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=config["param_dist"],
            cv=GroupTimeSeriesCV(config["tuning"]["cv"]),
            n_jobs=-1,
            verbose=1,
        )
    
    else:
        raise ValueError(f"Unknown tuning method: {method}")
    
    search.fit(X_train, y_train, groups=groups_train)
    logging.info("Training finished.")

    best_model = search.best_estimator_
    logging.info(f"Tuned model best score: {search.best_score_}")
    logging.info(f"Best hyperparameters: {search.best_params_}")

    joblib.dump(best_model, "models/best_model.pkl")
    logging.info("Model saved at models/best_model.pkl")

    return best_model


if __name__ == "__main__":
    model = train()