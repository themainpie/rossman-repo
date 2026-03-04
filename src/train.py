import numpy as np
import xgboost as xgb
import joblib
import yaml
import logging
import os

from src.preprocess import prepare_data, build_preprocessing_pipeline
from src.load_data import load_train, load_store, BASE_PATH
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, BaseCrossValidator, GridSearchCV
from sklearn.metrics import mean_squared_error
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

config_path = BASE_PATH / "config.yaml"
log_dir = BASE_PATH / "logs"

with config_path.open("r") as f:
    config = yaml.safe_load(f)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "train.log")
    ]
)

seed=config["seed"]

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


def split_and_save(X, y, groups, test_size=0.2, save_path=None):
    """
    Splits data into train and test sets based on unique group values (e.g., dates)
    and optionally saves the split datasets to a .pkl file.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe.
    y : pd.Series or pd.DataFrame
        Target values.
    groups : pd.Series
        Series indicating the group for each sample (e.g., dates). Splitting is
        performed based on unique group values to avoid data leakage.
    test_size : float, default=0.2
        Proportion of unique groups to include in the test set. Must be between 0 and 1.
    save_path : str or None, default=None
        Directory to save the split datasets as a .pkl file. If None, no file is saved.

    Returns
    -------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series or pd.DataFrame
        Training target.
    y_test : pd.Series or pd.DataFrame
        Test target.
    groups_train : pd.Series
        Groups corresponding to training samples.
    groups_test : pd.Series
        Groups corresponding to test samples.

    Notes
    -----
    - The function ensures that all samples with the same group value are kept in
      the same split (train or test).
    - If `save_path` is provided, the splits are saved with a timestamped filename
      in the format: split_data_YYYYMMDD_HHMMSS.pkl.
    """
    unique_dates = np.unique(groups)
    split_point = int(len(unique_dates) * (1 - test_size))

    train_dates = unique_dates[:split_point]
    test_dates = unique_dates[split_point:]

    train_idx = np.where(np.isin(groups, train_dates))[0]
    test_idx = np.where(np.isin(groups, test_dates))[0]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"{save_path}/split_data_{timestamp}.pkl"

        joblib.dump(
            {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "groups_train": groups_train,
                "groups_test": groups_test,
            },
            file_path
        )
        print(f"Data saved to {file_path}")

    return X_train, X_test, y_train, y_test, groups_train, groups_test


def demo_train(X_train, X_test, y_train, y_test, preprocessor):

    model_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", xgb.XGBRegressor(enable_categorical=True))
    ])
    print("starting pipeline fit...")
    model_pipeline.fit(X_train, y_train)
    print("fitting finished.")
    y_pred = model_pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    metrics = {'MSE': round(mse, 2), 'RMSE': round(rmse, 2)}
    return metrics


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
    y_sqrt = np.sqrt(y)

    logging.info("Splitting data with time-based strategy...")
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_and_save(X, y_sqrt, groups)
    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "groups_train": groups_train,
            "groups_test": groups_test,
        },
        f"data/processed/split_data_{timestamp}.pkl")


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
            param_distributions=config["tuning"]["param_dist"],
            n_iter=config["tuning"]["n_iter"],
            cv=GroupTimeSeriesCV(config["tuning"]["cv"]),
            n_jobs=-1,
            verbose=1,
            random_state=seed
        )
    
    elif method == "GridSearchCV":
        search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=config["tuning"]["param_dist"],
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

    joblib.dump(best_model, f"models/model_{timestamp}.pkl")
    logging.info(f"Model saved as model_{timestamp}.pkl")

    return best_model


if __name__ == "__main__":
    model = train()