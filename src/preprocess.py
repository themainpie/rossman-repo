import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

IMPUTE_ONLY_COLS = ["CompetitionDistance", "Promo2SinceWeek", "Promo2SinceYear", "PromoOpen"]
OHE_COLS = ["PromoInterval", "StateHoliday"]
OE_COLS = ["StoreType", "Assortment"]
LAGROLL_DIFF_COLS = ["lag1", "lag7", "rolling_3", "rolling_7", "diff1", "diff7"]


def merge_store_data(df_main, df_store):
    return df_main.merge(df_store, on="Store", how="left")


def clean_data(df, config):
    """
    Cleans the input dataframe by handling missing values and correcting data types,

    Returns:
    pandas.DataFrame
        Cleaned dataframe ready for feature engineering or preprocessing.
    """
    cfg = config["data_cleaning"]

    # log_sales filter
    if cfg["log_sales_filter"]["enabled"] and "Sales" in df.columns:
        ranges = cfg["log_sales_filter"]["keep_ranges"]

        log_sales = np.log1p(df["Sales"])

        mask = False
        for r in ranges:
            if "max" in r:
                mask |= log_sales <= r["max"]
            if "min" in r:
                mask |= log_sales >= r["min"]

        df = df[mask]

    # customers filter
    if cfg["customers_filter"]["enabled"] and "Customers" in df.columns:
        df = df[df["Customers"] < cfg["customers_filter"]["max_customers"]]

    # sales capping / replacement
    if cfg["sales_capping"]["enabled"] and "Sales" in df.columns:
        df.loc[
            df["Sales"] > cfg["sales_capping"]["threshold"],
            "Sales"
        ] = cfg["sales_capping"]["replace_with"]


    df["StateHoliday"] = df["StateHoliday"].astype("str")

    return df


def add_features(df):
    """
    Adds engineered features to a dataframe (train or test), including:
    - Date features (Year, Month, WeekOfYear)
    - Competition and promo durations
    - Lag, rolling, and diff features for Sales

    For test data (no 'Sales'), train history is automatically loaded to compute lag features.

    Args:
        df (pd.DataFrame): Input dataframe (train or test)

    Returns:
        pd.DataFrame: Dataframe with new feature columns
    """
    import pandas as pd

    df = df.copy()

    if "Year" not in df.columns:
        df["Year"] = pd.to_datetime(df["Date"]).dt.year
    if "Month" not in df.columns:
        df["Month"] = pd.to_datetime(df["Date"]).dt.month
    if "WeekOfYear" not in df.columns:
        df["WeekOfYear"] = pd.to_datetime(df["Date"]).dt.isocalendar().week.astype(int)

    df["competition_open"] = 12 * (df.Year - df.CompetitionOpenSinceYear) + (df.Month - df.CompetitionOpenSinceMonth)
    df["PromoOpen"] = 12 * (df.Year - df.Promo2SinceYear) + (df.WeekOfYear - df.Promo2SinceWeek) / 4

    if 'Sales' in df.columns:
        df = df.sort_values(['Store', 'Date'])
        df["lag1"] = df.groupby('Store')['Sales'].shift(1)
        df["lag7"] = df.groupby('Store')['Sales'].shift(7)
        df["rolling_3"] = df.groupby('Store')['Sales'].shift(1).rolling(3, min_periods=1).mean()
        df["rolling_7"] = df.groupby('Store')['Sales'].shift(1).rolling(7, min_periods=1).mean()
        df["diff1"] = df.groupby('Store')['Sales'].diff(1)
        df["diff7"] = df.groupby('Store')['Sales'].diff(7)
        return df.drop("Customers", axis=1)

    from src.load_data import load_train
    train_df = load_train()
    train_df = train_df.copy()
    train_df['is_train'] = 1
    df['is_train'] = 0

    combined = pd.concat([train_df, df], ignore_index=True)
    combined = combined.sort_values(['Store', 'Date'])

    combined['lag1'] = combined.groupby('Store')['Sales'].shift(1)
    combined['lag7'] = combined.groupby('Store')['Sales'].shift(7)
    combined['rolling_3'] = combined.groupby('Store')['Sales'].shift(1).rolling(3, min_periods=1).mean()
    combined['rolling_7'] = combined.groupby('Store')['Sales'].shift(1).rolling(7, min_periods=1).mean()
    combined['diff1'] = combined.groupby('Store')['Sales'].diff(1)
    combined['diff7'] = combined.groupby('Store')['Sales'].diff(7)

    test_features = combined[combined['is_train'] == 0].copy()
    test_features.drop(columns=['is_train', 'Id'], inplace=True)

    return test_features.drop("Customers", axis=1)


def prepare_data(df, extra_df=None, config=None, target_column=None):
    """
    Complete preparing pipeline.
    Returns: X, y (features and target) or prepared df
    """
    df = merge_store_data(df, extra_df)
    df = clean_data(df, config)
    df = add_features(df)

    if target_column:
        X = df.drop(["Sales", "CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"], axis=1)
        y = df["Sales"]
        return X, y
    return df


def build_preprocessing_pipeline():
    """
    Builds and returns the sklearn preprocessing pipeline.

    Returns:
    ColumnTransformer
        Configured preprocessing pipeline.
    """

    impute_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0))
    ])

    ohe_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    oe_pipeline = Pipeline([
        ("oe_encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    # Most frequent imputation for lag/rolling/diff features
    # to ensure stable SHAP value computation
    mostf_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("impute_only", impute_pipeline, IMPUTE_ONLY_COLS),
            ("ohe", ohe_pipeline, OHE_COLS),
            ("oe", oe_pipeline, OE_COLS),
            ("lag_features", mostf_pipeline, LAGROLL_DIFF_COLS),
        ],
        remainder="passthrough"
    )

    return preprocessor