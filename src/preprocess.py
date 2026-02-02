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
    if cfg["log_sales_filter"]["enabled"]:
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
    if cfg["customers_filter"]["enabled"]:
        df = df[df["Customers"] < cfg["customers_filter"]["max_customers"]]

    # sales capping / replacement
    if cfg["sales_capping"]["enabled"]:
        df.loc[
            df["Sales"] > cfg["sales_capping"]["threshold"],
            "Sales"
        ] = cfg["sales_capping"]["replace_with"]


    df["StateHoliday"] = df["StateHoliday"].astype("str")

    return df


def add_features(df):
    """
    Adds engineered features to the input dataframe.

    Returns:
    pandas.DataFrame
        Dataframe with newly added feature columns.
    """
    df = df.copy()

    # Prevent overwriting existing columns (e.g., if they were created earlier in notebook)
    if "Year" not in df.columns:
        df["Year"] = pd.to_datetime(df["Date"]).dt.year
    
    if "Month" not in df.columns:
        df["Month"] = pd.to_datetime(df["Date"]).dt.month

    if "WeekOfYear" not in df.columns:
        df["WeekOfYear"] = pd.to_datetime(df["Date"]).dt.isocalendar().week.astype(int)

    # Calculating how long each store's competitions and promotion has been open (in months)
    df["competition_open"] = 12 * (df.Year - df.CompetitionOpenSinceYear) + (df.Month - df.CompetitionOpenSinceMonth)
    df["PromoOpen"] = 12 * (df.Year - df.Promo2SinceYear) + (df.WeekOfYear - df.Promo2SinceWeek) / 4

    # Encode weekly cycle using sine and cosine to capture day-of-week patterns
    df["dow_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    df["is_weekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

    df["lag1"] = df["Sales"].shift(1)
    df["lag7"] = df["Sales"].shift(7)

    df["rolling_3"] = df["Sales"].shift(1).rolling(window=3).mean()
    df["rolling_7"] = df["Sales"].shift(1).rolling(window=7).mean()

    df["diff1"] = df["Sales"].diff(1)
    df["diff7"] = df["Sales"].diff(7)

    return df


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