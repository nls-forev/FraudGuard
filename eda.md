import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Config
# ============================================================
DATA_PATH = "data/base.csv"   # change if needed
TARGET = "fraud_bool"

DROP_COLS = [
    "days_since_request",     # extra column vs training schema
    "device_fraud_count",     # constant zero in your EDA
]

# Columns where -1 means "missing / not available"
SENTINEL_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

# Add missing-indicator flags for these columns
MISSING_FLAG_COLS = SENTINEL_COLS.copy()

# Strongly skewed, non-negative columns that benefit from log1p
LOG1P_COLS = [
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
    "proposed_credit_limit",
]

BINARY_COLS = [
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "has_other_cards",
    "foreign_request",
    "keep_alive_session",
]

CATEGORICAL_COLS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

NUMERIC_COLS = [
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "bank_months_count",
    "proposed_credit_limit",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
    "month",
]

# Numerical columns after adding missing flags
NUMERIC_WITH_FLAGS = NUMERIC_COLS + [f"{c}_missing" for c in MISSING_FLAG_COLS]


# ============================================================
# Cleaning / feature engineering
# ============================================================
def clean_baf_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop columns not used in the schema / no signal
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=col)

    # Standardize categorical strings
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            if col == "device_os":
                df[col] = df[col].astype(str).str.lower().str.strip()
            else:
                df[col] = df[col].astype(str).str.upper().str.strip()

    # Cast target
    df[TARGET] = df[TARGET].astype(int)

    # Cast binary columns to integers
    for col in BINARY_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Create missing flags before replacing sentinel values
    for col in SENTINEL_COLS:
        flag_col = f"{col}_missing"
        df[flag_col] = (df[col] == -1).astype("Int64")

    # Replace sentinel -1 with NaN in designated columns
    for col in SENTINEL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == -1, col] = np.nan

    # Convert all numeric columns to numeric dtype
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Log-transform skewed non-negative features
    # (sentinel -1 has already been replaced with NaN)
    for col in LOG1P_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    return df


# ============================================================
# Preprocessor
# ============================================================
def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """
    scale_numeric=False is a good default for tree models (XGBoost/LightGBM/CatBoost/RandomForest).
    Set scale_numeric=True for linear models / SVM.
    """

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_WITH_FLAGS),
            ("bin", binary_pipeline, BINARY_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ============================================================
# Full pipeline
# ============================================================
def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_numeric: bool = False,
):
    df = clean_baf_dataframe(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(scale_numeric=scale_numeric)

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_t = pd.DataFrame(X_train_t, columns=feature_names, index=X_train.index)
    X_test_t = pd.DataFrame(X_test_t, columns=feature_names, index=X_test.index)

    return X_train_t, X_test_t, y_train, y_test, preprocessor


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    X_train_t, X_test_t, y_train, y_test, preprocessor = prepare_data(
        df,
        test_size=0.2,
        random_state=42,
        scale_numeric=False,   # set True for Logistic Regression / SVM
    )

    print("X_train shape:", X_train_t.shape)
    print("X_test shape:", X_test_t.shape)
    print("Train fraud rate:", y_train.mean())
    print("Test fraud rate:", y_test.mean())

    # Save transformed data
    X_train_t.to_csv("X_train_transformed.csv", index=False)
    X_test_t.to_csv("X_test_transformed.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)

    print("Saved transformed datasets.")
