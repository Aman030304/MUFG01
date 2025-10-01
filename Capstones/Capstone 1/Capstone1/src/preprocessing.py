import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLUMNS = ["Shift", "Machine_Type", "Material_Grade", "Day_of_Week"]
TARGET_COLUMN = "Parts_Per_Hour"
TIMESTAMP_COLUMN = "Timestamp"


def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    if TIMESTAMP_COLUMN in df.columns:
        df = df.drop(columns=[TIMESTAMP_COLUMN])
        logging.info("Dropped column: Timestamp")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_feature_cols = [c for c in numeric_cols if c != TARGET_COLUMN]

    for col in numeric_feature_cols:
        if df[col].isna().any():
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)
            logging.info(
                f"Filled missing values in numeric column '{col}' with mean={mean_value:.4f}"
            )

    categorical_present = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    if categorical_present:
        df[categorical_present] = df[categorical_present].astype("category")
        df = pd.get_dummies(df, columns=categorical_present, drop_first=True)
        logging.info(f"Encoded categorical columns: {', '.join(categorical_present)}")
    else:
        logging.info("No specified categorical columns found to encode.")

    remaining_missing = int(df.isna().sum().sum())
    if remaining_missing > 0:
        logging.warning(
            f"Remaining missing values after preprocessing: {remaining_missing}"
        )

    return df


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    Scale numerical features using StandardScaler.
    
    Args:
        df: Preprocessed DataFrame with features and target
        
    Returns:
        tuple: (scaled_features, target, scaler)
    """
    # Separate features and target
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    
    # Identify numerical features (exclude one-hot encoded categorical columns)
    # One-hot columns start with the original categorical column name followed by '_'
    dummy_prefixes = [f"{c}_" for c in CATEGORICAL_COLUMNS]
    numerical_features = [
        c for c in X.columns
        if (pd.api.types.is_numeric_dtype(X[c]) and not any(c.startswith(pref) for pref in dummy_prefixes))
    ]
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_scaled = X.copy()
    
    if numerical_features:
        # Fit on DataFrame to capture feature_names_in_ for inference
        X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])
        logging.info(f"Scaled {len(numerical_features)} numerical features using StandardScaler")
    else:
        logging.info("No numerical features found to scale")
    
    return X_scaled, y, scaler


class ColumnOrderEncoder:
    """
    Lightweight encoder that preserves training feature order (after one-hot encoding)
    and can transform new raw inputs via pd.get_dummies with drop_first=True,
    then align columns to the training order.
    """

    def __init__(self, feature_names: list[str], categorical_columns: list[str] | None = None) -> None:
        self.feature_names: list[str] = feature_names
        self.categorical_columns: list[str] = categorical_columns or CATEGORICAL_COLUMNS

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()
        if TIMESTAMP_COLUMN in df.columns:
            df = df.drop(columns=[TIMESTAMP_COLUMN])
        # Ensure expected categorical columns exist (Streamlit form should supply them)
        cats_present = [c for c in self.categorical_columns if c in df.columns]
        if cats_present:
            df[cats_present] = df[cats_present].astype("category")
            df = pd.get_dummies(df, columns=cats_present, drop_first=True)
        # Align to training feature order
        df_aligned = df.reindex(columns=self.feature_names, fill_value=0)
        return df_aligned


def build_encoder_from_df(preprocessed_df: pd.DataFrame) -> ColumnOrderEncoder:
    """
    Build a ColumnOrderEncoder from a preprocessed (one-hot encoded) DataFrame.

    The DataFrame must still contain TARGET_COLUMN; the returned encoder stores
    the training feature order excluding the target.
    """
    feature_names = [c for c in preprocessed_df.columns if c != TARGET_COLUMN]
    return ColumnOrderEncoder(feature_names=feature_names, categorical_columns=CATEGORICAL_COLUMNS)


def preprocess_input(user_input: dict | pd.DataFrame, scaler: StandardScaler, encoder: ColumnOrderEncoder) -> np.ndarray:
    """
    Prepare a single-row input for inference using the saved encoder and scaler.

    Args:
        user_input: Mapping or single-row DataFrame of raw inputs (original schema)
        scaler: Fitted StandardScaler (trained on numeric feature subset)
        encoder: ColumnOrderEncoder with training feature order

    Returns:
        numpy.ndarray: Transformed row, ordered as during training, ready for model.predict
    """
    # Coerce to single-row DataFrame
    if isinstance(user_input, dict):
        input_df = pd.DataFrame([user_input])
    else:
        input_df = user_input.copy()

    # Encode and align columns
    X = encoder.transform(input_df)

    # Scale only numerical features that scaler was trained on
    numerical_features = getattr(scaler, 'feature_names_in_', None)
    if numerical_features is not None:
        numerical_features = list(numerical_features)
        X.loc[:, numerical_features] = scaler.transform(X[numerical_features])
    else:
        # Fallback: do nothing if feature names are unavailable
        logging.warning("Scaler is missing feature_names_in_; skipping scaling at inference.")

    return X.values
