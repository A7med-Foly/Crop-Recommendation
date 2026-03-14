import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET_COL   = 'label'

RANDOM_STATE = 42
TEST_SIZE    = 0.20


def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the raw CSV and return (X, y).

    Args:
        csv_path: path to Crop_recommendation.csv

    Returns:
        X: DataFrame with the 7 feature columns
        y: Series with the crop label
    """
    df = pd.read_csv(csv_path)

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    print(f"✓ Loaded {len(df)} rows, {df[TARGET_COL].nunique()} crop classes")
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split into train and test sets.

    Kept separate from load_data() so you can call load_data() alone
    during EDA without triggering a split.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"✓ Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


def build_pipeline() -> make_column_transformer:
    """
    Build the ColumnTransformer (scaler).

    Returns:
        An unfitted ColumnTransformer with MinMaxScaler on all 7 features
    """
    transformer = make_column_transformer(
        (MinMaxScaler(), FEATURE_COLS),
        remainder='drop'
    )
    return transformer


def fit_and_transform(
    transformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple:
    """
    Fit the transformer on X_train ONLY, then transform both sets.

    Args:
        transformer: unfitted ColumnTransformer from build_pipeline()
        X_train:     training features
        X_test:      test features

    Returns:
        X_train_scaled, X_test_scaled (both as numpy arrays)
    """
    X_train_scaled = transformer.fit_transform(X_train)
    X_test_scaled  = transformer.transform(X_test)

    print(f"✓ Scaled features shape — train: {X_train_scaled.shape}, test: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled