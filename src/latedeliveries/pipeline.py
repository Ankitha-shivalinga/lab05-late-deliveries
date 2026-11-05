"""Feature definitions and preprocessing for late-delivery prediction."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 7

NUMERIC_FEATURES = [
    "distance_km",
    "items",
    "hour",
    "courier_months",
]

CATEGORICAL_FEATURES = [
    "weather",
    "day_type",
    "vehicle",
]


def split_orders(frame):
    """Return a repeatable, stratified 75/25 train/test split."""
    X = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = frame["late"]

    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_pipeline(model):
    """Combine feature preprocessing with the supplied model."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
