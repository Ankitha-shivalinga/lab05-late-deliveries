"""Feature preprocessing and model comparison for late deliveries."""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from .data import make_orders

# Features, split and preprocessing are defined once, in pipeline.py.
# They are re-exported here so existing imports from this module keep working.
from .pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    build_pipeline,
    split_orders,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "NUMERIC_FEATURES",
    "RANDOM_STATE",
    "SCORING",
    "build_pipeline",
    "compare_models",
    "get_candidate_models",
    "save_cv_results",
    "split_orders",
]

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def get_candidate_models():
    """Return the baseline and three machine-learning candidate models."""
    return {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "BalancedDecisionTree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def compare_models():
    """Compare candidate models using five-fold stratified cross-validation.

    Only the training portion is used for cross-validation. The test portion
    is kept separate for the final evaluation stage of the assignment.
    """
    orders = make_orders(n=3000, seed=7)
    X_train, _X_test, y_train, _y_test = split_orders(orders)

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for model_name, model in get_candidate_models().items():
        model_pipeline = build_pipeline(model)

        scores = cross_validate(
            estimator=model_pipeline,
            X=X_train,
            y=y_train,
            cv=cross_validation,
            scoring=SCORING,
            n_jobs=-1,
            error_score="raise",
        )

        result_row = {"model": model_name}

        for metric_name in SCORING:
            fold_scores = scores[f"test_{metric_name}"]
            result_row[f"{metric_name}_mean"] = fold_scores.mean()
            result_row[f"{metric_name}_std"] = fold_scores.std()

        results.append(result_row)

    return pd.DataFrame(results)


def save_cv_results(output_path="reports/cv_results.csv"):
    """Save the model-comparison results as a CSV file and print the table."""
    results = compare_models()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)

    print("\nMODEL COMPARISON — 5-FOLD STRATIFIED CROSS-VALIDATION")
    print(results.round(4).to_string(index=False))
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    save_cv_results()
