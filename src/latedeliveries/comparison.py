"""Compare baseline and machine-learning models for late-delivery prediction.

Run from the project root with:
    python -m latedeliveries.evaluation

The program uses the project's split_orders and build_pipeline functions so
that the feature definitions, preprocessing, and train/test split stay
consistent with pipeline.py.
"""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from latedeliveries.data import make_orders
from latedeliveries.pipeline import (
    RANDOM_STATE,
    build_pipeline,
    split_orders,
)

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def get_candidate_models():
    """Return a baseline and three candidate classification models."""
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
    """Evaluate all candidates with five-fold stratified CV on training data.

    The test split is deliberately not used here. It should remain untouched
    until the final model evaluation stage.
    """
    # Generate the assignment dataset and use the split function in pipeline.py.
    frame = make_orders(n=3000, seed=7)
    X_train, _X_test, y_train, _y_test = split_orders(frame)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []
    for model_name, model in get_candidate_models().items():
        pipeline = build_pipeline(model)

        scores = cross_validate(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
            error_score="raise",
        )

        row = {"model": model_name}
        for metric_name in SCORING:
            fold_scores = scores[f"test_{metric_name}"]
            row[f"{metric_name}_mean"] = fold_scores.mean()
            row[f"{metric_name}_std"] = fold_scores.std()

        rows.append(row)

    return pd.DataFrame(rows)


def save_cv_results(output_path="reports/cv_results.csv"):
    """Save and print the cross-validation comparison table."""
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
