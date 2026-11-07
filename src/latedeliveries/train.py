"""Train, tune, score and save the late-delivery model.
Run with: python -m latedeliveries.train"""

import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from latedeliveries.data import make_orders
from latedeliveries.evaluation import (
    choose_threshold,
    order_cost,
    out_of_fold_probabilities,
)
from latedeliveries.grid_search import run_grid_search
from latedeliveries.pipeline_modified import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    save_cv_results,
    split_orders,
)

ROOT = Path(__file__).resolve().parents[2]  # the repository folder
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def score_at_threshold(pipeline, X, y, threshold) -> dict:
    """Confusion matrix, precision, recall, F1, ROC-AUC at the chosen threshold."""
    probability = pipeline.predict_proba(X)[:, 1]
    flagged = probability >= threshold
    return {
        "confusion_matrix": confusion_matrix(y, flagged).tolist(),
        "precision": round(float(precision_score(y, flagged)), 3),
        "recall": round(float(recall_score(y, flagged)), 3),
        "f1": round(float(f1_score(y, flagged)), 3),
        "roc_auc": round(float(roc_auc_score(y, probability)), 3),
    }


def policy_costs(pipeline, X, y, threshold) -> dict:
    """Total cost of three policies on the same orders."""
    probability = pipeline.predict_proba(X)[:, 1]
    return {
        "flag_nobody": order_cost(y, np.zeros(len(y), dtype=bool)),
        "flag_everybody": order_cost(y, np.ones(len(y), dtype=bool)),
        "your_model": order_cost(y, probability >= threshold),
    }


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1. data and split

    orders = make_orders(n=3000, seed=7)
    X_train, X_test, y_train, y_test = split_orders(orders)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 2. comparison -> reports/cv_results.csv
    results_table = save_cv_results(REPORTS_DIR / "cv_results.csv")
    untuned = results_table[results_table["model"] == "LogisticRegression"].iloc[0]
    untuned_mean = float(untuned["recall_mean"])
    untuned_std = float(untuned["recall_std"])

    # 3. tuning
    search = run_grid_search(X_train, y_train, folds)
    best_pipeline = search.best_estimator_
    best_params = search.best_params_
    improvement = search.best_score_ - untuned_mean
    tuning = {
        "scoring": "recall",
        "untuned_recall": round(untuned_mean, 4),
        "tuned_recall": round(float(search.best_score_), 4),
        "improvement": round(float(improvement), 4),
        "untuned_std": round(untuned_std, 4),
        "beats_spread": bool(improvement > untuned_std),
    }
    print("Tuning:", tuning)

    # 4. threshold, from training data only
    probabilities = out_of_fold_probabilities(best_pipeline, X_train, y_train, folds)
    threshold, train_cost = choose_threshold(y_train, probabilities)
    print(f"Chosen threshold: {threshold} (training cost {train_cost})")

    # 5. Final fit on the whole training set
    final = clone(best_pipeline)
    final.fit(X_train, y_train)

    # 6. the test set used ONCE

    test_scores = score_at_threshold(final, X_test, y_test, threshold)
    test_costs = policy_costs(final, X_test, y_test, threshold)
    print("Test scores:", test_scores)
    print("Test costs:", test_costs)

    # 7. save
    metrics = {"threshold": threshold, "test": test_scores, "costs": test_costs}
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    joblib.dump(final, MODELS_DIR / "late_orders.joblib")
    info = {
        "scikit_learn": sklearn.__version__,
        "seed": RANDOM_STATE,
        "threshold": threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_params": best_params,
        "test": test_scores,
    }
    (MODELS_DIR / "model_info.json").write_text(
        json.dumps(info, indent=2, default=str), encoding="utf-8"
    )
    print("Saved model and reports.")


if __name__ == "__main__":
    main()
