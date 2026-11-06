import numpy as np
from sklearn.model_selection import cross_val_predict

VOUCHER_COST = 4
LATE_COST = 15
THRESHOLDS = np.round(np.arange(0.05, 0.9501, 0.05), 2)


def order_cost(y_true, flagged) -> int:
    """Total cost: $4 per flagged order + $25 per late order not flagged."""

    late = np.asarray(y_true).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    vouchers = flagged.sum()
    missed_late = (late & ~flagged).sum()
    return int(VOUCHER_COST * vouchers + LATE_COST * missed_late)


def out_of_fold_probabilities(pipeline, X_train, y_train, folds):
    """Chance of 'late' for each TRAINING row, from a model that never saw that row."""
    return cross_val_predict(
        pipeline, X_train, y_train, cv=folds, method="predict_proba"
    )[:, 1]


def choose_threshold(y_true, probabilities):
    """Return (best_threshold, its_cost). Uses training data only."""
    probabilities = np.asarray(probabilities)
    costs = [order_cost(y_true, probabilities >= t) for t in THRESHOLDS]
    best = int(np.argmin(costs))  # on a tie, the lowest threshold wins
    return float(THRESHOLDS[best]), costs[best]
