
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from .pipeline_modified import build_pipeline


def run_grid_search(X_train, y_train, folds):
    model = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=7)
    )

    param_grid = {
        "model__C": [0.1, 1, 10],
        "model__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        model,
        param_grid,
        scoring="recall",
        cv=folds,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    return search

