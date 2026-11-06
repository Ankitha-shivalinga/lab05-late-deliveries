from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .data import make_orders
from .pipeline_modified import build_pipeline, split_orders

orders = make_orders(n=3000, seed=7)

X_train, X_test, y_train, y_test = split_orders(orders)

model = build_pipeline(
    LogisticRegression(max_iter=1000, random_state=7)
)
param_grid = {
    "model__C": [0.1, 1, 10],
    "model__class_weight": [None, "balanced"],
}

folds = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=7,
)
search = GridSearchCV(
    model,
    param_grid,
    scoring="recall",
    cv=folds,
    n_jobs=-1,
)

search.fit(X_train, y_train)

print(search.best_params_)
print(round(search.best_score_, 3))

print("Best settings:", search.best_params_)
print("Best recall:", round(search.best_score_, 3))

original_recall = 0.6611
original_std = 0.0683

improvement = search.best_score_ - original_recall

print("Original recall:", original_recall)
print("Tuned recall:", round(search.best_score_, 4))
print("Improvement:", round(improvement, 4))
print("Original standard deviation:", original_std)
print("Improvement greater than spread:", improvement > original_std)