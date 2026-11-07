import joblib
from sklearn.linear_model import LogisticRegression

from latedeliveries.data import make_orders
from latedeliveries.pipeline import build_pipeline, split_orders


def test_saved_model_predicts_the_same_after_reloading(tmp_path):
    orders = make_orders(n=600, seed=7)
    X_train, X_test, y_train, y_test = split_orders(orders)
    model = build_pipeline(LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    reloaded = joblib.load(path)

    assert (reloaded.predict(X_test) == model.predict(X_test)).all()
