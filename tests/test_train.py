import pandas as pd

from src.prepare import build_dataset
from src.train import make_estimator, split_data


def test_logistic_regression_reaches_quality_gate(tmp_path):
    data_path = build_dataset(tmp_path / "iris.csv")
    frame = pd.read_csv(data_path)
    x_train, x_test, y_train, y_test = split_data(frame, test_size=0.2, random_state=42)
    estimator = make_estimator(c=1.0, max_iter=400)
    estimator.fit(x_train, y_train)
    assert estimator.score(x_test, y_test) >= 0.90

