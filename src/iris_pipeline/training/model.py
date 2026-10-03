from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"


def make_estimator(c: float, max_iter: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(C=c, max_iter=max_iter, random_state=42),
            ),
        ]
    )


def split_data(frame: pd.DataFrame, test_size: float, random_state: int):
    return train_test_split(
        frame[FEATURES],
        frame[TARGET],
        test_size=test_size,
        random_state=random_state,
        stratify=frame[TARGET],
    )
