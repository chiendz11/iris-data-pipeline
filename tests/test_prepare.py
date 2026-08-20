from pathlib import Path

import pandas as pd

from src.prepare import build_dataset


def test_build_dataset_is_small_and_balanced(tmp_path: Path):
    output = build_dataset(tmp_path / "iris.csv")
    frame = pd.read_csv(output)
    assert frame.shape == (150, 5)
    assert frame["species"].value_counts().to_dict() == {
        "setosa": 50,
        "versicolor": 50,
        "virginica": 50,
    }
    assert not frame.isna().any().any()

