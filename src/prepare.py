from __future__ import annotations

from pathlib import Path

from sklearn.datasets import load_iris


OUTPUT_PATH = Path("data/iris.csv")


def build_dataset(output_path: Path = OUTPUT_PATH) -> Path:
    """Materialize sklearn's versioned Iris bundle as a deterministic CSV."""
    dataset = load_iris(as_frame=True)
    frame = dataset.frame.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )
    frame["species"] = frame["target"].map(dict(enumerate(dataset.target_names)))
    frame = frame.drop(columns=["target"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, lineterminator="\n")
    return output_path


if __name__ == "__main__":
    path = build_dataset()
    print(f"Wrote {path}")

