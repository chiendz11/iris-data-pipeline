from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"


@dataclass(frozen=True)
class TrainingResult:
    accuracy: float
    f1_macro: float
    run_id: str
    model_version: str | None
    promoted: bool
    baseline_accuracy: float | None = None


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
    x = frame[FEATURES]
    y = frame[TARGET]
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes"}


def _champion_accuracy(client: MlflowClient, model_name: str) -> float | None:
    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        return None
    run = client.get_run(champion.run_id)
    value = run.data.metrics.get("accuracy")
    return float(value) if value is not None else None


def _register_candidate(
    *,
    run_id: str,
    model_uri: str,
    model_name: str,
    accuracy: float,
    min_accuracy: float,
    min_improvement: float,
) -> tuple[str, bool, float | None]:
    client = MlflowClient()
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(registered.version)
    client.set_registered_model_alias(model_name, "candidate", version)
    baseline = _champion_accuracy(client, model_name)
    eligible = accuracy >= min_accuracy and (
        baseline is None or accuracy >= baseline + min_improvement
    )
    client.set_model_version_tag(model_name, version, "quality_gate", "passed" if eligible else "failed")
    client.set_model_version_tag(model_name, version, "source_run_id", run_id)
    client.set_model_version_tag(
        model_name, version, "baseline_accuracy", "none" if baseline is None else str(baseline)
    )
    # Champion is deliberately untouched here. Argo first rolls this candidate out at
    # 10%; src.promote changes the alias only after the online canary gate succeeds.
    return version, eligible, baseline


def train(
    data_path: str | Path = "data/iris.csv",
    config_path: str | Path = "params.yaml",
) -> TrainingResult:
    config = load_config(config_path)
    frame = pd.read_csv(data_path)
    x_train, x_test, y_train, y_test = split_data(
        frame,
        test_size=float(config["data"]["test_size"]),
        random_state=int(config["data"]["random_state"]),
    )
    estimator = make_estimator(
        c=float(config["model"]["c"]),
        max_iter=int(config["model"]["max_iter"]),
    )
    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    f1_macro = float(f1_score(y_test, predictions, average="macro"))

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-classification")
    model_name = os.getenv("MODEL_NAME", "iris-classifier")
    register_model = _bool_env("REGISTER_MODEL", tracking_uri.startswith("http"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run() as active_run:
        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "c": config["model"]["c"],
                "max_iter": config["model"]["max_iter"],
                "test_size": config["data"]["test_size"],
                "random_state": config["data"]["random_state"],
                "dataset": "sklearn.datasets.load_iris",
            }
        )
        mlflow.log_metrics({"accuracy": accuracy, "f1_macro": f1_macro})
        mlflow.set_tags(
            {
                "git_commit": os.getenv("GIT_COMMIT", "local"),
                "dataset_version": os.getenv("DATASET_VERSION", "dvc-workspace"),
            }
        )
        signature = infer_signature(x_test, predictions)
        model_info = mlflow.sklearn.log_model(
            sk_model=estimator,
            name="model",
            signature=signature,
            input_example=x_test.head(3),
        )
        run_id = active_run.info.run_id

    version = None
    promoted = False
    baseline_accuracy = None
    if register_model:
        version, promoted, baseline_accuracy = _register_candidate(
            run_id=run_id,
            model_uri=model_info.model_uri,
            model_name=model_name,
            accuracy=accuracy,
            min_accuracy=float(config["quality_gate"]["min_accuracy"]),
            min_improvement=float(config["quality_gate"]["min_improvement"]),
        )

    result = TrainingResult(accuracy, f1_macro, run_id, version, promoted, baseline_accuracy)
    Path("metrics.json").write_text(
        json.dumps(
            {
                "accuracy": result.accuracy,
                "f1_macro": result.f1_macro,
                "run_id": result.run_id,
                "model_version": result.model_version,
                "quality_gate_passed": result.promoted,
                "baseline_accuracy": result.baseline_accuracy,
                "bootstrap_required": result.promoted and result.baseline_accuracy is None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/model-version.txt").write_text((version or "unregistered") + "\n", encoding="utf-8")
    Path("outputs/quality-gate.txt").write_text(str(promoted).lower() + "\n", encoding="utf-8")
    Path("outputs/bootstrap-required.txt").write_text(
        str(promoted and baseline_accuracy is None).lower() + "\n", encoding="utf-8"
    )
    Path("outputs/promotion.json").write_text(
        json.dumps(
            {
                "model_version": version,
                "quality_gate_passed": promoted,
                "baseline_accuracy": baseline_accuracy,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", nargs="?", default="data/iris.csv")
    parser.add_argument("config_path", nargs="?", default="params.yaml")
    cli_args = parser.parse_args()
    print(train(cli_args.data_path, cli_args.config_path))
