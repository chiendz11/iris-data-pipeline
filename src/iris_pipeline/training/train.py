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

from iris_pipeline.contracts.model_result import (
    ModelMetrics,
    ModelResult,
    write_argo_parameters,
    write_model_result,
)
from iris_pipeline.evaluation.offline import evaluate_predictions, passes_quality_gate
from iris_pipeline.training.config import load_config
from iris_pipeline.training.model import make_estimator, split_data


@dataclass(frozen=True)
class TrainingResult:
    accuracy: float
    f1_macro: float
    run_id: str
    model_version: str | None
    quality_gate_passed: bool
    baseline_accuracy: float | None = None
    baseline_model_version: str | None = None


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes"}


def _champion_baseline(
    client: MlflowClient, model_name: str
) -> tuple[float | None, str | None]:
    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException as error:
        if error.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None, None
        raise
    run = client.get_run(champion.run_id)
    value = run.data.metrics.get("accuracy")
    if value is None:
        raise RuntimeError(
            f"Champion {model_name} v{champion.version} has no accuracy metric"
        )
    return float(value), str(champion.version)


def _register_model_result(
    *,
    run_id: str,
    model_uri: str,
    model_name: str,
    accuracy: float,
    min_accuracy: float,
    min_improvement: float,
) -> tuple[str, bool, float | None, str | None]:
    """Register an immutable candidate; deployment aliases remain GitOps-owned."""
    client = MlflowClient()
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(registered.version)
    baseline, baseline_version = _champion_baseline(client, model_name)
    eligible = passes_quality_gate(
        accuracy,
        min_accuracy=min_accuracy,
        baseline_accuracy=baseline,
        min_improvement=min_improvement,
    )
    client.set_model_version_tag(
        model_name, version, "quality_gate", "passed" if eligible else "failed"
    )
    client.set_model_version_tag(model_name, version, "source_run_id", run_id)
    client.set_model_version_tag(
        model_name, version, "baseline_accuracy", "none" if baseline is None else str(baseline)
    )
    client.set_model_version_tag(
        model_name,
        version,
        "baseline_model_version",
        "none" if baseline_version is None else baseline_version,
    )
    return version, eligible, baseline, baseline_version


def train(
    data_path: str | Path = "data/iris.csv",
    config_path: str | Path = "params.yaml",
    result_path: str | Path = "outputs/model-result.json",
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
    metrics = evaluate_predictions(y_test, predictions)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-classification")
    model_name = os.getenv("MODEL_NAME", "iris-classifier")
    register_model = _bool_env("REGISTER_MODEL", tracking_uri.startswith("http"))
    dataset_version = os.getenv("DATASET_VERSION", "dvc-workspace")
    # Keep local/offline runs schema-valid while production always supplies the real Git SHA.
    source_sha = os.getenv("GIT_COMMIT", "0000000")
    source_repository = os.getenv("SOURCE_REPOSITORY", "chiendz11/iris-data-pipeline")
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
        mlflow.log_metrics({"accuracy": metrics.accuracy, "f1_macro": metrics.f1_macro})
        mlflow.set_tags({"git_commit": source_sha, "dataset_version": dataset_version})
        signature = infer_signature(x_test, predictions)
        model_info = mlflow.sklearn.log_model(
            sk_model=estimator,
            name="model",
            signature=signature,
            input_example=x_test.head(3),
        )
        run_id = active_run.info.run_id

    version = None
    baseline_accuracy = None
    baseline_model_version = None
    quality_gate_passed = passes_quality_gate(
        metrics.accuracy,
        min_accuracy=float(config["quality_gate"]["min_accuracy"]),
        baseline_accuracy=None,
        min_improvement=float(config["quality_gate"]["min_improvement"]),
    )
    if register_model:
        (
            version,
            quality_gate_passed,
            baseline_accuracy,
            baseline_model_version,
        ) = _register_model_result(
            run_id=run_id,
            model_uri=model_info.model_uri,
            model_name=model_name,
            accuracy=metrics.accuracy,
            min_accuracy=float(config["quality_gate"]["min_accuracy"]),
            min_improvement=float(config["quality_gate"]["min_improvement"]),
        )

    result = TrainingResult(
        metrics.accuracy,
        metrics.f1_macro,
        run_id,
        version,
        quality_gate_passed,
        baseline_accuracy,
        baseline_model_version,
    )
    Path("metrics.json").write_text(
        json.dumps(
            {
                "accuracy": result.accuracy,
                "f1_macro": result.f1_macro,
                "run_id": result.run_id,
                "model_version": result.model_version,
                "quality_gate_passed": result.quality_gate_passed,
                "baseline_accuracy": result.baseline_accuracy,
                "baseline_model_version": result.baseline_model_version,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    model_result = ModelResult(
        model_name=model_name,
        model_version=version,
        baseline_model_version=baseline_model_version,
        quality_gate_passed=quality_gate_passed,
        bootstrap_required=(register_model and quality_gate_passed and baseline_accuracy is None),
        run_id=run_id,
        dataset_version=dataset_version,
        source_repository=source_repository,
        source_sha=source_sha,
        metrics=ModelMetrics(
            accuracy=metrics.accuracy,
            f1_macro=metrics.f1_macro,
            baseline_accuracy=baseline_accuracy,
        ),
    )
    contract_path = write_model_result(model_result, result_path)
    write_argo_parameters(model_result, contract_path.parent)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate an Iris classifier.")
    parser.add_argument("data_path", nargs="?", default="data/iris.csv")
    parser.add_argument("config_path", nargs="?", default="params.yaml")
    parser.add_argument("--result-path", default="outputs/model-result.json")
    args = parser.parse_args()
    print(train(args.data_path, args.config_path, args.result_path))


if __name__ == "__main__":
    main()
