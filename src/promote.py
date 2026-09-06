from __future__ import annotations

import argparse

import mlflow
from mlflow import MlflowClient


def promote(model_name: str, version: str, tracking_uri: str, alias: str = "champion") -> None:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    candidate = client.get_model_version(model_name, version)
    if candidate.tags.get("quality_gate") != "passed":
        raise RuntimeError(f"Model {model_name} v{version} did not pass the offline quality gate")
    client.set_registered_model_alias(model_name, alias, version)
    client.set_model_version_tag(model_name, version, "deployment_status", "promoted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--model-name", default="iris-classifier")
    parser.add_argument("--version", required=True)
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()
    promote(args.model_name, args.version, args.tracking_uri, args.alias)


if __name__ == "__main__":
    main()
