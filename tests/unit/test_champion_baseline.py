from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST

from iris_pipeline.training.train import _champion_baseline


def test_missing_alias_is_the_only_bootstrap_signal() -> None:
    client = Mock()
    client.get_model_version_by_alias.side_effect = MlflowException(
        "missing alias", error_code=RESOURCE_DOES_NOT_EXIST
    )

    assert _champion_baseline(client, "iris-classifier") == (None, None)


def test_mlflow_failure_is_not_treated_as_first_bootstrap() -> None:
    client = Mock()
    client.get_model_version_by_alias.side_effect = MlflowException("server unavailable")

    with pytest.raises(MlflowException):
        _champion_baseline(client, "iris-classifier")


def test_mlflow_sql_backend_missing_alias_is_bootstrap() -> None:
    client = Mock()
    client.get_model_version_by_alias.side_effect = MlflowException(
        "Registered model alias champion not found.", error_code=INVALID_PARAMETER_VALUE
    )
    assert _champion_baseline(client, "iris-classifier") == (None, None)


def test_unrelated_invalid_parameter_is_not_bootstrap() -> None:
    client = Mock()
    client.get_model_version_by_alias.side_effect = MlflowException(
        "Invalid model name", error_code=INVALID_PARAMETER_VALUE
    )
    with pytest.raises(MlflowException):
        _champion_baseline(client, "iris-classifier")


def test_champion_must_have_accuracy_metric() -> None:
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(
        run_id="run-1", version="6"
    )
    client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(metrics={}))

    with pytest.raises(RuntimeError, match="has no accuracy metric"):
        _champion_baseline(client, "iris-classifier")


def test_returns_champion_accuracy_and_version() -> None:
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(
        run_id="run-1", version="6"
    )
    client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(metrics={"accuracy": 0.94})
    )

    assert _champion_baseline(client, "iris-classifier") == (0.94, "6")
