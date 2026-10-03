import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from iris_pipeline.contracts.model_result import (
    ModelMetrics,
    ModelResult,
    write_argo_parameters,
    write_model_result,
)


def valid_result(**overrides) -> ModelResult:
    values = {
        "model_name": "iris-classifier",
        "model_version": "12",
        "baseline_model_version": "11",
        "quality_gate_passed": True,
        "bootstrap_required": False,
        "run_id": "run-123",
        "dataset_version": "etag-456",
        "source_repository": "chiendz11/iris-data-pipeline",
        "source_sha": "abc1234",
        "metrics": ModelMetrics(accuracy=0.96, f1_macro=0.96, baseline_accuracy=0.95),
    }
    values.update(overrides)
    return ModelResult(**values)


def test_writes_versioned_contract_and_derived_argo_parameters(tmp_path):
    result = valid_result()
    contract = write_model_result(result, tmp_path / "model-result.json")
    write_argo_parameters(result, tmp_path)

    payload = json.loads(contract.read_text(encoding="utf-8"))
    schema = json.loads(Path("contracts/model-result-v1.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    assert payload["contract_version"] == "v1"
    assert payload["model_version"] == "12"
    assert payload["baseline_model_version"] == "11"
    assert payload["metrics"]["baseline_accuracy"] == 0.95
    assert (tmp_path / "model-version.txt").read_text(encoding="utf-8") == "12\n"
    assert (tmp_path / "baseline-model-version.txt").read_text(encoding="utf-8") == "11\n"
    assert (tmp_path / "quality-gate.txt").read_text(encoding="utf-8") == "true\n"
    assert (tmp_path / "bootstrap-required.txt").read_text(encoding="utf-8") == "false\n"


def test_rejects_invalid_bootstrap_contract():
    with pytest.raises(ValueError, match="quality_gate_passed"):
        valid_result(
            quality_gate_passed=False,
            bootstrap_required=True,
            baseline_model_version=None,
        )


def test_bootstrap_contract_has_no_baseline_model_version():
    result = valid_result(bootstrap_required=True, baseline_model_version=None)
    assert result.baseline_model_version is None


def test_eligible_non_bootstrap_contract_requires_baseline_model_version():
    with pytest.raises(ValueError, match="requires baseline_model_version"):
        valid_result(baseline_model_version=None)


def test_training_package_has_no_deployment_mutation_capabilities():
    source_root = Path("src/iris_pipeline")
    forbidden_names = {"smoke.py", "evaluate_canary.py", "promote.py", "gitops_rollout.py"}
    assert not {path.name for path in source_root.rglob("*.py")} & forbidden_names

    source = "\n".join(path.read_text(encoding="utf-8") for path in source_root.rglob("*.py"))
    for forbidden_token in (
        "kubectl",
        "kustomization.yaml",
        "canaryTrafficPercent",
        "GITHUB_APP_PRIVATE_KEY",
        "set_registered_model_alias",
    ):
        assert forbidden_token not in source
