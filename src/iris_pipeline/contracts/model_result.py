from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CONTRACT_VERSION = "v1"


@dataclass(frozen=True)
class ModelMetrics:
    accuracy: float
    f1_macro: float
    baseline_accuracy: float | None


@dataclass(frozen=True)
class ModelResult:
    """Deployment-neutral result consumed by the GitOps release automation."""

    model_name: str
    model_version: str | None
    baseline_model_version: str | None
    quality_gate_passed: bool
    bootstrap_required: bool
    run_id: str
    dataset_version: str
    source_repository: str
    source_sha: str
    metrics: ModelMetrics
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.contract_version}")
        if self.bootstrap_required and not self.quality_gate_passed:
            raise ValueError("bootstrap_required implies quality_gate_passed")
        if self.bootstrap_required and self.model_version is None:
            raise ValueError("bootstrap_required requires a registered model_version")
        if self.bootstrap_required and self.baseline_model_version is not None:
            raise ValueError("bootstrap release cannot have a baseline_model_version")
        if (
            self.model_version is not None
            and self.quality_gate_passed
            and not self.bootstrap_required
            and self.baseline_model_version is None
        ):
            raise ValueError(
                "an eligible non-bootstrap candidate requires baseline_model_version"
            )
        for field_name, version in (
            ("model_version", self.model_version),
            ("baseline_model_version", self.baseline_model_version),
        ):
            if version is not None and not version.isdigit():
                raise ValueError(f"{field_name} must be a numeric MLflow model version")
        if not 0.0 <= self.metrics.accuracy <= 1.0:
            raise ValueError("accuracy must be between 0 and 1")
        if not 0.0 <= self.metrics.f1_macro <= 1.0:
            raise ValueError("f1_macro must be between 0 and 1")

    def to_dict(self) -> dict:
        return asdict(self)


def write_model_result(result: ModelResult, path: str | Path) -> Path:
    """Atomically write the only cross-repository output of the training step."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def write_argo_parameters(result: ModelResult, directory: str | Path) -> None:
    """Derive scalar DAG parameters from the authoritative JSON contract.

    Argo uses these tiny files only for `when` expressions. Release automation
    must consume ``model-result.json`` rather than treating these as a contract.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    values = {
        "model-version.txt": result.model_version or "unregistered",
        "baseline-model-version.txt": result.baseline_model_version or "none",
        "quality-gate.txt": str(result.quality_gate_passed).lower(),
        "bootstrap-required.txt": str(result.bootstrap_required).lower(),
    }
    for name, value in values.items():
        (target / name).write_text(f"{value}\n", encoding="utf-8")
