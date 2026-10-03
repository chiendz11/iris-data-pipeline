"""Versioned output contracts published by the training pipeline."""

from iris_pipeline.contracts.model_result import ModelResult, write_model_result

__all__ = ["ModelResult", "write_model_result"]
