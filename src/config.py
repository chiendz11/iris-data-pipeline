from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path = "params.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"data", "model", "quality_gate"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    return config

