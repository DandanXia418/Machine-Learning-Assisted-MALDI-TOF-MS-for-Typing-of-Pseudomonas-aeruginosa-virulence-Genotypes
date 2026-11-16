from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a YAML mapping.")
    return config, config_path.parent


def resolve_path(config: dict[str, Any], base: Path, key: str) -> Path:
    value = config["paths"][key]
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()

