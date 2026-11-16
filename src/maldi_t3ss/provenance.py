from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(output: Path, config: dict, inputs: Iterable[Path]) -> None:
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "PyYAML"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    record = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "seed": config.get("seed"),
        "config": config,
        "input_sha256": {str(path): sha256(path) for path in inputs if path.exists()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

