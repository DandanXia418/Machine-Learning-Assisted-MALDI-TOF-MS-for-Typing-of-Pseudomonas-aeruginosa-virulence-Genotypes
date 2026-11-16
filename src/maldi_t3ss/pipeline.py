from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve_path
from .data import StudyData, load_study
from .features import build_feature_spaces


def load_analysis(config: dict[str, Any], base: Path):
    study = load_study(config, base)
    isolate_id = config["columns"]["isolate_id"]
    index = {value: i for i, value in enumerate(study.table[isolate_id].astype(str))}
    mapped = study.spectrum_map[isolate_id].astype(str).map(index)
    if mapped.isna().any():
        raise ValueError("Unmapped isolate IDs remain in spectrum_to_isolate.csv.")
    spaces = build_feature_spaces(
        study.isolate_spectra,
        study.replicate_spectra,
        mapped.to_numpy(dtype=int),
        int(config["analysis"]["mz_start"]),
    )
    return study, spaces


def output_directory(config: dict[str, Any], base: Path) -> Path:
    output = resolve_path(config, base, "output")
    output.mkdir(parents=True, exist_ok=True)
    return output

