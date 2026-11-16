from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_path


@dataclass(frozen=True)
class StudyData:
    table: pd.DataFrame
    isolate_spectra: np.ndarray
    replicate_spectra: np.ndarray
    spectrum_map: pd.DataFrame


def _unique(table: pd.DataFrame, column: str, name: str) -> None:
    if column not in table:
        raise ValueError(f"{name} is missing required column {column!r}.")
    if table[column].isna().any() or table[column].astype(str).str.len().eq(0).any():
        raise ValueError(f"{name}.{column} contains missing/empty IDs.")
    if table[column].duplicated().any():
        examples = table.loc[table[column].duplicated(False), column].astype(str).head().tolist()
        raise ValueError(f"{name}.{column} is not unique; examples: {examples}")


def load_study(config: dict[str, Any], base: Path, mmap: bool = True) -> StudyData:
    c = config["columns"]
    isolate_id = c["isolate_id"]
    center = c["center"]
    spectrum_id = c["spectrum_id"]

    metadata_path = resolve_path(config, base, "metadata")
    genes_path = resolve_path(config, base, "vfdb_presence")
    mlst_path = resolve_path(config, base, "mlst")
    map_path = resolve_path(config, base, "spectrum_map")
    isolate_path = resolve_path(config, base, "isolate_spectra")
    replicate_path = resolve_path(config, base, "replicate_spectra")

    metadata = pd.read_csv(metadata_path, dtype={isolate_id: str})
    genes = pd.read_csv(genes_path, dtype={isolate_id: str})
    mlst = pd.read_csv(mlst_path, dtype=str)
    mapping = pd.read_csv(map_path, dtype={isolate_id: str, spectrum_id: str})
    _unique(metadata, isolate_id, "metadata")
    _unique(genes, isolate_id, "vfdb_presence")
    _unique(mlst, isolate_id, "mlst")
    _unique(mapping, spectrum_id, "spectrum_to_isolate")
    if center not in metadata:
        raise ValueError(f"metadata is missing center column {center!r}.")

    expected = set(metadata[isolate_id])
    for name, table in [("vfdb_presence", genes), ("mlst", mlst)]:
        observed = set(table[isolate_id])
        if observed != expected:
            raise ValueError(
                f"{name} isolate IDs differ from metadata: "
                f"missing={sorted(expected-observed)[:5]}, extra={sorted(observed-expected)[:5]}"
            )
    unknown = set(mapping[isolate_id]) - expected
    if unknown:
        raise ValueError(f"Spectrum mapping contains unknown isolate IDs: {sorted(unknown)[:5]}")
    absent = expected - set(mapping[isolate_id])
    if absent:
        raise ValueError(f"Isolates without spectra: {sorted(absent)[:5]}")

    table = metadata.merge(genes, on=isolate_id, validate="one_to_one").merge(
        mlst, on=isolate_id, validate="one_to_one", suffixes=("", "_mlst")
    )
    mode = "r" if mmap else None
    isolate_spectra = np.load(isolate_path, mmap_mode=mode)
    replicate_spectra = np.load(replicate_path, mmap_mode=mode)
    grid = int(config["analysis"]["expected_grid_length"])
    if isolate_spectra.shape != (len(table), grid):
        raise ValueError(
            f"isolate_spectra has shape {isolate_spectra.shape}; expected {(len(table), grid)}. "
            "Rows must follow metadata.csv order."
        )
    if replicate_spectra.shape != (len(mapping), grid):
        raise ValueError(
            f"replicate_spectra has shape {replicate_spectra.shape}; expected {(len(mapping), grid)}."
        )
    if not np.isfinite(isolate_spectra).all() or not np.isfinite(replicate_spectra).all():
        raise ValueError("Spectrum matrices contain NaN or infinite values.")
    return StudyData(table, isolate_spectra, replicate_spectra, mapping)


def input_paths(config: dict[str, Any], base: Path) -> list[Path]:
    return [
        resolve_path(config, base, key)
        for key in [
            "metadata", "spectrum_map", "isolate_spectra", "replicate_spectra",
            "vfdb_presence", "mlst",
        ]
    ]

