"""Convert vendor-exported two-column TXT spectra to the fixed-grid matrices used by the study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d


def load_txt(path: Path, grid: np.ndarray, baseline_window: int, smooth_sigma: float) -> np.ndarray:
    with path.open("rb") as handle:
        first = handle.readline().lower()
    skiprows = 1 if b"m/z" in first or b"intensity" in first else 0
    values = np.loadtxt(path, dtype=np.float64, skiprows=skiprows)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Expected at least two numeric columns: m/z and intensity.")
    mz = values[:, 0]
    intensity = np.maximum(values[:, 1], 0.0)
    keep = np.isfinite(mz) & np.isfinite(intensity)
    mz, intensity = mz[keep], intensity[keep]
    order = np.argsort(mz)
    mz, intensity = mz[order], intensity[order]
    baseline = maximum_filter1d(
        minimum_filter1d(intensity, size=baseline_window, mode="nearest"),
        size=baseline_window,
        mode="nearest",
    )
    corrected = np.maximum(intensity - baseline, 0.0)
    corrected = gaussian_filter1d(corrected, sigma=smooth_sigma, mode="nearest")
    transformed = np.sqrt(corrected)
    interpolated = np.interp(grid, mz, transformed, left=0.0, right=0.0)
    norm = np.linalg.norm(interpolated)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Spectrum is empty or non-finite after preprocessing.")
    return (interpolated / norm).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV: spectrum_id,isolate_id,center,txt_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mz-start", type=int, default=2001)
    parser.add_argument("--mz-stop", type=int, default=20000, help="exclusive upper limit")
    parser.add_argument("--baseline-window", type=int, default=401)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype=str)
    required = {"spectrum_id", "isolate_id", "center", "txt_path"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest requires columns {sorted(required)}")
    if manifest["spectrum_id"].duplicated().any():
        raise ValueError("spectrum_id values must be unique.")
    if manifest[["spectrum_id", "isolate_id", "center", "txt_path"]].isna().any().any():
        raise ValueError("Manifest contains missing required values.")
    inconsistent = manifest.groupby("isolate_id")["center"].nunique()
    if inconsistent.gt(1).any():
        raise ValueError("An isolate is assigned to more than one centre.")

    grid = np.arange(args.mz_start, args.mz_stop, dtype=np.float64)
    replicate = np.empty((len(manifest), len(grid)), dtype=np.float32)
    failures = []
    for index, row in enumerate(manifest.itertuples(index=False)):
        try:
            replicate[index] = load_txt(
                Path(row.txt_path), grid, args.baseline_window, args.smooth_sigma
            )
        except Exception as error:
            failures.append({"spectrum_id": row.spectrum_id, "error": repr(error)})
    if failures:
        destination = Path(args.output)
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(destination / "preprocessing_failures.csv", index=False)
        raise RuntimeError(f"{len(failures)} spectra failed preprocessing.")

    isolate_ids = sorted(manifest["isolate_id"].unique())
    isolate = np.empty((len(isolate_ids), len(grid)), dtype=np.float32)
    qc = []
    for index, isolate_id in enumerate(isolate_ids):
        rows = np.flatnonzero(manifest["isolate_id"].eq(isolate_id).to_numpy())
        median = np.median(replicate[rows], axis=0)
        median /= max(np.linalg.norm(median), 1e-12)
        isolate[index] = median
        similarity = replicate[rows] @ median
        qc.append(
            {
                "isolate_id": isolate_id,
                "replicate_n": len(rows),
                "median_cosine_similarity": float(np.median(similarity)),
                "minimum_cosine_similarity": float(np.min(similarity)),
            }
        )
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "replicate_spectra.npy", replicate)
    np.save(destination / "isolate_median_spectra.npy", isolate)
    np.save(destination / "mz_grid.npy", grid.astype(np.float32))
    manifest[["spectrum_id", "isolate_id"]].to_csv(destination / "spectrum_to_isolate.csv", index=False)
    (
        manifest.drop_duplicates("isolate_id").set_index("isolate_id").loc[isolate_ids, ["center"]]
        .reset_index().to_csv(destination / "metadata.csv", index=False)
    )
    pd.DataFrame(qc).to_csv(destination / "replicate_qc.csv", index=False)
    parameters = {
        "mz_start": args.mz_start,
        "mz_stop_exclusive": args.mz_stop,
        "grid_step_da": 1,
        "baseline": f"morphological opening, window={args.baseline_window} points",
        "smoothing": f"Gaussian sigma={args.smooth_sigma} points",
        "transform": "square root",
        "normalization": "L2 per spectrum; median technical-replicate spectrum followed by L2",
        "statistical_unit": "isolate",
        "spectra_n": len(manifest),
        "isolates_n": len(isolate_ids),
    }
    (destination / "preprocessing_parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

