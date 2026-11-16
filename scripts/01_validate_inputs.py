from __future__ import annotations

import argparse
import json

from maldi_t3ss.config import load_config
from maldi_t3ss.data import input_paths, load_study
from maldi_t3ss.pipeline import output_directory
from maldi_t3ss.provenance import write_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, base = load_config(args.config)
    study = load_study(config, base)
    columns = config["columns"]
    isolate_id = columns["isolate_id"]
    center = columns["center"]
    counts = study.spectrum_map.groupby(isolate_id).size()
    summary = {
        "isolates": len(study.table),
        "spectra": len(study.spectrum_map),
        "grid_length": int(study.isolate_spectra.shape[1]),
        "centers": study.table[center].value_counts().to_dict(),
        "replicates_per_isolate": {
            "minimum": int(counts.min()), "median": float(counts.median()), "maximum": int(counts.max())
        },
    }
    output = output_directory(config, base)
    (output / "input_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_provenance(output / "provenance_inputs.json", config, input_paths(config, base))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

