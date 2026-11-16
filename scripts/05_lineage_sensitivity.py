from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from maldi_t3ss.config import load_config, resolve_path
from maldi_t3ss.features import fit_predict_ensemble, subset_spaces
from maldi_t3ss.metrics import metric_row
from maldi_t3ss.pipeline import load_analysis, output_directory
from maldi_t3ss.validation import grouped_oof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, base = load_config(args.config)
    cluster_path = resolve_path(config, base, "core_snp_clusters")
    clusters = pd.read_csv(cluster_path, dtype=str)
    study, spaces_all = load_analysis(config, base)
    c, a = config["columns"], config["analysis"]
    isolate_id, center = c["isolate_id"], c["center"]
    merged = study.table[[isolate_id]].merge(clusters, on=isolate_id, validate="one_to_one")
    exou, exos = c["exo_u"], c["exo_s"]
    exclusive = study.table[exou].ne(study.table[exos]).to_numpy()
    table = study.table.loc[exclusive].reset_index(drop=True)
    spaces = subset_spaces(spaces_all, exclusive)
    y = table[exou].astype(int).to_numpy()
    centers = table[center].astype(str).to_numpy()
    merged = merged.loc[exclusive].reset_index(drop=True)
    source = np.flatnonzero(centers == config["source_center"])
    target = np.flatnonzero(centers == config["target_center"])
    rows = []
    sensitivity_seed = int(config["seed"]) + 1
    endpoint_number = 3  # exoU-only versus exoS-only followed two individual-gene sensitivity endpoints
    external_score, _ = fit_predict_ensemble(
        spaces, y, source, target, int(a["feature_count"]), float(a["reliability_quantile"]),
        float(a["svm_c"]), sensitivity_seed + endpoint_number * 50000,
    )
    for number, threshold in enumerate(a["core_snp_thresholds"], 1):
        column = f"core_snp_cluster_{threshold}"
        if column not in merged:
            raise ValueError(f"Missing {column!r} in {cluster_path}")
        groups = merged[column].astype(str).to_numpy()
        for offset, site in enumerate((config["source_center"], config["target_center"]), 1):
            keep = centers == site
            score, _ = grouped_oof(
                subset_spaces(spaces, keep), y[keep], groups[keep], int(a["n_splits"]), 1,
                int(a["feature_count"]), float(a["reliability_quantile"]), float(a["svm_c"]),
                sensitivity_seed + endpoint_number * 100000 + number * 10000 + offset * 1000,
            )
            rows.append(
                {
                    "snp_threshold": threshold,
                    "scenario": f"within_{site}",
                    "lineage_groups_n": len(np.unique(groups[keep])),
                    **metric_row(y[keep], score, int(config["seed"]) + number + offset, int(a["bootstrap_repeats"])),
                }
            )
        unseen_mask = ~np.isin(groups[target], np.unique(groups[source]))
        if unseen_mask.any() and len(np.unique(y[target][unseen_mask])) == 2:
            rows.append(
                {
                    "snp_threshold": threshold,
                    "scenario": "target_unseen_source_snp_cluster",
                    "lineage_groups_n": len(np.unique(groups[target][unseen_mask])),
                    **metric_row(
                        y[target][unseen_mask], external_score[unseen_mask], sensitivity_seed + endpoint_number * 2000 + number * 100,
                        int(a["bootstrap_repeats"]),
                    ),
                }
            )
        mixed = pd.DataFrame({"group": groups, "label": y}).groupby("group")["label"].nunique()
        rows.append(
            {
                "snp_threshold": threshold,
                "scenario": "label_lineage_composition",
                "lineage_groups_n": len(mixed),
                "mixed_label_groups_n": int((mixed > 1).sum()),
                "n": len(y),
                "positive_n": int(y.sum()),
            }
        )
    result = pd.DataFrame(rows)
    output = output_directory(config, base)
    result.to_csv(output / "core_snp_lineage_sensitivity.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

