from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from maldi_t3ss.config import load_config
from maldi_t3ss.features import fit_predict_ensemble, subset_spaces
from maldi_t3ss.metrics import bh_fdr, metric_row
from maldi_t3ss.pipeline import load_analysis, output_directory
from maldi_t3ss.validation import allele_cluster_groups, grouped_oof


def evaluate_task(label, y, table, spaces, groups, config, seed):
    c, a = config["columns"], config["analysis"]
    center = table[c["center"]].astype(str).to_numpy()
    uid = table[c["isolate_id"]].astype(str).to_numpy()
    source, target = config["source_center"], config["target_center"]
    rows, predictions = [], []
    for offset, site in enumerate((source, target), 1):
        keep = center == site
        score, _ = grouped_oof(
            subset_spaces(spaces, keep), y[keep], groups[keep], int(a["n_splits"]), 1,
            int(a["feature_count"]), float(a["reliability_quantile"]), float(a["svm_c"]), seed + offset * 1000,
        )
        metric = metric_row(y[keep], score, seed + offset, int(a["bootstrap_repeats"]))
        rows.append({"endpoint": label, "scenario": f"within_{site}", **metric})
        predictions.extend(
            {"endpoint": label, "scenario": f"within_{site}", "isolate_id": ident, "truth": int(truth), "score": float(value)}
            for ident, truth, value in zip(uid[keep], y[keep], score)
        )
    for offset, (train_site, test_site) in enumerate(((source, target), (target, source)), 1):
        train = np.flatnonzero(center == train_site)
        test = np.flatnonzero(center == test_site)
        score, _ = fit_predict_ensemble(
            spaces, y, train, test, int(a["feature_count"]), float(a["reliability_quantile"]),
            float(a["svm_c"]), seed + offset * 10000,
        )
        metric = metric_row(y[test], score, seed + 100 + offset, int(a["bootstrap_repeats"]))
        scenario = f"{train_site}_to_{test_site}"
        rows.append({"endpoint": label, "scenario": scenario, **metric})
        predictions.extend(
            {"endpoint": label, "scenario": scenario, "isolate_id": ident, "truth": int(truth), "score": float(value)}
            for ident, truth, value in zip(uid[test], y[test], score)
        )
    return rows, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, base = load_config(args.config)
    study, spaces_all = load_analysis(config, base)
    output = output_directory(config, base)
    landscape = pd.read_csv(output / "virulence_landscape.csv")
    patterns = pd.read_csv(output / "equivalent_gene_patterns.csv")
    eligible = set(landscape.loc[landscape["model_eligible"], "gene"])
    tasks = []
    for row in patterns.itertuples(index=False):
        members = str(row.genes).split(";")
        representative = next((gene for gene in members if gene in eligible), None)
        if representative:
            tasks.append((representative, np.ones(len(study.table), dtype=bool)))
    exou, exos = config["columns"]["exo_u"], config["columns"]["exo_s"]
    exclusive = study.table[exou].ne(study.table[exos]).to_numpy()
    tasks.append(("T3SS_exoU_only_vs_exoS_only", exclusive))

    allele_columns = config["columns"]["mlst_alleles"]
    all_groups = allele_cluster_groups(study.table[allele_columns].fillna("").astype(str).to_numpy(), 2)
    all_rows, all_predictions = [], []
    for number, (label, keep) in enumerate(tasks, 1):
        table = study.table.loc[keep].reset_index(drop=True)
        spaces = subset_spaces(spaces_all, keep)
        groups = all_groups[keep]
        y = table[exou].astype(int).to_numpy() if label.startswith("T3SS_") else table[label].astype(int).to_numpy()
        rows, predictions = evaluate_task(
            label, y, table, spaces, groups, config, int(config["seed"]) + number * 10000
        )
        all_rows.extend(rows)
        all_predictions.extend(predictions)
        print(f"[{number}/{len(tasks)}] {label}", flush=True)
    result = pd.DataFrame(all_rows)
    result["fdr"] = np.nan
    for _, indices in result.groupby("scenario").groups.items():
        result.loc[indices, "fdr"] = bh_fdr(result.loc[indices, "auc_one_sided_p"].to_numpy())
    result.to_csv(output / "endpoint_screen_metrics.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(output / "endpoint_screen_predictions.csv", index=False)


if __name__ == "__main__":
    main()

