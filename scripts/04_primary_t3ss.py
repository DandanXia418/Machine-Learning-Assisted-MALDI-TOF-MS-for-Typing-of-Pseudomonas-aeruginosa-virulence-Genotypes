from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from maldi_t3ss.config import load_config
from maldi_t3ss.features import fit_predict_ensemble, subset_spaces
from maldi_t3ss.metrics import metric_row, youden_threshold
from maldi_t3ss.pipeline import load_analysis, output_directory
from maldi_t3ss.validation import allele_cluster_groups, grouped_oof


def classification_metrics(y, score, threshold):
    predicted = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, base = load_config(args.config)
    study, spaces_all = load_analysis(config, base)
    c, a = config["columns"], config["analysis"]
    exou, exos = c["exo_u"], c["exo_s"]
    exclusive = study.table[exou].ne(study.table[exos]).to_numpy()
    table = study.table.loc[exclusive].reset_index(drop=True)
    spaces = subset_spaces(spaces_all, exclusive)
    y = table[exou].astype(int).to_numpy()
    centers = table[c["center"]].astype(str).to_numpy()
    uid = table[c["isolate_id"]].astype(str).to_numpy()
    alleles = study.table[c["mlst_alleles"]].fillna("").astype(str).to_numpy()
    groups = allele_cluster_groups(alleles, 2)[exclusive]
    source = np.flatnonzero(centers == config["source_center"])
    target = np.flatnonzero(centers == config["target_center"])
    if not len(source) or not len(target):
        raise ValueError("Source and target centres must both be present in the exclusive T3SS subset.")

    source_oof, folds = grouped_oof(
        subset_spaces(spaces, centers == config["source_center"]), y[source], groups[source],
        int(a["n_splits"]), int(a["n_repeats"]), int(a["feature_count"]),
        float(a["reliability_quantile"]), float(a["svm_c"]), int(config["seed"]) + 1000,
    )
    locked_threshold = youden_threshold(y[source], source_oof)
    target_score, selected = fit_predict_ensemble(
        spaces, y, source, target, int(a["feature_count"]), float(a["reliability_quantile"]),
        float(a["svm_c"]), int(config["seed"]) + 2000,
    )
    target_posthoc_threshold = youden_threshold(y[target], target_score)
    unseen = target[~np.isin(groups[target], np.unique(groups[source]))]
    target_lookup = dict(zip(target, target_score))
    unseen_score = np.asarray([target_lookup[index] for index in unseen])

    rows = []
    for number, (name, truth, score) in enumerate(
        [
            ("source_grouped_oof", y[source], source_oof),
            ("target_cross_center", y[target], target_score),
            ("target_unseen_mlst_cluster", y[unseen], unseen_score),
        ], 1
    ):
        if len(np.unique(truth)) < 2:
            continue
        rows.append(
            {
                "dataset": name,
                **metric_row(truth, score, int(config["seed"]) + number, int(a["bootstrap_repeats"])),
                **classification_metrics(truth, score, locked_threshold),
            }
        )
    metrics = pd.DataFrame(rows)

    predictions = []
    for name, indices, scores in [
        ("source_grouped_oof", source, source_oof),
        ("target_cross_center", target, target_score),
    ]:
        predictions.extend(
            {
                "dataset": name,
                "isolate_id": uid[index],
                "lineage_group": groups[index],
                "unseen_in_source": bool(index in set(unseen)),
                "truth": int(y[index]),
                "class": "exoU-only" if y[index] else "exoS-only",
                "score": float(score),
                "prediction_locked_threshold": int(score >= locked_threshold),
            }
            for index, score in zip(indices, scores)
        )
    threshold_diagnostics = pd.DataFrame(
        [
            {
                "threshold_type": "source_locked_youden",
                **classification_metrics(y[target], target_score, locked_threshold),
                "validity": "primary target-centre threshold diagnostic",
            },
            {
                "threshold_type": "target_posthoc_youden",
                **classification_metrics(y[target], target_score, target_posthoc_threshold),
                "validity": "post-hoc platform-shift diagnostic only",
            },
        ]
    )
    selected_rows = []
    for space_name, indices in selected.items():
        space = spaces[space_name]
        for rank, feature in enumerate(indices, 1):
            selected_rows.append(
                {
                    "space": space_name,
                    "rank": rank,
                    "feature_index": int(feature),
                    "mz_start": int(space.starts[feature]),
                    "mz_end": int(space.starts[feature] + space.widths[feature] - 1),
                    "width_da": int(space.widths[feature]),
                }
            )
    output = output_directory(config, base)
    metrics.to_csv(output / "primary_t3ss_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "primary_t3ss_predictions.csv", index=False)
    pd.DataFrame(folds).to_csv(output / "primary_t3ss_source_folds.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output / "primary_t3ss_selected_features.csv", index=False)
    threshold_diagnostics.to_csv(output / "primary_t3ss_threshold_diagnostics.csv", index=False)
    print(metrics.to_string(index=False))
    print(threshold_diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()

