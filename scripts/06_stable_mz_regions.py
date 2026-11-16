from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from maldi_t3ss.config import load_config, resolve_path
from maldi_t3ss.features import FeatureSpace, feature_order, subset_spaces
from maldi_t3ss.pipeline import load_analysis, output_directory
from maldi_t3ss.validation import grouped_splits


@dataclass
class Member:
    name: str
    selected: np.ndarray
    transformed_target: np.ndarray
    model: SVC
    train_mean: float
    train_sd: float


def fit_member(space: FeatureSpace, y, train, target, n_features, quantile, c, seed):
    selected = feature_order(space, y, train, quantile)[:n_features]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(space.values[train][:, selected])
    x_target = scaler.transform(space.values[target][:, selected])
    model = SVC(C=c, kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed)
    model.fit(x_train, y[train])
    train_score = model.decision_function(x_train)
    return Member(
        space.name, selected, x_target, model, float(train_score.mean()),
        float(max(train_score.std(), 1e-6)),
    )


def member_score(member: Member, values=None):
    matrix = member.transformed_target if values is None else values
    return expit((member.model.decision_function(matrix) - member.train_mean) / member.train_sd)


def cohen_d(values, y):
    positive, negative = values[y == 1], values[y == 0]
    pooled = np.sqrt(
        ((len(positive) - 1) * np.var(positive, ddof=1) + (len(negative) - 1) * np.var(negative, ddof=1))
        / max(len(values) - 2, 1)
    )
    return float((positive.mean() - negative.mean()) / max(pooled, 1e-12))


def merge_intervals(features: pd.DataFrame):
    ordered = features.sort_values(["mz_start", "mz_end"]).reset_index(drop=True)
    regions, members = [], []
    current_start = current_end = None
    current_members = []
    for row in ordered.itertuples(index=False):
        if current_start is None or row.mz_start > current_end + 2:
            if current_members:
                regions.append((current_start, current_end, current_members))
            current_start, current_end, current_members = row.mz_start, row.mz_end, [(row.space, row.feature_index)]
        else:
            current_end = max(current_end, row.mz_end)
            current_members.append((row.space, row.feature_index))
    if current_members:
        regions.append((current_start, current_end, current_members))
    return regions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection-repeats", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=50)
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
    source = np.flatnonzero(centers == config["source_center"])
    target = np.flatnonzero(centers == config["target_center"])

    clusters = pd.read_csv(resolve_path(config, base, "core_snp_clusters"), dtype=str)
    cluster_column = "core_snp_cluster_5000"
    cluster = study.table[[c["isolate_id"]]].merge(
        clusters[[c["isolate_id"], cluster_column]], on=c["isolate_id"], validate="one_to_one"
    )[cluster_column].astype(str).to_numpy()[exclusive]

    sensitivity_seed = int(config["seed"]) + 1
    selections = []
    for repeat in range(args.selection_repeats):
        splits = grouped_splits(y[source], cluster[source], int(a["n_splits"]), sensitivity_seed + repeat * 1000)
        for fold, (train_local, _) in enumerate(splits, 1):
            train = source[train_local]
            for name, space in spaces.items():
                chosen = feature_order(space, y, train, float(a["reliability_quantile"]))[: int(a["feature_count"])]
                for rank, feature in enumerate(chosen, 1):
                    selections.append(
                        {
                            "repeat": repeat + 1, "fold": fold, "space": name,
                            "feature_index": int(feature), "rank": rank,
                            "mz_start": int(space.starts[feature]),
                            "mz_end": int(space.starts[feature] + space.widths[feature] - 1),
                            "width_da": int(space.widths[feature]),
                        }
                    )
    selections = pd.DataFrame(selections)
    denominator = selections[["repeat", "fold"]].drop_duplicates().shape[0]
    frequency = selections.groupby(
        ["space", "feature_index", "mz_start", "mz_end", "width_da"], as_index=False
    ).agg(selected_folds=("rank", "size"), median_rank=("rank", "median"))
    frequency["selection_frequency"] = frequency["selected_folds"] / denominator

    members = {
        name: fit_member(
            space, y, source, target, int(a["feature_count"]), float(a["reliability_quantile"]),
            float(a["svm_c"]), sensitivity_seed + offset,
        )
        for offset, (name, space) in enumerate(spaces.items(), 1)
    }
    selected_final = {(name, int(feature)) for name, member in members.items() for feature in member.selected}
    frequency["selected_in_final_source_model"] = [
        (row.space, int(row.feature_index)) in selected_final for row in frequency.itertuples(index=False)
    ]
    stable = frequency[
        frequency["selected_in_final_source_model"] & frequency["selection_frequency"].ge(0.40)
    ].copy()
    effect_rows = []
    for row in stable.itertuples(index=False):
        space = spaces[row.space]
        feature = int(row.feature_index)
        source_d = cohen_d(space.values[source, feature], y[source])
        target_d = cohen_d(space.values[target, feature], y[target])
        orientation = 1 if source_d >= 0 else -1
        effect_rows.append(
            {
                **row._asdict(),
                "source_cohen_d_exou_minus_exos": source_d,
                "target_cohen_d_exou_minus_exos": target_d,
                "same_effect_direction": bool(source_d * target_d > 0),
                "target_single_feature_auc_descriptive": roc_auc_score(
                    y[target], orientation * space.values[target, feature]
                ),
            }
        )
    stable = pd.DataFrame(effect_rows)

    baseline_member_scores = {name: member_score(member) for name, member in members.items()}
    baseline = np.mean(list(baseline_member_scores.values()), axis=0)
    baseline_auc = roc_auc_score(y[target], baseline)
    rng = np.random.default_rng(sensitivity_seed)
    region_rows = []
    for region_no, (start, end, features) in enumerate(merge_intervals(stable), 1):
        permuted_auc = []
        for _ in range(args.permutations):
            scores = []
            for name, member in members.items():
                positions = [
                    position for position, feature in enumerate(member.selected)
                    if (name, int(feature)) in features
                ]
                if not positions:
                    scores.append(baseline_member_scores[name])
                    continue
                values = member.transformed_target.copy()
                for position in positions:
                    values[:, position] = values[rng.permutation(len(values)), position]
                scores.append(member_score(member, values))
            permuted_auc.append(roc_auc_score(y[target], np.mean(scores, axis=0)))
        local = stable[
            stable.apply(lambda row: (row["space"], int(row["feature_index"])) in features, axis=1)
        ]
        drops = baseline_auc - np.asarray(permuted_auc)
        region_rows.append(
            {
                "region": region_no, "mz_start": start, "mz_end": end,
                "stable_features_n": len(local),
                "maximum_selection_frequency": float(local["selection_frequency"].max()),
                "same_direction_fraction": float(local["same_effect_direction"].mean()),
                "external_ensemble_auc": baseline_auc,
                "permutation_auc_drop_descriptive": float(drops.mean()),
                "permutation_auc_drop_ci_low_descriptive": float(np.quantile(drops, 0.025)),
                "permutation_auc_drop_ci_high_descriptive": float(np.quantile(drops, 0.975)),
            }
        )
    output = output_directory(config, base)
    selections.to_csv(output / "stable_feature_fold_selections.csv", index=False)
    stable.to_csv(output / "stable_mz_features.csv", index=False)
    pd.DataFrame(region_rows).sort_values("permutation_auc_drop_descriptive", ascending=False).to_csv(
        output / "stable_mz_regions.csv", index=False
    )
    print(f"stable_features={len(stable)}, regions={len(region_rows)}, external_auc={baseline_auc:.3f}")


if __name__ == "__main__":
    main()

