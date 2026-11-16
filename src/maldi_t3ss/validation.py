from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .features import FeatureSpace, fit_predict_ensemble


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def allele_cluster_groups(alleles: np.ndarray, max_distance: int) -> np.ndarray:
    profiles = np.asarray(alleles, dtype=object)
    n = len(profiles)
    uf = UnionFind(n)
    missing = np.equal(profiles, "") | pd_isna(profiles)
    for i in range(n):
        for j in range(i + 1, n):
            if missing[i].any() or missing[j].any():
                continue
            if int(np.sum(profiles[i] != profiles[j])) <= max_distance:
                uf.union(i, j)
    roots = [uf.find(i) for i in range(n)]
    labels = {root: f"CC{max_distance}_{k:03d}" for k, root in enumerate(sorted(set(roots)), 1)}
    return np.asarray([labels[root] for root in roots])


def pd_isna(values: np.ndarray) -> np.ndarray:
    # Avoid importing pandas in the hot path while handling None and float NaN.
    return np.vectorize(lambda x: x is None or (isinstance(x, float) and np.isnan(x)))(values)


def grouped_splits(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    for n in range(min(n_splits, len(np.unique(groups))), 1, -1):
        splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y, groups))
        valid = all(
            len(np.unique(y[train])) == 2
            and len(np.unique(y[test])) == 2
            and not set(groups[train]).intersection(groups[test])
            for train, test in splits
        )
        if valid:
            return splits
    raise RuntimeError("Unable to construct class-complete, lineage-disjoint grouped folds.")


def grouped_oof(
    spaces: dict[str, FeatureSpace],
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    n_repeats: int,
    n_features: int,
    reliability_quantile: float,
    c: float,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    values: list[list[float]] = [[] for _ in range(len(y))]
    records = []
    for repeat in range(n_repeats):
        splits = grouped_splits(y, groups, n_splits, seed + repeat * 1000)
        for fold, (train, test) in enumerate(splits, 1):
            score, selected = fit_predict_ensemble(
                spaces, y, train, test, n_features, reliability_quantile, c,
                seed + repeat * 1000 + fold * 10,
            )
            for index, prediction in zip(test, score):
                values[index].append(float(prediction))
            records.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "train_n": len(train),
                    "test_n": len(test),
                    "test_positive_n": int(y[test].sum()),
                    "test_groups_n": len(np.unique(groups[test])),
                    "rank10_features": len(selected["rank10"]),
                    "multiscale_features": len(selected["multiscale"]),
                }
            )
    if any(not item for item in values):
        raise RuntimeError("Incomplete out-of-fold predictions.")
    return np.asarray([np.mean(item) for item in values]), records

