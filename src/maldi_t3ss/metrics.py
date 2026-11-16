from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests


def bootstrap_auc(y: np.ndarray, score: np.ndarray, seed: int, repeats: int) -> tuple[float, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Both classes are required for bootstrap AUC.")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats)
    for index in range(repeats):
        sample = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        estimates[index] = roc_auc_score(y[sample], score[sample])
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def metric_row(y: np.ndarray, score: np.ndarray, seed: int, repeats: int) -> dict:
    low, high = bootstrap_auc(y, score, seed, repeats)
    _, pvalue = mannwhitneyu(score[y == 1], score[y == 0], alternative="greater")
    return {
        "n": len(y),
        "positive_n": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)),
        "roc_auc_ci_low": float(low),
        "roc_auc_ci_high": float(high),
        "pr_auc": float(average_precision_score(y, score)),
        "pr_baseline": float(y.mean()),
        "auc_one_sided_p": float(pvalue),
    }


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    result = np.full(len(values), np.nan)
    keep = np.isfinite(values)
    if keep.any():
        result[keep] = multipletests(values[keep], method="fdr_bh")[1]
    return result


def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, score)
    valid = np.isfinite(thresholds)
    objective = tpr[valid] - fpr[valid]
    winners = np.flatnonzero(objective == objective.max())
    return float(np.median(thresholds[valid][winners]))

