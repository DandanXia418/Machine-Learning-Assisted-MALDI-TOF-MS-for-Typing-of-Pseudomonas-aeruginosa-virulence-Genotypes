from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class FeatureSpace:
    name: str
    values: np.ndarray
    repeat_mean: np.ndarray
    repeat_within: np.ndarray
    widths: np.ndarray
    starts: np.ndarray


def l2(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float32)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def pool(x: np.ndarray, width: int) -> np.ndarray:
    usable = x.shape[1] // width * width
    return np.asarray(x[:, :usable].reshape(len(x), -1, width).max(axis=2), dtype=np.float32)


def tied_rank_l2(x: np.ndarray) -> np.ndarray:
    z = rankdata(x, axis=1, method="average").astype(np.float32)
    z -= z.mean(axis=1, keepdims=True)
    return l2(z)


def repeat_moments(values: np.ndarray, isolate_index: np.ndarray, n_isolates: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(isolate_index, minlength=n_isolates).astype(np.float32)
    sums = np.zeros((n_isolates, values.shape[1]), dtype=np.float32)
    squares = np.zeros_like(sums)
    np.add.at(sums, isolate_index, values)
    np.add.at(squares, isolate_index, values * values)
    means = sums / np.maximum(counts[:, None], 1)
    within = squares / np.maximum(counts[:, None], 1) - means * means
    return means, np.maximum(within, 0)


def _coordinates(length: int, width: int, mz_start: int) -> tuple[np.ndarray, np.ndarray]:
    n = length // width
    return np.full(n, width, dtype=int), mz_start + width * np.arange(n, dtype=int)


def build_feature_spaces(
    isolate_spectra: np.ndarray,
    replicate_spectra: np.ndarray,
    spectrum_isolate_index: np.ndarray,
    mz_start: int,
) -> dict[str, FeatureSpace]:
    isolate = np.asarray(isolate_spectra, dtype=np.float32)
    replicate = np.asarray(replicate_spectra, dtype=np.float32)
    n_isolates = len(isolate)

    i5 = l2(pool(isolate, 5))
    i10 = tied_rank_l2(pool(isolate, 10))
    i20 = l2(pool(isolate, 20))
    multi = l2(np.concatenate([i5, i10, i20], axis=1))

    r10 = tied_rank_l2(pool(replicate, 10))
    mean10, within10 = repeat_moments(r10, spectrum_isolate_index, n_isolates)
    r5 = l2(pool(replicate, 5))
    r20 = l2(pool(replicate, 20))
    rmulti = l2(np.concatenate([r5, r10, r20], axis=1))
    meanmulti, withinmulti = repeat_moments(rmulti, spectrum_isolate_index, n_isolates)

    w5, s5 = _coordinates(isolate.shape[1], 5, mz_start)
    w10, s10 = _coordinates(isolate.shape[1], 10, mz_start)
    w20, s20 = _coordinates(isolate.shape[1], 20, mz_start)
    return {
        "rank10": FeatureSpace("rank10", i10, mean10, within10, w10, s10),
        "multiscale": FeatureSpace(
            "multiscale", multi, meanmulti, withinmulti,
            np.concatenate([w5, w10, w20]), np.concatenate([s5, s10, s20]),
        ),
    }


def subset_spaces(spaces: dict[str, FeatureSpace], keep: np.ndarray) -> dict[str, FeatureSpace]:
    return {
        name: FeatureSpace(
            space.name, space.values[keep], space.repeat_mean[keep], space.repeat_within[keep],
            space.widths, space.starts,
        )
        for name, space in spaces.items()
    }


def feature_order(
    space: FeatureSpace,
    y: np.ndarray,
    train: np.ndarray,
    reliability_quantile: float,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        score, _ = f_classif(space.values[train], y[train])
    score[~np.isfinite(score)] = -np.inf
    between = np.var(space.repeat_mean[train], axis=0)
    within = np.mean(space.repeat_within[train], axis=0)
    positive_within = within[within > 0]
    stabilizer = np.median(positive_within) * 1e-3 if len(positive_within) else 1e-12
    reliability = between / (within + stabilizer + 1e-12)
    allowed = reliability >= np.quantile(reliability, reliability_quantile)
    index = np.flatnonzero(allowed)
    return index[np.argsort(score[index], kind="mergesort")[::-1]]


def fit_predict_member(
    space: FeatureSpace,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    n_features: int,
    reliability_quantile: float,
    c: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = feature_order(space, y, train, reliability_quantile)[:n_features]
    if len(selected) == 0:
        raise RuntimeError("No features survived training-set reliability filtering.")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(space.values[train][:, selected])
    x_test = scaler.transform(space.values[test][:, selected])
    model = SVC(C=c, kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed)
    model.fit(x_train, y[train])
    train_score = model.decision_function(x_train)
    test_score = model.decision_function(x_test)
    calibrated = expit((test_score - train_score.mean()) / max(train_score.std(), 1e-6))
    return calibrated, selected


def fit_predict_ensemble(
    spaces: dict[str, FeatureSpace],
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    n_features: int,
    reliability_quantile: float,
    c: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    predictions, selected = [], {}
    for offset, name in enumerate(("rank10", "multiscale"), 1):
        pred, features = fit_predict_member(
            spaces[name], y, train, test, n_features, reliability_quantile, c, seed + offset
        )
        predictions.append(pred)
        selected[name] = features
    return np.mean(predictions, axis=0), selected

