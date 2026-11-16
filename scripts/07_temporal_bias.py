from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency

from maldi_t3ss.config import load_config
from maldi_t3ss.pipeline import load_analysis, output_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--permutations", type=int, default=20000)
    args = parser.parse_args()
    config, base = load_config(args.config)
    study, _ = load_analysis(config, base)
    output = output_directory(config, base)
    screen_path = output / "endpoint_screen_predictions.csv"
    if screen_path.exists():
        predictions = pd.read_csv(screen_path, dtype={"isolate_id": str})
        predictions = predictions[predictions["endpoint"].eq("T3SS_exoU_only_vs_exoS_only")].copy()
        predictions["dataset"] = predictions["scenario"]
        predictions["class"] = np.where(predictions["truth"].eq(1), "exoU-only", "exoS-only")
    else:
        predictions = pd.read_csv(output / "primary_t3ss_predictions.csv", dtype={"isolate_id": str})
    c = config["columns"]
    date_column = c.get("collection_date", "collection_date")
    if date_column not in study.table:
        raise ValueError(f"metadata is missing optional temporal-analysis column {date_column!r}")
    dates = study.table[[c["isolate_id"], c["center"], date_column]].copy()
    dates.columns = ["isolate_id", "center", "collection_date"]
    dates["collection_date"] = pd.to_datetime(dates["collection_date"], errors="raise")
    dates["year"] = dates["collection_date"].dt.year.astype(int)
    data = predictions.merge(dates, on="isolate_id", validate="many_to_one")

    rows = []
    for dataset, group in data.groupby("dataset", sort=False):
        design = pd.DataFrame(
            {
                "intercept": 1.0,
                "truth": group["truth"].astype(float),
                "year_centered": group["year"] - group["year"].median(),
            }
        )
        design["truth_x_year"] = design["truth"] * design["year_centered"]
        fit = sm.OLS(group["score"].astype(float), design).fit(cov_type="HC3")
        joint = fit.wald_test("year_centered = 0, truth_x_year = 0", scalar=True)
        rows.append(
            {
                "dataset": dataset,
                "n": len(group),
                "year_min": int(group["year"].min()),
                "year_max": int(group["year"].max()),
                "year_main_effect": float(fit.params["year_centered"]),
                "year_main_effect_p": float(fit.pvalues["year_centered"]),
                "label_year_interaction": float(fit.params["truth_x_year"]),
                "label_year_interaction_p": float(fit.pvalues["truth_x_year"]),
                "year_drift_joint_p": float(joint.pvalue),
            }
        )
    yearly = data.groupby(["dataset", "center", "year", "class"], as_index=False).agg(
        n=("isolate_id", "size"), mean_score=("score", "mean")
    )
    # Label prevalence trends are estimated once per centre on mutually exclusive isolates.
    label_data = data.drop_duplicates("isolate_id")
    rng = np.random.default_rng(int(config["seed"]) + 1)
    trend_rows = []
    for center, group in label_data.groupby("center", sort=False):
        centered = group["year"].astype(float) - float(group["year"].median())
        design = sm.add_constant(centered.rename("year"))
        fit = sm.GLM(group["truth"], design, family=sm.families.Binomial()).fit(cov_type="HC3")
        low, high = fit.conf_int().loc["year"].astype(float)
        contingency = pd.crosstab(group["year"], group["truth"])
        observed = float(chi2_contingency(contingency, correction=False)[0])
        greater = 0
        truth = group["truth"].to_numpy()
        year = group["year"].to_numpy()
        for _ in range(args.permutations):
            shuffled = rng.permutation(truth)
            candidate = pd.crosstab(year, shuffled).reindex(
                index=contingency.index, columns=contingency.columns, fill_value=0
            )
            greater += float(chi2_contingency(candidate, correction=False)[0]) >= observed - 1e-12
        trend_rows.append(
            {
                "center": center, "n": len(group), "year_min": int(group["year"].min()),
                "year_max": int(group["year"].max()), "exou_only_or_per_year": float(np.exp(fit.params["year"])),
                "or_ci_low": float(np.exp(low)), "or_ci_high": float(np.exp(high)),
                "trend_p": float(fit.pvalues["year"]),
                "year_label_permutation_p": (greater + 1) / (args.permutations + 1),
            }
        )
    pd.DataFrame(rows).to_csv(output / "temporal_score_drift.csv", index=False)
    pd.DataFrame(trend_rows).to_csv(output / "temporal_label_trends.csv", index=False)
    yearly.to_csv(output / "temporal_yearly_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(trend_rows).to_string(index=False))


if __name__ == "__main__":
    main()

