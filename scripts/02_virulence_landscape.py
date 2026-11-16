from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from maldi_t3ss.config import load_config
from maldi_t3ss.data import load_study
from maldi_t3ss.metrics import bh_fdr
from maldi_t3ss.pipeline import output_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, base = load_config(args.config)
    study = load_study(config, base)
    table = study.table
    c = config["columns"]
    isolate_id, center = c["isolate_id"], c["center"]
    excluded = set(table.columns[: table.columns.get_loc(center) + 1])
    excluded.update(c["mlst_alleles"])
    excluded.update({c.get("collection_date", "collection_date")})
    genes = [
        column for column in table.columns
        if column not in excluded and pd.to_numeric(table[column], errors="coerce").dropna().isin([0, 1]).all()
    ]
    centers = [config["source_center"], config["target_center"]]
    if set(centers) - set(table[center]):
        raise ValueError(f"Configured centres {centers} are not both present in {center}.")
    rows = []
    for gene in genes:
        y = table[gene].astype(int)
        counts = {}
        for site in centers:
            local = y[table[center].eq(site)]
            counts[site] = (int(local.sum()), int(len(local) - local.sum()))
        odds, pvalue = fisher_exact([counts[centers[0]], counts[centers[1]]])
        rows.append(
            {
                "gene": gene,
                "total_n": len(y),
                "total_positive_n": int(y.sum()),
                "total_prevalence": float(y.mean()),
                f"{centers[0]}_positive_n": counts[centers[0]][0],
                f"{centers[0]}_negative_n": counts[centers[0]][1],
                f"{centers[1]}_positive_n": counts[centers[1]][0],
                f"{centers[1]}_negative_n": counts[centers[1]][1],
                "center_odds_ratio": odds,
                "center_fisher_p": pvalue,
            }
        )
    result = pd.DataFrame(rows)
    result["center_fdr"] = bh_fdr(result["center_fisher_p"].to_numpy())
    minimum = int(config["analysis"]["min_class_per_center"])
    primary_minimum = int(config["analysis"].get("primary_min_class_per_center", 15))
    result["model_eligible"] = np.logical_and.reduce(
        [result[f"{site}_{label}_n"].ge(minimum) for site in centers for label in ("positive", "negative")]
    )
    result["primary_count_eligible"] = np.logical_and.reduce(
        [result[f"{site}_{label}_n"].ge(primary_minimum) for site in centers for label in ("positive", "negative")]
    )

    patterns = defaultdict(list)
    for gene in genes:
        patterns[bytes(table[gene].astype(np.uint8).to_numpy())].append(gene)
    group_map = {}
    group_rows = []
    for number, members in enumerate(patterns.values(), 1):
        group = f"PATTERN_{number:03d}"
        for gene in members:
            group_map[gene] = group
        group_rows.append({"pattern_group": group, "representative": members[0], "genes": ";".join(members)})
    result["pattern_group"] = result["gene"].map(group_map)
    output = output_directory(config, base)
    result.sort_values(["model_eligible", "total_prevalence"], ascending=[False, False]).to_csv(
        output / "virulence_landscape.csv", index=False
    )
    pd.DataFrame(group_rows).to_csv(output / "equivalent_gene_patterns.csv", index=False)
    print(f"genes={len(result)}, eligible={int(result.model_eligible.sum())}, independent_patterns={len(patterns)}")


if __name__ == "__main__":
    main()

