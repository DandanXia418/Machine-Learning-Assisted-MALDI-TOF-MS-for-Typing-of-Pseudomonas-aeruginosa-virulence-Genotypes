"""Call VFDB protein homologues in assembled genomes with documented thresholds."""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


FIELDS = [
    "query_id", "subject_id", "pident", "length", "qstart", "qend",
    "sstart", "send", "evalue", "bitscore", "qlen", "slen",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, help="VFDB protein FASTA")
    parser.add_argument("--query-map", required=True, help="CSV: query_id,gene")
    parser.add_argument("--assemblies", required=True, help="CSV: isolate_id,assembly_path")
    parser.add_argument("--blast-bin", default="tblastn")
    parser.add_argument("--identity", type=float, default=80.0)
    parser.add_argument("--coverage", type=float, default=80.0)
    parser.add_argument("--evalue", type=float, default=1e-10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    query_map = pd.read_csv(args.query_map, dtype=str)
    assemblies = pd.read_csv(args.assemblies, dtype=str)
    required_query = {"query_id", "gene"}
    required_assembly = {"isolate_id", "assembly_path"}
    if not required_query.issubset(query_map.columns):
        raise ValueError(f"query-map requires {sorted(required_query)}")
    if not required_assembly.issubset(assemblies.columns):
        raise ValueError(f"assemblies requires {sorted(required_assembly)}")
    if query_map["query_id"].duplicated().any():
        raise ValueError("query_id must be unique in query-map")
    if assemblies["isolate_id"].duplicated().any():
        raise ValueError("isolate_id must be unique in assemblies")

    all_hits = []
    with tempfile.TemporaryDirectory(prefix="vfdb_call_") as temporary:
        for row in assemblies.itertuples(index=False):
            output = Path(temporary) / f"{row.isolate_id}.tsv"
            command = [
                args.blast_bin, "-query", args.queries, "-subject", str(row.assembly_path),
                "-evalue", str(args.evalue), "-seg", "yes", "-max_target_seqs", "100000",
                "-num_threads", str(args.threads), "-out", str(output),
                "-outfmt", "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore qlen slen",
            ]
            subprocess.run(command, check=True)
            if output.stat().st_size:
                hits = pd.read_csv(output, sep="\t", names=FIELDS)
                hits.insert(0, "isolate_id", str(row.isolate_id))
                all_hits.append(hits)
    hits = pd.concat(all_hits, ignore_index=True) if all_hits else pd.DataFrame(columns=["isolate_id", *FIELDS])
    hits["query_coverage"] = 100.0 * hits["length"] / hits["qlen"]
    passing = hits[
        hits["pident"].ge(args.identity)
        & hits["query_coverage"].ge(args.coverage)
        & hits["evalue"].le(args.evalue)
    ].merge(query_map, on="query_id", validate="many_to_one")
    positive = passing[["isolate_id", "gene"]].drop_duplicates().assign(present=1)
    matrix = positive.pivot(index="isolate_id", columns="gene", values="present").fillna(0)
    matrix = matrix.reindex(index=assemblies["isolate_id"], columns=sorted(query_map["gene"].unique()), fill_value=0)
    matrix = matrix.astype(np.uint8).reset_index()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(destination, index=False)
    hits.to_csv(destination.with_name(destination.stem + "_all_hits.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()

