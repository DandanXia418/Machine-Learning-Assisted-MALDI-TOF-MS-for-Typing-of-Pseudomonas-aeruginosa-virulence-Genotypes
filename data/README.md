# Data contract

All IDs must be pseudonymous. Rows in `metadata.csv`, `isolate_median_spectra.npy`, `vfdb_presence.csv`, and `mlst.csv` must describe the same isolates in the same explicit ID order. Code joins tables by ID and checks array dimensions; it never relies on hospital row order alone.

## `metadata.csv`

Required columns:

| column | type | description |
|---|---|---|
| `isolate_id` | string | unique pseudonymous isolate identifier |
| `center` | string | instrument/centre label, e.g. `Autobio` or `Bruker` |
| `collection_date` | ISO date, optional | deidentified collection date for temporal sensitivity analysis |

## `spectrum_to_isolate.csv`

Required columns: `spectrum_id`, `isolate_id`. Row `i` maps row `i` of `replicate_spectra.npy` to an isolate. Spectrum IDs must be unique. Each isolate should ordinarily have ten technical replicates; deviations are reported rather than silently removed.

## Spectrum arrays

`isolate_median_spectra.npy` has shape `(n_isolates, 17999)` and contains the isolate median of preprocessed 1-Da intensities from m/z 2001 through 19999. `replicate_spectra.npy` has shape `(n_spectra, 17999)` and contains corresponding technical replicate spectra.

The raw TXT-to-grid preprocessing must be documented separately with baseline correction, smoothing, total-ion-current normalisation, alignment/tolerance, and exclusion rules. If raw spectra cannot be shared, provide checksums and a detailed preprocessing log.

## `vfdb_presence.csv`

One row per isolate. The first column is `isolate_id`; remaining columns are binary gene presence calls. The archived analysis used VFDB Set A *P. aeruginosa* proteins and `tblastn`, with amino-acid identity >=80%, query coverage >=80%, and E-value <=1e-10. The database release/access date and query accession must be archived.

## `mlst.csv`

One row per isolate with `isolate_id` and the seven allele columns `acsA`, `aroE`, `guaA`, `mutL`, `nuoD`, `ppsA`, `trpE`. Missing allele calls are kept missing and are not treated as equal to a numbered allele.

## `core_snp_clusters.csv` (optional)

One row per isolate with `isolate_id` and columns such as `core_snp_cluster_500`, `core_snp_cluster_1000`, `core_snp_cluster_2500`, `core_snp_cluster_5000`, and `core_snp_cluster_10000`.

## Public release checklist

- remove direct identifiers and local absolute paths;
- confirm institutional data-sharing approval;
- use Git LFS or a data repository for large arrays;
- publish SHA-256 hashes and exact database versions;
- retain all eligible endpoint results, including negative findings.

The archived screen used at least 10 positive and 10 negative isolates in each centre for exploratory modelling. A stricter flag of at least 15 per class per centre was retained for the primary-count threshold.

