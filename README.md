# Cross-platform MALDI-TOF prediction of *Pseudomonas aeruginosa* T3SS virulotypes

This repository contains the analysis code for a two-centre, cross-platform study of WGS-defined virulence genotypes in *Pseudomonas aeruginosa*. The primary endpoint is the mutually exclusive type III secretion system (T3SS) virulotype: `exoU-only` versus `exoS-only`.

## Scope and claims

The code predicts **WGS-defined gene carriage**, not gene expression and not direct detection of ExoU or ExoS proteins. The MALDI-TOF acquisition range was 2,000–20,000 m/z. Cross-centre results are described as cross-centre generalisation analyses because both centres contributed to endpoint discovery and method development.

The final classifier was fixed before the primary analysis:

- one isolate is the statistical unit;
- technical replicate spectra never cross validation folds;
- replicate spectra are first aggregated into isolate-level features;
- feature filtering and univariate feature ranking are fitted using training isolates only;
- the model is an unweighted ensemble of two RBF-SVMs;
- the two feature spaces are 10-Da rank intensity and 5/10/20-Da multiscale intensity;
- each member uses the 150 highest-ranked features among the top 50% by replicate reliability;
- `C=3`, `gamma="scale"`, and balanced class weights are fixed;
- within-centre validation uses lineage-grouped folds;
- both cross-centre directions are reported;
- confidence intervals use class-stratified isolate-level bootstrap resampling.

## Repository layout

```text
config.example.yml          paths and analysis constants
src/maldi_t3ss/             reusable, tested analysis functions
scripts/                    numbered command-line analyses
tests/                      leakage and reproducibility tests
data/README.md              input schema and sharing guidance
docs/analysis_decisions.md  distinction between screening and confirmatory analyses
```

## Input data

The repository does not contain identifiable clinical data, raw hospital workbooks, or local file paths. Prepare the files described in [data/README.md](data/README.md). Large matrices should be deposited in an appropriate controlled repository or release archive rather than committed directly to GitHub.

Required prepared inputs are:

1. `metadata.csv`: one row per isolate;
2. `spectrum_to_isolate.csv`: one row per technical replicate spectrum;
3. `isolate_median_spectra.npy`: isolate-level median spectra on the 1-Da grid;
4. `replicate_spectra.npy`: preprocessed replicate spectra on the same grid;
5. `vfdb_presence.csv`: WGS gene presence/absence matrix;
6. `mlst.csv`: seven-locus MLST alleles;
7. optionally, `core_snp_clusters.csv` and deidentified collection dates.

All tables must use pseudonymous isolate IDs. Do not upload names, admission numbers, record numbers, dates of birth, or free-text clinical notes.

## Installation

Python 3.11 was used for the archived analysis.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Copy and edit the configuration file:

```bash
cp config.example.yml config.yml
```

If starting from raw two-column TXT exports, first prepare a pseudonymous manifest with `spectrum_id,isolate_id,center,txt_path`, then run:

```bash
python scripts/preprocess_spectra.py --manifest data/raw_manifest.csv --output data
```

This implements the archived fixed-grid pipeline: morphological-opening baseline subtraction, Gaussian smoothing, square-root transformation, interpolation to the 1-Da grid, L2 normalisation, and isolate-level median aggregation.

## Reproduce the main analysis

```bash
python scripts/01_validate_inputs.py --config config.yml
python scripts/02_virulence_landscape.py --config config.yml
python scripts/03_screen_endpoints.py --config config.yml
python scripts/04_primary_t3ss.py --config config.yml
python scripts/05_lineage_sensitivity.py --config config.yml
python scripts/06_stable_mz_regions.py --config config.yml
python scripts/07_temporal_bias.py --config config.yml
```

`03_screen_endpoints.py` is explicitly exploratory and reports all eligible endpoints. `04_primary_t3ss.py` runs the fixed primary endpoint and model. The target-centre labels are not used to select features, tune the model, or choose the transferable threshold.

Run the automated checks with:

```bash
python -m unittest discover -s tests -v
```

## Reproducibility safeguards

- Each script records the configuration, package versions, random seed and SHA-256 hashes of its inputs.
- Merge operations assert one-to-one isolate identity.
- The replicate mapping is checked against matrix row order and isolate IDs.
- Grouped folds are rejected if either class is absent or a lineage crosses train and validation partitions.
- The external target is evaluated only after the source-centre model has been fitted.
- The source-derived classification threshold and the target post-hoc diagnostic threshold are stored separately.
- All negative and non-significant endpoint results are retained.

## Interpretation limits

High ROC AUC does not imply that a single threshold transfers between instrument platforms. The primary analysis therefore reports ranking discrimination, PR AUC, class counts and source-locked threshold diagnostics separately. Because `exoU`/`exoS` is associated with population structure, lineage-blocked and unseen-lineage analyses are required and do not establish a direct causal protein assignment for any m/z interval.

## Citation

Please cite the associated article and this repository release. See `CITATION.cff`.

Before a public upload, run `python scripts/99_release_check.py` and then `python scripts/package_release.py`. The latter creates a clean ZIP that excludes private data, local configuration, generated results, caches and large matrices.

