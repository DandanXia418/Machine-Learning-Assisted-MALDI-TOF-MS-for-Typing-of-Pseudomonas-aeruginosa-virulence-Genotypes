# Archived result checks

With the complete deidentified 216-isolate dataset and the archived configuration, the core outputs should reproduce the following values up to floating-point rounding.

## Cohort and screening

- isolates with WGS and MALDI-TOF: 216;
- source/Autobio isolates: 103;
- target/Bruker isolates: 113;
- replicate spectra: 2,150;
- VFDB gene labels: 315;
- exploratory model-eligible labels (at least 10 positive and 10 negative per centre): 54;
- independent presence/absence patterns among eligible labels: 27;
- evaluated tasks after adding the mutually exclusive T3SS endpoint: 28.

## Fixed primary endpoint

`1 = exoU-only`; `0 = exoS-only`. Five double-positive and one double-negative isolate are excluded.

| analysis | N | exoU-only | ROC AUC | PR AUC |
|---|---:|---:|---:|---:|
| source-centre lineage-grouped OOF | 102 | 39 | 0.906797 | 0.872157 |
| source to target cross-centre | 108 | 21 | 0.950739 | 0.811058 |
| target isolates from unseen MLST double-locus clusters | 67 | 19 | 0.967105 | 0.925031 |

The source-derived Youden threshold is 0.524162. Applied unchanged to the target centre, sensitivity is 0 and specificity is 1 because of platform score shift. The target-optimal post-hoc diagnostic threshold is approximately 0.432223; it must not be reported as independently validated classification performance.

## Core-SNP sensitivity

At the 5,000-SNP cluster definition, expected within-centre AUCs are approximately 0.862 for Autobio and 0.924 for Bruker. The source model evaluated on target isolates from unseen 5,000-SNP clusters has AUC approximately 0.955. No tested core-SNP cluster contains both `exoU-only` and `exoS-only` isolates, so genotype and deep population structure cannot be fully separated in this cohort.

## Stable spectral intervals

The archived selection analysis returns 261 features selected in the final source model and in at least 40% of 25 lineage-grouped training folds. Overlapping 5-, 10-, and 20-Da features merge into 64 candidate m/z regions. These are interval features and must not be assigned to specific proteins without orthogonal identification.

