# Analysis decisions and reporting boundary

## Exploratory stage

The virulence landscape and modelability screen evaluate all VFDB labels that meet the minimum class counts in both centres. Labels with identical presence/absence vectors are collapsed to one task. All eligible tasks and both cross-centre directions are retained. False-discovery-rate correction is applied within each reported evaluation family.

The `exoU-only` versus `exoS-only` endpoint emerged from this systematic screen. It must not be described as a prospectively prespecified endpoint. Because both centres contributed to screening, neither cross-centre direction is a fully untouched external validation cohort.

## Fixed primary analysis

After endpoint selection, the following pipeline was frozen: 10-Da rank-intensity plus 5/10/20-Da multiscale feature spaces; replicate-reliability filtering at the training-set median; 150 training-ranked features per space; balanced RBF-SVM with `C=3`; equal-probability ensemble.

Source-centre grouped out-of-fold predictions are used to select a Youden threshold. The target-centre labels are used only for evaluation. A target-optimal threshold may be reported as a clearly marked post-hoc platform-shift diagnostic and must not be presented as transferable classification performance.

## Population structure

MLST grouped folds and core-SNP-cluster sensitivity analyses reduce train/test lineage overlap. They cannot prove a direct genotype-specific mass spectral signal when `exoU`/`exoS` remains strongly associated with deep lineage. Stable m/z intervals are therefore described as genotype- and lineage-associated signatures until orthogonal protein identification is performed.

## Statistical unit

The isolate is the statistical unit. Technical repeat spectra are used to estimate repeat reliability and construct isolate-level summaries. Treating each spectrum as an independent sample would inflate precision and is prohibited in the primary analysis.

