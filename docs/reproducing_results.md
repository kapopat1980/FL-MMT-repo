# Reproducing paper results

This document maps each config file and script to the corresponding
table/figure in the manuscript.

| Manuscript item | Config | Script |
|---|---|---|
| Table 4 (centralised performance) | `configs/centralised.yaml` | `scripts/train.py` |
| Table 5 (federated vs. centralised, privacy budget) | `configs/federated_dp.yaml` | `scripts/federate.py` |
| Table 6 (comparison with re-implemented SOTA) | `configs/eval.yaml` | `scripts/evaluate.py` |
| Figure 2 (macro-AUC comparison) | `configs/eval.yaml` | `scripts/evaluate.py` (plotting in `notebooks/figures.ipynb`) |
| Section 4.4 (MC Dropout calibration, ECE) | `configs/eval.yaml` | `src/evaluation/calibration.py` |
| Figure 5 (three-tier explainability) | — | `scripts/explain.py` |
| Section 4.5 (quantitative XAI validation) | — | `src/explainability/three_tier_xai.py` |

## Notes on reproducibility

- All random seeds are fixed via `training.seed` in each config (default: 42).
- Federated experiments use a Dirichlet(alpha=0.5) client partition by
  default; sweeping `federated.dirichlet_alpha` reproduces the
  heterogeneity-sensitivity analysis mentioned in the manuscript's limitations.
- The exact (epsilon, delta) privacy budget reported in the paper
  (epsilon=2.1, delta=1e-5) is reproduced by `configs/federated_dp.yaml`
  as configured; Opacus's accountant will compute the actual epsilon spent
  each round, printed during training.
- Checkpoints and logs are intentionally excluded from version control
  (see `.gitignore`); trained checkpoints will be released via a separate
  archive (Zenodo/institutional storage) linked here once available.

## Known gaps in this initial release

This is the code accompanying manuscript submission, and is being completed
incrementally. Current known gaps (tracked in the repository's Issues tab):

- `src/data/dataset.py`: manifest-loading logic is dataset-specific and
  requires the preprocessing scripts referenced in `docs/dataset_setup.md`,
  which are being finalized.
- `scripts/evaluate.py` and `scripts/explain.py` contain the full pipeline
  structure with core computation stubbed pending final checkpoint release.

We are actively completing these prior to/around acceptance; the core
modeling, federated learning, calibration, and explainability logic
(HCMA encoder, FedProx + DP-SGD, MC Dropout/ECE, three-tier XAI) is fully
implemented and unit-tested.
