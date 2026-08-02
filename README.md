# FL-MMT: Federated Multi-Modal Transformer for Longitudinal Lung Cancer Diagnosis and Prognosis

Official code repository for:

> "A Privacy-Preserving Federated Transformer with Cross-Modal Attention for Longitudinal Lung Cancer Diagnosis and Prognosis" — submitted to *Computers in Biology and Medicine*.

This repository accompanies the manuscript and provides the implementation of:

- **HCMA** — a hierarchical cross-modal attention encoder that fuses CT volumes, whole-slide histopathology, genomic mutation profiles, and structured clinical metadata.
- A **simulated federated training pipeline** (FedProx aggregation + DP-SGD client-side privacy) across 8 simulated clients.
- **Calibrated uncertainty quantification** via Monte Carlo Dropout across three task heads (classification, risk stratification, survival).
- A **three-tier explainability layer** (Integrated Gradients, attention-flow visualization, SHAP) validated quantitatively against pixel-level annotations and mutation ground truth.

## Repository structure

```
FL-MMT/
├── configs/                # YAML experiment configs (model, federated, training)
├── src/
│   ├── data/                # Dataset loaders and preprocessing for the 5 public datasets
│   ├── models/               # HCMA encoder, task heads, backbone modules
│   ├── federated/             # FedProx client/server logic, DP-SGD wrapper
│   ├── training/              # Training loops, schedulers, checkpointing
│   ├── explainability/        # Integrated Gradients, attention flow, SHAP wrappers
│   └── evaluation/            # Metrics, calibration (ECE), statistical significance tests
├── scripts/                # Entry-point scripts (train, evaluate, federate, explain)
├── notebooks/               # Exploratory analysis and figure-generation notebooks
├── tests/                   # Unit tests for core modules
└── docs/                    # Additional documentation (dataset setup, reproducing results)
```

## Datasets

All experiments use exclusively public, de-identified datasets:

| Dataset | Modality | Source |
|---|---|---|
| LC25000 | Histopathology | Kaggle / Borkowski et al. |
| TCGA-LUAD / TCGA-LUSC | Histopathology + genomic | NCI GDC Data Portal |
| LIDC-IDRI | CT | The Cancer Imaging Archive |
| NLST | CT + clinical | The Cancer Imaging Archive |
| IQ-OTH/NCCD | CT | IEEE DataPort |

See `docs/dataset_setup.md` for download and preprocessing instructions.

## Installation

```bash
git clone https://github.com/kapopat1980/FL-MMT.git
cd FL-MMT
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
# Train the centralised baseline
python scripts/train.py --config configs/centralised.yaml

# Run the simulated federated training (8 clients, FedProx + DP-SGD)
python scripts/federate.py --config configs/federated_dp.yaml

# Evaluate a trained checkpoint
python scripts/evaluate.py --checkpoint checkpoints/fl_mmt_best.pt --config configs/eval.yaml

# Generate explainability outputs (Integrated Gradients / attention flow / SHAP)
python scripts/explain.py --checkpoint checkpoints/fl_mmt_best.pt --sample_id <id>
```

## Reproducing paper results

Configs in `configs/` correspond directly to the experiments reported in Tables 2–6 and Figures 1–5 of the manuscript. See `docs/reproducing_results.md` for the exact mapping between config files and reported numbers.

## Citation

If you use this code, please cite:

```bibtex
@article{popat_fl_mmt,
  title   = {A Privacy-Preserving Federated Transformer with Cross-Modal Attention for Longitudinal Lung Cancer Diagnosis and Prognosis},
  author  = {Meva, Divyakant T. and Popat, Kalpesh},
  journal = {Computers in Biology and Medicine},
  year    = {2026}
}
```

## License

MIT License — see `LICENSE`.

## Contact

Kalpesh Popat — kapopat@gmail.com — Faculty of Computer Applications, Marwadi University, Rajkot, Gujarat, India
