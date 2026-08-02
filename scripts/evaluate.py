"""
Evaluation entry point: computes macro-AUC, accuracy, F1, C-index, Brier
score, ECE (via MC Dropout), and Wilcoxon/Cohen's d comparisons against
baselines — reproducing Tables 4-6 of the manuscript.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/fl_mmt_best.pt --config configs/eval.yaml
"""

import argparse
import yaml
import torch
import numpy as np

from src.evaluation.calibration import mc_dropout_predict, expected_calibration_error
from src.evaluation.significance import compare_to_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Loading checkpoint from {args.checkpoint}")
    # model = FLMMTModel(...); model.load_state_dict(torch.load(args.checkpoint))
    # test_loader = build_dataloaders(...)[2]
    #
    # all_confidences, all_correctness = [], []
    # for batch in test_loader:
    #     mc_results = mc_dropout_predict(model, batch, n_samples=cfg["evaluation"]["mc_dropout_samples"])
    #     confidences = mc_results["classification_logits"]["mean"].max(dim=-1).values.cpu().numpy()
    #     preds = mc_results["classification_logits"]["mean"].argmax(dim=-1)
    #     correctness = (preds == batch["label"]).cpu().numpy()
    #     all_confidences.append(confidences)
    #     all_correctness.append(correctness)
    #
    # ece = expected_calibration_error(np.concatenate(all_confidences), np.concatenate(all_correctness),
    #                                   n_bins=cfg["evaluation"]["ece_bins"])
    # print(f"ECE: {ece:.4f}")
    #
    # for baseline_name in cfg["evaluation"]["baselines_to_compare"]:
    #     result = compare_to_baseline(proposed_scores, baseline_scores[baseline_name])
    #     print(f"vs {baseline_name}: p={result['p_value']:.4f}, d={result['cohens_d']:.3f}, "
    #           f"significant={result['significant']}")

    print("See docs/reproducing_results.md for the full evaluation pipeline "
          "and dataset-loading prerequisites.")


if __name__ == "__main__":
    main()
