"""
Monte Carlo Dropout uncertainty estimation and Expected Calibration Error
(ECE), as reported jointly across the classification, risk, and survival
heads in Section 4.4 of the manuscript.
"""

import torch
import numpy as np


@torch.no_grad()
def mc_dropout_predict(model, batch, n_samples: int = 30):
    """
    Runs `n_samples` stochastic forward passes with dropout active
    (see FLMMTTaskHeads.enable_mc_dropout) to estimate predictive mean
    and epistemic uncertainty (variance) per task head.

    Returns:
        dict with keys per head: {"mean": tensor, "std": tensor}
    """
    model.eval()
    model.task_heads.enable_mc_dropout()

    samples = {"classification_logits": [], "risk_logits": [], "survival_logits": []}
    for _ in range(n_samples):
        outputs = model(batch)
        for key in samples:
            samples[key].append(torch.softmax(outputs[key], dim=-1))

    results = {}
    for key, vals in samples.items():
        stacked = torch.stack(vals, dim=0)  # (n_samples, B, num_classes)
        results[key] = {"mean": stacked.mean(dim=0), "std": stacked.std(dim=0)}

    return results


def expected_calibration_error(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 15):
    """
    Computes the Expected Calibration Error (ECE).

    Args:
        confidences: (N,) array of predicted-class confidence scores in [0, 1]
        correctness: (N,) binary array, 1 if prediction was correct else 0
        n_bins: number of confidence bins
    Returns:
        ece: scalar Expected Calibration Error
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correctness[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return ece
