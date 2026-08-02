"""
Statistical significance testing used for the state-of-the-art comparison
in Table 6 of the manuscript: paired Wilcoxon signed-rank test plus
Cohen's d effect size between FL-MMT and each re-implemented baseline.
"""

import numpy as np
from scipy.stats import wilcoxon


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples a (proposed) vs b (baseline)."""
    diff = a - b
    return diff.mean() / diff.std(ddof=1)


def compare_to_baseline(proposed_scores: np.ndarray, baseline_scores: np.ndarray, alpha: float = 0.05):
    """
    Runs a paired Wilcoxon signed-rank test and reports Cohen's d.

    Args:
        proposed_scores: per-fold or per-bootstrap metric values for FL-MMT
        baseline_scores: matched per-fold or per-bootstrap metric values for the baseline
        alpha: significance threshold
    Returns:
        dict with keys: statistic, p_value, significant, cohens_d
    """
    statistic, p_value = wilcoxon(proposed_scores, baseline_scores)
    d = cohens_d(proposed_scores, baseline_scores)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": bool(p_value < alpha),
        "cohens_d": d,
    }
