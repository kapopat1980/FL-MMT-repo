import numpy as np
from src.evaluation.calibration import expected_calibration_error


def test_ece_perfect_calibration():
    confidences = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    correctness = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])  # 90% accuracy at 90% confidence
    ece = expected_calibration_error(confidences, correctness, n_bins=10)
    assert ece < 0.05


def test_ece_poor_calibration():
    confidences = np.array([0.95] * 10)
    correctness = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 10% accuracy at 95% confidence
    ece = expected_calibration_error(confidences, correctness, n_bins=10)
    assert ece > 0.5
