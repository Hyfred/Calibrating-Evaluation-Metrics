# based on: Gupta and Ramdas - Distribution-free calibration guarantees for histogram binning without sample splitting
import numpy as np

from src.utils import (
    get_binned_probabilities_discrete,
)


def ece(y, pred_prob, n_bins=15, quiet=False):
    n_elem, pi_pred, _, pi_true = get_binned_probabilities_discrete(y, pred_prob)
    return np.sum(n_elem * np.abs(pi_pred - pi_true)) / y.size
