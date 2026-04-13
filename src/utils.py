import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from scipy.optimize import fsolve


def bin_points(scores, bin_edges):
    assert bin_edges is not None, "Bins have not been defined"
    scores = scores.squeeze()
    assert np.size(scores.shape) < 2, "scores should be a 1D vector or singleton"
    scores = np.reshape(scores, (scores.size, 1))
    bin_edges = np.reshape(bin_edges, (1, bin_edges.size))
    return np.sum(scores > bin_edges, axis=1)


def get_uniform_mass_bins(probs, n_bins):
    assert probs.size >= n_bins, "Fewer points than bins"

    probs_sorted = np.sort(probs)

    # split probabilities into groups of approx equal size
    groups = np.array_split(probs_sorted, n_bins)
    bin_edges = list()
    bin_upper_edges = list()

    for cur_group in range(n_bins - 1):
        bin_upper_edges += [max(groups[cur_group])]
    bin_upper_edges += [np.inf]

    return np.array(bin_upper_edges)


def bin_points_uniform(x, n_bins):
    x = x.squeeze()
    bin_upper_edges = get_uniform_mass_bins(x, n_bins)
    return np.sum(x.reshape((-1, 1)) > bin_upper_edges, axis=1)


def get_binned_probabilities_discrete(y, pred_prob, pred_prob_base=None):
    # assert len(np.unique(pred_prob)) <= (
    #     pred_prob.shape[0] / 10
    # ), "Predicted probabilities are not sufficiently discrete; using corresponding continuous method"
    bin_edges = np.sort(np.unique(pred_prob))
    true_n_bins = len(bin_edges)
    pi_pred = np.zeros(true_n_bins)
    pi_base = np.zeros(true_n_bins)
    pi_true = np.zeros(true_n_bins)
    n_elem = np.zeros(true_n_bins)
    bin_assignment = bin_points(pred_prob, bin_edges)

    for i in range(true_n_bins):
        bin_idx = bin_assignment == i
        assert sum(bin_idx) > 0, "This assert should pass by construction of the code"
        n_elem[i] = sum(bin_idx)
        pi_pred[i] = pred_prob[bin_idx].mean()
        if pred_prob_base is not None:
            pi_base[i] = pred_prob_base[bin_idx].mean()
        pi_true[i] = y[bin_idx].mean()

    assert sum(n_elem) == y.size

    return n_elem, pi_pred, pi_base, pi_true


def get_binned_probabilities_continuous(y, pred_prob, n_bins, pred_prob_base=None):
    pi_pred = np.zeros(n_bins)
    pi_base = np.zeros(n_bins)
    pi_true = np.zeros(n_bins)
    n_elem = np.zeros(n_bins)
    bin_assignment = bin_points_uniform(pred_prob, n_bins)

    for i in range(n_bins):
        bin_idx = bin_assignment == i
        assert sum(bin_idx) > 0, "This assert should pass by construction of the code"
        n_elem[i] = sum(bin_idx)
        pi_pred[i] = pred_prob[bin_idx].mean()
        if pred_prob_base is not None:
            pi_base[i] = pred_prob_base[bin_idx].mean()
        pi_true[i] = y[bin_idx].mean()

    assert sum(n_elem) == y.size

    return n_elem, pi_pred, pi_base, pi_true


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Set the seed for all GPUs
    np.random.seed(seed)
    random.seed(seed)

    # Ensures that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def solve_for_b(N, eps_val, nu=0, alpha=0.01):
    def eps_value(N, b, nu=0, alpha=0.01):
        return np.sqrt(np.log2(2 * N / (b * alpha)) / (2 * (b - 1))) + nu

    def equation(b):
        return eps_value(N, b, nu, alpha) - eps_val

    b_guess = 2.0

    b_solution = fsolve(equation, b_guess)
    return b_solution[0]


def solve_for_B(N, eps_val, alpha=0.01):
    def eps_value(N, B, nu=0, alpha=0.01):
        return np.sqrt(np.log2(2 * B / alpha) / (2 * (np.floor(N / B) - 1)))

    def equation(B):
        return eps_value(N, B, alpha) - eps_val

    B_guess = 2.0

    B_solution = fsolve(equation, B_guess)
    return B_solution[0]
