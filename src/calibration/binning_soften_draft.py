import numpy as np
from scipy.interpolate import interp1d


def get_uniform_mass_bins(probs, n_bins):
    assert probs.size >= n_bins, "Fewer points than bins"
    probs_sorted = np.sort(probs)
    groups = np.array_split(probs_sorted, n_bins)
    bin_upper_edges = [max(groups[i]) for i in range(n_bins - 1)]
    bin_upper_edges.append(np.inf)
    return np.array(bin_upper_edges)


def bin_points(points, bin_upper_edges):
    return np.digitize(points, bin_upper_edges, right=True)


def nudge(matrix, delta, rng):
    return (matrix + rng.uniform(low=0, high=delta, size=matrix.shape)) / (1 + delta)


class HB_binary:
    def __init__(self, seed, n_bins=15):
        self.delta = 1e-10
        self.n_bins = n_bins
        self.bin_upper_edges = None
        self.mean_pred_values = None
        self.num_calibration_examples_in_bin = None
        self.bin_centers = None
        self.interpolator = None
        self.fitted = False
        self.rng = np.random.default_rng(seed)

    def fit(self, y_score, y):
        assert self.n_bins is not None, "Number of bins must be specified"
        y_score = y_score.squeeze()
        y = y.squeeze()
        assert y_score.size == y.size, "Check dimensions of input matrices"
        assert y.size >= self.n_bins, "Number of bins must be ≤ number of calibration points"

        y_score = nudge(y_score, self.delta, self.rng)
        self.bin_upper_edges = get_uniform_mass_bins(y_score, self.n_bins)
        bin_assignment = bin_points(y_score, self.bin_upper_edges)

        self.num_calibration_examples_in_bin = np.zeros([self.n_bins, 1])
        self.mean_pred_values = np.empty(self.n_bins)

        for i in range(self.n_bins):
            bin_idx = bin_assignment == i
            self.num_calibration_examples_in_bin[i] = sum(bin_idx)
            if sum(bin_idx) > 0:
                self.mean_pred_values[i] = nudge(y[bin_idx].mean(), self.delta, self.rng)
            else:
                self.mean_pred_values[i] = nudge(0.5, self.delta, self.rng)

        assert np.sum(self.num_calibration_examples_in_bin) == y.size

        # Compute bin centers
        bin_lower_edges = np.concatenate(([0.0], self.bin_upper_edges[:-1]))
        self.bin_centers = (bin_lower_edges + self.bin_upper_edges) / 2

        # Fit interpolation function
        self.interpolator = interp1d(
            self.bin_centers,
            self.mean_pred_values,
            kind="linear",
            fill_value=(self.mean_pred_values[0], self.mean_pred_values[-1]),
            bounds_error=False,
        )

        self.fitted = True

    def predict_proba(self, y_score):
        assert self.fitted, "Call HB_binary.fit() first"
        y_score = y_score.squeeze()
        y_score = nudge(y_score, self.delta, self.rng)
        return self.interpolator(y_score)

def train(*args, **kwargs):
    data = kwargs["data"]
    calibrator_param = kwargs["calibrator_param"]
    seed = kwargs["seed"]

    model = HB_binary(seed=seed, **calibrator_param)
    model.fit(y_score=data["confidence_score"], y=data["judgement_value"])

    return model


def test(*args, **kwargs):
    data = kwargs["data"]
    model = kwargs["model"]

    return model.predict_proba(y_score=data["confidence_score"])