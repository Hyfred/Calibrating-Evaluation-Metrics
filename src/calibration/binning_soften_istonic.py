import numpy as np
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt


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


def plot_scatter_with_fit_line(x, y, bin_center, mean_pred, fit_y, title='Scatter Plot with Fitted Curve'):
    """
    scatter plot with isotonic regression fit
    """
    x = np.array(x)
    y = np.array(y)

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, label='Individual Points', color='blue', alpha=0.5)
    plt.scatter(bin_center, mean_pred, label='Group Means', color='green')
    plt.plot(bin_center, fit_y, color='red', label='Isotonic Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('scatter_plot_isotonic.png')


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
                print('Warning: empty bin detected')
                self.mean_pred_values[i] = nudge(0.5, self.delta, self.rng)

        assert np.sum(self.num_calibration_examples_in_bin) == y.size

        # Compute bin centers
        self.bin_upper_edges[-1] = 1  # not inf
        bin_lower_edges = np.concatenate(([0.0], self.bin_upper_edges[:-1]))
        self.bin_centers = (bin_lower_edges + self.bin_upper_edges) / 2

        # Fit isotonic regression
        if len(self.bin_centers) > 1:
            self.interpolator = IsotonicRegression(out_of_bounds="clip")
            self.interpolator.fit(self.bin_centers, self.mean_pred_values)
            self.fitted = True

            # 可选：画图
            fit_y = self.interpolator.predict(self.bin_centers)
            # plot_scatter_with_fit_line(y_score, y, self.bin_centers, self.mean_pred_values, fit_y)

    def predict_proba(self, y_score):
        y_score = np.atleast_1d(y_score).astype(float)  # 保证至少1维
        y_score = nudge(y_score, self.delta, self.rng)
        if self.fitted:
            y_score = self.interpolator.predict(y_score)
        return np.clip(y_score, 0, 1)


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