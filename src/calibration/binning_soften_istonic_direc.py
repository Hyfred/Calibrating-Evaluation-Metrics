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
    def __init__(self, seed, **kwargs):
        self.delta = 1e-10
        self.interpolator = None
        self.fitted = False
        self.rng = np.random.default_rng(seed)

    def fit(self, y_score, y):
        """
        take Isotonic Regression
        """
        y_score = y_score.squeeze()
        y = y.squeeze()
        assert y_score.size == y.size, "Check dimensions of input matrices"

        # add noise
        y_score = nudge(y_score, self.delta, self.rng)

        # Fit isotonic regression
        self.interpolator = IsotonicRegression(out_of_bounds="clip")
        self.interpolator.fit(y_score, y)
        self.fitted = True

    def predict_proba(self, y_score):
        """
        calibrate prob
        """
        y_score = np.atleast_1d(y_score).astype(float)
        y_score = nudge(y_score, self.delta, self.rng)
        if self.fitted:
            y_score = self.interpolator.predict(y_score)
        return np.clip(y_score, 0, 1)


def train(*args, **kwargs):
    data = kwargs["data"]
    seed = kwargs["seed"]

    model = HB_binary(seed=seed)
    model.fit(y_score=data["confidence_score"], y=data["judgement_value"])
    return model


def test(*args, **kwargs):
    data = kwargs["data"]
    model = kwargs["model"]
    return model.predict_proba(y_score=data["confidence_score"])