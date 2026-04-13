import numpy as np

from src.utils import bin_points


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


def nudge(matrix, delta, rng):
    return (matrix + rng.uniform(low=0, high=delta, size=(matrix.shape))) / (1 + delta)


class HB_binary(object):
    def __init__(self, seed, n_bins=15):
        ### Hyperparameters
        self.delta = 1e-10
        self.n_bins = n_bins

        ### Parameters to be learnt
        self.bin_upper_edges = None
        self.mean_pred_values = None
        self.num_calibration_examples_in_bin = None

        ### Internal variables
        self.fitted = False

        self.rng = np.random.default_rng(seed)

    def fit(self, y_score, y):
        assert self.n_bins is not None, "Number of bins has to be specified"
        y_score = y_score.squeeze()
        y = y.squeeze()
        assert y_score.size == y.size, "Check dimensions of input matrices"
        assert (
            y.size >= self.n_bins
        ), "Number of bins should be less than the number of calibration points"

        ### All required (hyper-)parameters have been passed correctly
        ### Uniform-mass binning/histogram binning code starts below

        # delta-randomization
        y_score = nudge(y_score, self.delta, self.rng)

        # compute uniform-mass-bins using calibration data
        self.bin_upper_edges = get_uniform_mass_bins(y_score, self.n_bins)

        # assign calibration data to bins
        bin_assignment = bin_points(y_score, self.bin_upper_edges)

        # compute bias of each bin
        self.num_calibration_examples_in_bin = np.zeros([self.n_bins, 1])
        self.mean_pred_values = np.empty(self.n_bins)
        for i in range(self.n_bins):
            bin_idx = bin_assignment == i
            self.num_calibration_examples_in_bin[i] = sum(bin_idx)

            # nudge performs delta-randomization
            if sum(bin_idx) > 0:
                self.mean_pred_values[i] = nudge(
                    y[bin_idx].mean(), self.delta, self.rng
                )
            else:
                self.mean_pred_values[i] = nudge(0.5, self.delta, self.rng)

        # check that my code is correct
        assert np.sum(self.num_calibration_examples_in_bin) == y.size

        # histogram binning done
        self.fitted = True

    def predict_proba(self, y_score):
        assert self.fitted is True, "Call HB_binary.fit() first"
        y_score = y_score.squeeze()

        # delta-randomization
        y_score = nudge(y_score, self.delta, self.rng)

        # assign test data to bins
        y_bins = bin_points(y_score, self.bin_upper_edges)

        # get calibrated predicted probabilities
        y_pred_prob = self.mean_pred_values[y_bins]
        return y_pred_prob


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
