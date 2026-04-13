import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.calibration.binning_raw import HB_binary


class identity:
    def predict_proba(self, x):
        return x

    def predict(self, x):
        return np.argmax(x, axis=1)


class BetaBinning(object):
    def __init__(self, seed, points_per_bin=50):        
        self.points_per_bin = points_per_bin

        ### UMD models to be learnt
        self.hb_binary_list = dict()

        self.num_classes = None

        self.seed = seed

    def fit(self, pred_prob, y, beta_values):
        y = y.squeeze()
        
        for s in np.unique(beta_values):
            s_inds = np.argwhere(beta_values == s)
            n_s = np.size(s_inds)

            bins_s = np.floor(n_s / self.points_per_bin).astype("int")
            if bins_s == 0:
                self.hb_binary_list[s] = identity()
            else:
                hb = HB_binary(n_bins=bins_s, seed=self.seed)
                hb.fit(pred_prob[s_inds], y[s_inds])
                self.hb_binary_list[s] = hb

        self.fitted = True

    def predict_proba(self, pred_prob, beta_values):
        confidence_scores = []
        for _prob, _beta_value in zip(pred_prob, beta_values):
            if _beta_value not in self.hb_binary_list:
                confidence_scores.append(_prob)
            else:
                confidence_scores.append(
                    self.hb_binary_list[_beta_value].predict_proba(_prob).item()
                )
        return np.array(confidence_scores)


def train(*args, **kwargs):
    data = kwargs["data"]
    beta_values = kwargs["beta_values"]
    calibrator_param = kwargs["calibrator_param"]
    seed = kwargs["seed"]

    kept_idxs = kwargs.get("kept_idxs")

    indices = [*range(len(data))]

    (
        indices_scaling,
        _,
        data_scaling,
        data_binning,
        beta_values_scaling,
        beta_values_binning,
    ) = train_test_split(indices, data, beta_values, train_size=0.5, random_state=seed)

    if kept_idxs is not None:
        # sparsify
        _idxs = [idx for idx in indices_scaling if idx in kept_idxs]
        data_scaling = data.iloc[_idxs]
        beta_values_scaling = beta_values[_idxs]

    data = pd.concat([data_scaling, data_binning])
    beta_values = np.hstack([beta_values_scaling, beta_values_binning])

    model = BetaBinning(seed=seed, **calibrator_param)
    model.fit(
        pred_prob=data["confidence_score"].values,
        y=data["judgement_value"].values,
        beta_values=beta_values,
    )

    return model


def test(*args, **kwargs):
    data = kwargs["data"]
    model = kwargs["model"]
    beta_values = kwargs["beta_values"]

    return model.predict_proba(data["confidence_score"].values, beta_values)
