import os
import pickle

from sklearn.model_selection import train_test_split
import torch

from src.calibration.beta_binning_ir import BetaBinning
from src.calibration.models import logistic_train


def train(*args, **kwargs):
    data = kwargs["data"]
    beta_values = kwargs["beta_values"]
    calibrator_param = kwargs["calibrator_param"]
    seed = kwargs["seed"]

    kept_idxs = kwargs.get("kept_idxs")
    dropped_idxs = kwargs.get("dropped_idxs")

    cache_file = str(
        hash(
            (
                "scaling_beta_binning",
                str(data),
                str(calibrator_param),
                str(seed),
                str(kept_idxs),
                str(dropped_idxs),
            )
        )
    )

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
        data_scaling = data.iloc[[idx for idx in indices_scaling if idx in kept_idxs]]

    # check if both labels are represented, if not take a random sample from dropped_idxs
    for label in [0, 1]:
        if (
            dropped_idxs is not None
            and label not in data_scaling["judgement_value"].values
        ):

            _idxs = [idx for idx in indices_scaling if idx in dropped_idxs]
            _data = data.iloc[_idxs]
            extra_idx = None
            for _idx in _idxs:
                if extra_idx is not None:
                    data_scaling.loc[len(data_scaling)] = data.iloc[extra_idx]
                    # beta_values_scaling = np.hstack([beta_values_scaling, beta_values[extra_idx]])
                    break

                if data.iloc[_idx].judgement_value == label:
                    extra_idx = _idx

    model_path = os.path.join("models_cache", f"{cache_file}.pickle")
    # if os.path.exists(model_path):
    if False:
        print("Using cache")
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)
    else:        
        model = logistic_train(
            seed=seed,
            h=data_scaling["confidence_score"].values,
            y=data_scaling["judgement_value"].values,
            l1_lambda=0,
            n_epochs=10000,
            lr=0.01,
        )

        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)
    
    confidence_scores = (
        model(
            torch.from_numpy(data_binning["confidence_score"].values)
            .unsqueeze(-1)
            .float()
        )
        .detach()
        .numpy()
    )

    model = BetaBinning(
        seed=seed, **{k: calibrator_param[k] for k in ("points_per_bin",)}
    )
    model.fit(
        pred_prob=data_binning["confidence_score"].values,
        y=confidence_scores,
        beta_values=beta_values_binning,
    )

    return model


def test(*args, **kwargs):
    data = kwargs["data"]
    model = kwargs["model"]
    beta_values = kwargs["beta_values"]

    return model.predict_proba(data["confidence_score"].values, beta_values)
