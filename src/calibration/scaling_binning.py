import os
import pickle

from sklearn.model_selection import train_test_split
import torch

from src.calibration.binning_raw import HB_binary
from src.calibration.models import logistic_train


def train(*args, **kwargs):
    data = kwargs["data"]
    calibrator_param = kwargs["calibrator_param"]
    seed = kwargs["seed"]

    kept_idxs = kwargs.get("kept_idxs")
    dropped_idxs = kwargs.get("dropped_idxs")

    cache_file = str(
        hash(
            (
                "scaling_binning",
                str(data),
                str(calibrator_param),
                str(seed),
                str(kept_idxs),
                str(dropped_idxs),
            )
        )
    )

    data_scaling, data_binning = train_test_split(
        data, train_size=0.5, random_state=seed
    )

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

    # binning
    confidence_scores = (
        model(
            torch.from_numpy(data_binning["confidence_score"].values)
            .unsqueeze(-1)
            .float()
        )
        .detach()
        .numpy()
    )
    
    model = HB_binary(seed=seed, **{k: calibrator_param[k] for k in ("n_bins",)})
    model.fit(y_score=data_binning["confidence_score"], y=confidence_scores)

    return model


def test(*args, **kwargs):
    data = kwargs["data"]
    model = kwargs["model"]

    return model.predict_proba(y_score=data["confidence_score"])
