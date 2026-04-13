import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

from src.calibration.models import logistic_train, print_train_accuracy


def train(*args, **kwargs):
    data = kwargs["data"]
    calibrator_param = kwargs["calibrator_param"]
    seed = kwargs["seed"]

    kept_idxs = kwargs.get("kept_idxs")
    dropped_idxs = kwargs.get("dropped_idxs")

    cache_file = str(
        hash(
            (
                "scaling",
                str(data),
                str(calibrator_param),
                str(seed),
                str(kept_idxs),
                str(dropped_idxs),
            )
        )
    )

    model_path = os.path.join("models_cache", f"{cache_file}.pickle")
    # if os.path.exists(model_path):
    if False:
        with open(model_path, "rb") as fh:
            print("Using cache")
            model = pickle.load(fh)
    else:
        model = logistic_train(
            seed=seed,
            h=data["confidence_score"].values,
            y=data["judgement_value"].values,
            l1_lambda=0,
            n_epochs=10000,
            lr=0.01,
        )

        model = LogisticRegression(random_state=seed).fit(
            data["confidence_score"].values[:, np.newaxis], data["judgement_value"].values
        )

        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)

    print_train_accuracy(
        model,
        confidence=data["confidence_score"].values,
        judgements=data["judgement_value"].values,
    )

    return model


def test(*args, **kwargs):
    data = kwargs["data"]
    model = kwargs["model"]

    try:
        # try to use pytorch model
        out = (
            model(
                torch.from_numpy(data["confidence_score"].values).unsqueeze(-1).float()
            )
            .detach()
            .numpy()
        )
    except:
        # use sklearn model
        out = model.predict_proba(data["confidence_score"].values[:, np.newaxis])[:, 1]

    return out
