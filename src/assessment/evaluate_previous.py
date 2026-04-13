import numpy as np
from sklearn.metrics import brier_score_loss, davies_bouldin_score

from src.assessment.auac import auac
from src.assessment.beta_ece import beta_ece
from src.assessment.ece import ece


def evaluate(*args, **kwargs):
    data = kwargs["data"]
    embeddings = kwargs["embeddings"]
    confidence_scores = kwargs["confidence_scores"]
    beta_values = kwargs["beta_values"]
    n_bins = 7

    try:
        ece_value = ece(
            y=data["judgement_value"].values, pred_prob=confidence_scores, n_bins=n_bins
        )
    except:
        breakpoint()

    beta_ece_value = beta_ece(
        y=data["judgement_value"].values,
        beta_values=beta_values,
        pred_prob=confidence_scores,
        n_bins=n_bins,
    )

    try:
        brier_score_loss_value = brier_score_loss(
            y_true=data["judgement_value"], y_proba=confidence_scores
        )
    except:
        breakpoint()

    auac_value = auac(
        accuracies=data["judgement_value"].values, confidences=confidence_scores
    )

    try:
        clustering_metric = davies_bouldin_score(
            X=np.stack(embeddings), labels=beta_values
        )
    except:
        clustering_metric = None

    return {
        "beta_ece": beta_ece_value,
        "ece": ece_value,
        "brier_score": brier_score_loss_value,
        "auac": auac_value,
        "clustering_metric": clustering_metric,
    }
