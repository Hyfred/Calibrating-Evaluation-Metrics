import numpy as np

from src.assessment.ece import ece


def beta_ece(y, pred_prob, beta_values, n_bins=15):
    pred_prob = pred_prob.squeeze()
    beta_values = beta_values.squeeze()
    y = y.squeeze()
    assert np.size(pred_prob.shape) == 1, "Check dimensions of input matrices"  #

    _beta_ece = 0
    for s in np.unique(beta_values):
        s_inds = np.argwhere(beta_values == s)

        if s_inds.size == 0:
            # s_inds is 0 because s is np.nan, so we find the inds here
            s_inds = np.argwhere(np.isnan(beta_values))

        _beta_ece += s_inds.size * (
            ece(y[s_inds], pred_prob[s_inds], n_bins, quiet=True)
        )

    _beta_ece = _beta_ece / y.size
    return _beta_ece
