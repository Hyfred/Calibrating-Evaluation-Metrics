from itertools import product

from src.calibration import (
    # binning,
    # hierarchical_scaling_v2,
    none,
    # scaling,
    scaling_v2,
    scaling_binning,
    # hierarchical_scaling,
    # hierarchical_scaling_binning,
    beta_binning,
    scaling_beta_binning,
    hierarchical_scaling_beta_binning,
    beta_binning_ir,
    scaling_beta_binning_ir,
    hierarchical_scaling_beta_binning_ir,
)

from src.calibration import binning_raw as binning
from src.calibration import binning_soften_istonic_direc as binning_ir

# POINTS_PER_BIN = [2, 10, 20, 21, 22, 30, 40, 50, 60, 70, 80, 90, 100, 200]
# N_BINS = [2, 5, 10, 20, 30, 40, 50]
POINTS_PER_BIN = range(2, 500, 5)
N_BINS = range(2, 102, 5)  # 1000, 100)
PRIOR_SD = [2.5, 5, 10, 25, 50]
# PPI_LAMBDAS = [0.05, 0.25, 0.5, 0.75, 0.95]
PPI_LAMBDAS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.95]
# POINTS_PER_BIN = [30, 50]
# PRIOR_SD = [5]
# PPI_LAMBDAS = [0]#, 0.01, 0.05]

POINTS_PER_BIN = [131]  # , 50, 100, 200, 400]
N_BINS = [5]
PRIOR_SD = [2.5]
# PPI_LAMBDAS = [0.05, 0.25, 0.5, 0.75, 0.95]
PPI_LAMBDAS = [0.01]  # , 0.05, 0.1, 0.25, 0.5, 0.95]


def _flatten(calibrator_params):
    names = [item["name"] for item in calibrator_params]
    values = [item["values"] for item in calibrator_params]
    combinations = list(product(*values))
    output = [{names[i]: comb[i] for i in range(len(names))} for comb in combinations]

    return output


CALIBRATORS = {
    "none": {
        "module": none,
        "params": _flatten([]),
    },
    "binning": {
        "module": binning,
        "params": _flatten(
            [
                # {"name": "n_bins", "values": N_BINS}
            ]
        ),
    },
    "binning_ir": {
        "module": binning_ir,
        "params": _flatten(
            [
                # {"name": "n_bins", "values": N_BINS}
            ]
        ),
    },
    # "scaling": {
    #     "module": scaling,
    #     "params": _flatten(
    #         [
    #             {"name": "prior_sd", "values": PRIOR_SD},
    #         ]
    #     ),
    # },
    "scaling_v2": {
        # "module": scaling,
        "module": scaling_v2,
        "params": _flatten(
            [
                # {"name": "prior_sd", "values": PRIOR_SD},
            ]
        ),
    },
    "scaling_binning": {
        "module": scaling_binning,
        "params": _flatten(
            [
                # {"name": "prior_sd", "values": PRIOR_SD},
                # {"name": "n_bins", "values": N_BINS},
            ]
        ),
    },
    # "hierarchical_scaling_v2": {
    #     "module": hierarchical_scaling_v2,
    #     "params": _flatten(
    #         [
    #             # {"name": "prior_sd", "values": PRIOR_SD},
    #         ]
    #     ),
    # },
    "beta_binning": {
        "module": beta_binning,
        "params": _flatten(
            [
                # {
                # "name": "points_per_bin",
                # "values": POINTS_PER_BIN,
                # }
            ]
        ),
    },
    "scaling_beta_binning": {
        "module": scaling_beta_binning,
        "params": _flatten(
            [
                # {"name": "prior_sd", "values": PRIOR_SD},
                # {"name": "points_per_bin", "values": POINTS_PER_BIN},
            ]
        ),
    },
    "hierarchical_scaling_beta_binning": {
        "module": hierarchical_scaling_beta_binning,
        "params": _flatten(
            [
                # {"name": "prior_sd", "values": PRIOR_SD},
                # {"name": "points_per_bin", "values": POINTS_PER_BIN},
            ]
        ),
    },
    "beta_binning_ir": {
        "module": beta_binning_ir,
        "params": _flatten(
            [
                # {
                # "name": "points_per_bin",
                # "values": POINTS_PER_BIN,
                # }
            ]
        ),
    },
    "scaling_beta_binning_ir": {
        "module": scaling_beta_binning_ir,
        "params": _flatten(
            [
                # {"name": "prior_sd", "values": PRIOR_SD},
                # {"name": "points_per_bin", "values": POINTS_PER_BIN},
            ]
        ),
    },
    "hierarchical_scaling_beta_binning_ir": {
        "module": hierarchical_scaling_beta_binning_ir,
        "params": _flatten(
            [
                # {"name": "prior_sd", "values": PRIOR_SD},
                # {"name": "points_per_bin", "values": POINTS_PER_BIN},
            ]
        ),
    },
}
