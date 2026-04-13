import pickle
import re
from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from paretoset import paretoset

from src.assessment.evaluate import evaluate
from src.beta.kdtree import KDTree
from src.calibration import CALIBRATORS
# from src.justask.parsing import (
#     CONFIDENCE_EXPRESSIONS,
#     CONFIDENCE_EXPRESSIONS_PROBABILITIES,
# )
from src.utils import solve_for_B, solve_for_b

import argparse

kd_tree_split_size = 0.2
calibration_train_split_size = 0.6
calibration_val_split_size = 0.1
calibration_test_split_size = 0.1
SPLITS = [
    kd_tree_split_size,
    calibration_train_split_size,
    calibration_val_split_size,
    calibration_test_split_size,
]

assert sum(SPLITS) == 1.0


def mo_best_param_retrieval(calibration_result):
    multi_objective_metrics = ["beta_ece", "brier_score"]
    calibrator_key_to_best_param_idx = {}
    for calibrator_key, param_to_metrics_maps in calibration_result.items():
        param_idx_to_metric_0 = defaultdict(list)
        param_idx_to_metric_1 = defaultdict(list)
        for param_to_metrics_map in param_to_metrics_maps:
            for param_idx, metrics in param_to_metrics_map.items():
                param_idx_to_metric_0[param_idx].append(
                    metrics["validation_metrics"][multi_objective_metrics[0]]
                )
                param_idx_to_metric_1[param_idx].append(
                    metrics["validation_metrics"][multi_objective_metrics[1]]
                )

        param_idx_to_metric_0 = {
            param_idx: np.mean(metric_values)
            for param_idx, metric_values in param_idx_to_metric_0.items()
        }
        param_idx_to_metric_1 = {
            param_idx: np.mean(metric_values)
            for param_idx, metric_values in param_idx_to_metric_1.items()
        }

        mo_metrics = pd.DataFrame(
            {
                multi_objective_metrics[0]: param_idx_to_metric_0.values(),
                multi_objective_metrics[1]: param_idx_to_metric_1.values(),
            }
        )

        mask = paretoset(mo_metrics, sense=["min", "min"])
        paretoset_idxs = mo_metrics[mask]
        calibrator_key_to_best_param_idx[calibrator_key] = (
            paretoset_idxs.index.to_list()
        )
    return calibrator_key_to_best_param_idx


def best_model_retrieval(evaluation_results, metric_name, metric_direction):

    # Initialize a dictionary to store sum of auac and count of occurrences per method and calibrator_param_idx
    metric_sum_count = {}

    # Iterate over each seed and method
    for seed, methods in evaluation_results.items():
        for method, calibrators in methods.items():
            for calibrator in calibrators:
                calibrator_idx = calibrator["calibrator_param_idx"]
                metric = calibrator["validation_metrics"][metric_name]

                # Create a unique key for method and calibrator_param_idx
                key = (method, calibrator_idx)

                # If the key doesn't exist, initialize it
                if key not in metric_sum_count:
                    metric_sum_count[key] = {"sum": 0, "count": 0}

                # Accumulate the sum of auac and increment the count
                metric_sum_count[key]["sum"] += metric
                metric_sum_count[key]["count"] += 1

    # Now, compute the average for each method and calibrator_param_idx
    metric_avg_per_method = {}

    for key, values in metric_sum_count.items():
        method, calibrator_idx = key
        avg_auac = values["sum"] / values["count"]

        # Store the average auac for the method and calibrator_param_idx
        metric_avg_per_method[key] = avg_auac

    # Initialize an empty dictionary to store the minimum values
    best_value_per_method = {}

    # Iterate through the dictionary
    for key, value in metric_avg_per_method.items():
        method = key[0]
        calibrator_idx = key[1]

        # Check if method is already in the result dictionary
        if method not in best_value_per_method:
            # If not, add the current key and value
            best_value_per_method[method] = (calibrator_idx, value)
        else:
            # If it is, compare the current value with the stored one and update if it's smaller
            current_min_value = best_value_per_method[method][1]
            if metric_direction == min:
                if value < current_min_value:
                    best_value_per_method[method] = (calibrator_idx, value)
            elif metric_direction == max:
                if value > current_min_value:
                    best_value_per_method[method] = (calibrator_idx, value)

    # Prepare the final result dictionary where the key is the original tuple and value is the minimum value
    result = {
        method: (calibrator_idx, value)
        for method, (calibrator_idx, value) in best_value_per_method.items()
    }

    return result


def split_data(ds, embeddings, seed):
    # Split data to d1 (beta), d2 (calibrator), d3 (validation), d4 (test)
    data_d1, data_d234, embeddings_d1, embeddings_d234 = train_test_split(
        ds, embeddings, train_size=kd_tree_split_size, random_state=seed
    )

    data_d2, data_d34, embeddings_d2, embeddings_d34 = train_test_split(
        data_d234,
        embeddings_d234,
        train_size=calibration_train_split_size / sum(SPLITS[1:]),
        random_state=seed,
    )

    data_d3, data_d4, embeddings_d3, embeddings_d4 = train_test_split(
        data_d34,
        embeddings_d34,
        train_size=calibration_val_split_size / sum(SPLITS[2:]),
        random_state=seed,
    )

    return {
        "data_kdtree_build": data_d1,
        "data_train": data_d2,
        "data_val": data_d3,
        "data_test": data_d4,
        "embeddings_kdtree_build": embeddings_d1,
        "embeddings_train": embeddings_d2,
        "embeddings_val": embeddings_d3,
        "embeddings_test": embeddings_d4,
    }


def read_and_preprocess_data(cfg):
    ds = pd.read_csv(
        f"{cfg.processed_data_dir}/{cfg.dataset_name}_{cfg.prompt_name}_{cfg.qa_llm}_{cfg.judgement_llm}.csv"
    )
    with open(
        f"{cfg.processed_data_dir}/{cfg.dataset_name}_{cfg.prompt_name}_{cfg.qa_llm}_embeddings.pickle",
        "rb",
    ) as file:
        embeddings = pickle.load(file)

    # Remove unusable rows and prepare column names
    ds = ds[~ds["llm_confidence"].isna()]
    ds = ds[~ds["judgement_value"].isna()]
    ds = ds[ds["llm_confidence"] >= 0]
    ds.rename(
        columns={
            "llm_confidence": "confidence_score",
        },
        inplace=True,
    )
    # print(list(embeddings.keys()))
    # print(('16 + 72 =', 88))

    embeddings = [
        embeddings[(row["question"], str(row["llm_answer"]))] for _, row in ds.iterrows()
    ]

    return ds, embeddings


def create_binning_hyperparameters(
    N, nu_ranges, eps_val=0.2, alpha=0.1, scaling_binning_split_ratio=0.5, b=8
):
    # Based on the size of embeddings d1 and embeddings d2, decide on max_depth and b (and max_depth and B)

    # binning approaches:
    b_binning_lst = [
        np.ceil(solve_for_b(N=N, eps_val=eps_val, nu=nu, alpha=alpha)).astype("int")
        for nu in nu_ranges
    ]
    B_binning = np.ceil(solve_for_B(N=N, eps_val=eps_val, alpha=alpha)).astype("int")

    # scaling-binning approaches:
    b_scaling_binning_lst = [
        np.ceil(
            solve_for_b(
                N=N * scaling_binning_split_ratio, eps_val=eps_val, nu=nu, alpha=alpha
            )
        ).astype("int")
        for nu in nu_ranges
    ]
    B_scaling_binning = np.ceil(
        solve_for_B(N=N * scaling_binning_split_ratio, eps_val=eps_val, alpha=alpha)
    ).astype("int")

    B_binning = N // b
    B_scaling_binning = N // b
    b_binning_lst = [b]
    b_scaling_binning_lst = [b]

    calibrator_params_from_theory = {
        "binning": [{"n_bins": B_binning}],
        "scaling_binning": [{"n_bins": B_scaling_binning}],
        "beta_binning": [{"points_per_bin": b_binning} for b_binning in b_binning_lst],
        "scaling_beta_binning": [
            {"points_per_bin": b_binning} for b_binning in b_scaling_binning_lst
        ],
        "hierarchical_scaling_beta_binning": [
            {"points_per_bin": b_binning} for b_binning in b_scaling_binning_lst
        ],
        "beta_binning_ir": [{"points_per_bin": b_binning} for b_binning in b_binning_lst],
        "scaling_beta_binning_ir": [
            {"points_per_bin": b_binning} for b_binning in b_scaling_binning_lst
        ],
        "hierarchical_scaling_beta_binning_ir": [
            {"points_per_bin": b_binning} for b_binning in b_scaling_binning_lst
        ],
    }

    return calibrator_params_from_theory


def get_recommended_depth(N_kd_tree, b):
    number_of_min_points_in_leaf = np.ceil(b * 2).astype("int")
    recommended_max_depth = np.ceil(
        np.log(N_kd_tree / number_of_min_points_in_leaf)
    ).astype("int")
    print(
        f"Number of min points in leaf according to theory: {number_of_min_points_in_leaf}"
    )
    print(f"Recommended maximum depth according to theory: {recommended_max_depth}")

    return recommended_max_depth


def workflow(
    cfg, seed, data, beta, calibrator_module, calibrator_param, mode, model=None
):
    if model is None:
        # Train model
        beta_values = beta.get_partition_indices(data["embeddings_train"])
        try:
            model = calibrator_module.train(
                dataset=cfg.dataset_name,
                prompt=cfg.prompt_name,
                data=data["data_train"],
                calibrator_param=calibrator_param,
                seed=seed,
                beta_values=beta_values,
            )
        except Exception as exc:
            print(exc)
            breakpoint()
            pass

    if mode == "validation":
        beta_values_val = beta.get_partition_indices(data["embeddings_val"])
        try:
            confidence_scores_val = calibrator_module.test(
                data=data["data_val"], model=model, beta_values=beta_values_val
            )
        except Exception as exc:
            print(exc)
            breakpoint()
            pass

        evaluation_metrics = evaluate(
            data=data["data_val"],
            embeddings=data["embeddings_val"],
            confidence_scores=confidence_scores_val,
            beta_values=beta_values_val,
        )
    else:
        beta_values_test = beta.get_partition_indices(data["embeddings_test"])
        try:
            confidence_scores_test = calibrator_module.test(
                data=data["data_test"], model=model, beta_values=beta_values_test
            )
        except Exception as exc:
            print(exc)
            breakpoint()
            pass

        evaluation_metrics = evaluate(
            data=data["data_test"],
            embeddings=data["embeddings_test"],
            confidence_scores=confidence_scores_test,
            beta_values=beta_values_test,
        )

    return model, evaluation_metrics


def construct_summary_results(test_results, max_depth):
    # Add brier_score to metrics
    metrics = ["beta_ece", "ece", "brier_score", "auac"]
    grouped = (
        pd.DataFrame(test_results)[["calibrator"] + metrics]
        .groupby("calibrator")
        .agg(["mean", "std", "count"])
    )

    # Define the z-score for a 95% confidence interval
    z = 1.96

    # Calculate the 95% confidence interval for all metrics
    for metric in metrics:
        grouped[metric, "ci_lower"] = grouped[metric, "mean"] - z * (
            grouped[metric, "std"] / np.sqrt(grouped[metric, "count"])
        )
        grouped[metric, "ci_upper"] = grouped[metric, "mean"] + z * (
            grouped[metric, "std"] / np.sqrt(grouped[metric, "count"])
        )
        grouped[metric, "ci"] = z * (
            grouped[metric, "std"] / np.sqrt(grouped[metric, "count"])
        )

    grouped["max_depth"] = max_depth
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    grouped = grouped.merge(
        pd.DataFrame(test_results)[
            ["calibrator", "best_calibrator_param"]
        ].drop_duplicates(),
        on="calibrator",
    )

    return grouped



# @hydra.main(version_base=None, config_path="../configs", config_name="experiment")

class Config:
    processed_data_dir = "processed_data"
    max_depths = list(range(11))
    qa_llm = "gemma" # change to gemma or mistral it would shows error
    judgement_llm = "llama2"
    prompt_name = "ling1s-topk" #"ling1s-topk" "verb1s-topk"
    dataset_name = "truthfulqa" #"mathqa" "mmlu" "openbookqa" "sciq" "triviaqa" "truthfulqa"
    # seeds = [111, 222, 333, 444, 555, 666, 777]#[42, 2345, 888, 124, 8181818, 92929, 11123, 99087]#, 2345, 888, 124, 8181818, 92929, 11123, 99087
    seeds = [6, 66, 666, 6666, 66666] # change to more it would shows error
    # seeds = [7, 77, 777, 7777, 77777] # change to more it would shows error
    calibrators = [
        "none",
        "binning",
        "binning_ir",
        "scaling_v2",
        "scaling_binning",
        "beta_binning",
        "scaling_beta_binning",
        "hierarchical_scaling_beta_binning",
        "beta_binning_ir",
        "scaling_beta_binning_ir",
        "hierarchical_scaling_beta_binning_ir",
    ]
    output_result_dir = "results"

cfg = Config()

def main(cfg: DictConfig) -> None:

    ds, embeddings = read_and_preprocess_data(cfg)

    calibrator_params_from_theory = dict()

    # Repeat train-validation + train-test for different max_depths
    final_results = []
    for b in [8]:#, 16, 32, 64, 128 # change to more it would shows error
        for max_depth in cfg.max_depths:

            # Compute validation metrics over various seeds and calibrator hyperparameters
            validation_results = defaultdict(lambda: defaultdict(list))
            for seed in cfg.seeds:

                data = split_data(ds, embeddings, seed)

                calibrator_params_from_theory[seed] = create_binning_hyperparameters(
                    N=len(data["data_train"]),
                    nu_ranges=[0, 0.001, 0.002, 0.003, 0.004, 0.005],
                    eps_val=0.1,
                    alpha=0.1,
                    scaling_binning_split_ratio=0.5,
                    b=b,
                )

                # Compute beta
                beta = KDTree(max_depth=max_depth)
                beta.fit(data["embeddings_kdtree_build"])

                for calibrator_key in cfg.calibrators:

                    calibrator_module = CALIBRATORS[calibrator_key]["module"]
                    calibrator_params = calibrator_params_from_theory[seed].get(
                        calibrator_key, [{}]
                    )

                    for calibrator_param_idx, calibrator_param in enumerate(
                        calibrator_params
                    ):
                        print(
                            f"Seed: {seed}, Dataset: {cfg.dataset_name}, Prompt: {cfg.prompt_name}, Calibrator: {calibrator_key}, Calibrator param: {str(calibrator_param)}"
                        )

                        model, validation_metrics = workflow(
                            cfg=cfg,
                            seed=seed,
                            data=data,
                            beta=beta,
                            calibrator_module=calibrator_module,
                            calibrator_param=calibrator_param,
                            mode="validation",
                        )

                        print(validation_metrics)

                        validation_results[seed][calibrator_key].append(
                            {
                                "calibrator_param_idx": calibrator_param_idx,
                                "model": model,
                                "validation_metrics": validation_metrics,
                            }
                        )

            best_calibrator_param_idx_to_metric_value = best_model_retrieval(
                validation_results, metric_name="auac", metric_direction=min
            )

            # train and test the calibrators with the best hyperparameters
            for calibrator_key in cfg.calibrators:
                calibrator_results = []
                for seed in cfg.seeds:
                    print(f"Seed: {seed}, Calibrator: {calibrator_key}")

                    calibrator_module = CALIBRATORS[calibrator_key]["module"]

                    calibrator_param_idx = best_calibrator_param_idx_to_metric_value[
                        calibrator_key
                    ][0]
                    model = validation_results[seed][calibrator_key][0]["model"]

                    best_calibrator_param = calibrator_params_from_theory[seed].get(
                        calibrator_key, [{}]
                    )[calibrator_param_idx]

                    data = split_data(ds, embeddings, seed)

                    _, test_metrics = workflow(
                        cfg=cfg,
                        seed=seed,
                        data=data,
                        beta=beta,
                        calibrator_module=calibrator_module,
                        calibrator_param=calibrator_param,
                        mode="test",
                        model=model,
                    )

                    test_metrics["seed"] = seed
                    test_metrics["dataset_name"] = cfg.dataset_name
                    test_metrics["prompt_name"] = cfg.prompt_name
                    test_metrics["qa_llm"] = cfg.qa_llm
                    test_metrics["judgement_llm"] = cfg.judgement_llm
                    test_metrics["max_depth"] = max_depth
                    test_metrics["calibrator"] = calibrator_key
                    test_metrics["best_calibrator_param"] = str(best_calibrator_param)
                    calibrator_results.append(test_metrics)

                calibrator_result = construct_summary_results(
                    calibrator_results, max_depth
                )
                final_results.append(calibrator_result)

    pd.concat(final_results).to_csv(
        f"all_result_include_brier/aggregated_results_{cfg.dataset_name}_{cfg.qa_llm}_{cfg.prompt_name}_raw.csv",
        index=False,
    )

    # Iterating for the paper tables
    final_aggregation_result = pd.concat(final_results).round(3)
    final_aggregation_result["Dataset"] = cfg.dataset_name
    final_aggregation_result["Prompt"] = cfg.prompt_name
    final_aggregation_result["LLM"] = cfg.qa_llm

    final_aggregation_result["\\ECE"] = (
        "$"
        + final_aggregation_result["ece_mean"].astype(str)
        + " \pm "
        + final_aggregation_result["ece_ci"].astype(str)
        + "$"
    )
    final_aggregation_result["\\Brier"] = (
        "$"
        + final_aggregation_result["brier_score_mean"].astype(str)
        + " \pm "
        + final_aggregation_result["brier_score_ci"].astype(str)
        + "$"
    )

    final_aggregation_result["\CE(h;\\beta)"] = (
        "$"
        + final_aggregation_result["beta_ece_mean"].astype(str)
        + " \pm "
        + final_aggregation_result["beta_ece_ci"].astype(str)
        + "$"
    )
    final_aggregation_result["\AUAC"] = (
        "$"
        + final_aggregation_result["auac_mean"].astype(str)
        + " \pm "
        + final_aggregation_result["auac_ci"].astype(str)
        + "$"
    )

    max_value = final_aggregation_result["max_depth_"].max()
    leaf_rows = final_aggregation_result[
        final_aggregation_result["max_depth_"] == max_value
    ]
    leaf_rows = leaf_rows.reset_index()

    best_ce_idx = leaf_rows["beta_ece_mean"].idxmin()
    ce = leaf_rows.at[best_ce_idx, "\CE(h;\\beta)"]
    bolded_ce = re.sub(r"\$(.*?)\$", r"$\\mathbf{\1}$", ce)
    leaf_rows.at[best_ce_idx, "\CE(h;\\beta)"] = bolded_ce

    best_auac_idx = leaf_rows["auac_mean"].idxmax()
    auac = leaf_rows.at[best_auac_idx, "\AUAC"]
    bolded_auac = re.sub(r"\$(.*?)\$", r"$\\mathbf{\1}$", auac)
    leaf_rows.at[best_auac_idx, "\AUAC"] = bolded_auac

    leaf_rows["calibrator"] = leaf_rows["calibrator"].replace(
        "hierarchical_scaling_v2", "hierarchical_scaling"
    )
    leaf_rows["calibrator"] = leaf_rows["calibrator"].replace("scaling_v2", "scaling")

    leaf_rows = leaf_rows.fillna("None")
    custom_sort_order = [
        "beta_binning",
        "beta_binning_ir",
        "hierarchical_scaling_beta_binning",
        "hierarchical_scaling_beta_binning_ir",
        "scaling_beta_binning",
        "scaling_beta_binning_ir",
        # "hierarchical_scaling",
        "scaling_binning",
        "scaling",
        "binning",
        "binning_ir",
        "none",
    ]
    leaf_rows["calibrator"] = pd.Categorical(
        leaf_rows["calibrator"], categories=custom_sort_order, ordered=True
    )
    leaf_rows = leaf_rows.sort_values("calibrator")

    leaf_rows["calibrator"] = leaf_rows["calibrator"].replace(
        {
            "beta_binning": "QAB",
            "hierarchical_scaling_beta_binning": "HS-QAB",
            "scaling_beta_binning": "S-QAB",
            "beta_binning_ir": "GIRB",
            "hierarchical_scaling_beta_binning_ir": "HS-GIRB",
            "scaling_beta_binning_ir": "S-GIRB",
            # "hierarchical_scaling": "HS",
            "scaling_binning": "S-B",
            "scaling": "S",
            "binning": "B",
            "binning_ir": "IRB",
            "none": "None",
        }
    )

    leaf_rows = leaf_rows[
        ~leaf_rows["calibrator"].isin(
            [
                "HS",
            ]
        )
    ]

    print(
        leaf_rows[
            ["Dataset", "Prompt", "LLM", "calibrator", "\\ECE", "\\CE(h;\\beta)", "\\Brier", "\\AUAC"]
        ].to_latex(index=False)
    )

    leaf_rows[
        ["Dataset", "Prompt", "LLM", "calibrator", "\\ECE", "\\CE(h;\\beta)", "\\Brier", "\\AUAC"]
    ].to_csv(
        f"all_result_include_brier/paper_table_{cfg.dataset_name}_{cfg.qa_llm}_{cfg.prompt_name}_raw.csv", 
        index=False
    )
    final_aggregation_result.reset_index().to_excel(
        f"all_result_include_brier/{cfg.dataset_name}_{cfg.qa_llm}_{cfg.prompt_name}_raw.xlsx"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default=cfg.dataset_name)
    parser.add_argument("--prompt_name", type=str, default=cfg.prompt_name)
    parser.add_argument("--qa_llm", type=str, default=cfg.qa_llm)
    args = parser.parse_args()

    # Override config
    cfg.dataset_name = args.dataset_name
    cfg.prompt_name = args.prompt_name
    cfg.qa_llm = args.qa_llm

    main(cfg)
