# refactor_experiments.py
import pickle
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from paretoset import paretoset

from src.assessment.evaluate import evaluate
from src.beta.kdtree import KDTree
from src.calibration import CALIBRATORS
from src.utils import solve_for_B, solve_for_b
from src.assessment.ece import ece_continuous

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
import logging
import math
import os
import random

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------
# Utility partitioner classes
# ----------------------------
class KMeansPartitioner:
    def __init__(self, n_clusters: int = 256, random_state: int = 42):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, embeddings: List[np.ndarray]):
        self.kmeans.fit(embeddings)

    def get_partition_indices(self, embeddings: List[np.ndarray]) -> np.ndarray:
        return self.kmeans.predict(embeddings)


class KDTreePartitionerWrapper:
    """Thin wrapper to unify API of KDTree in src.beta.kdtree"""
    def __init__(self, max_depth: int = 8):
        self._tree = KDTree(max_depth=max_depth)

    def fit(self, embeddings: List[np.ndarray]):
        self._tree.fit(embeddings)

    def get_partition_indices(self, embeddings: List[np.ndarray]) -> np.ndarray:
        # KDTree.get_partition_indices is assumed to be `get_partition_indices` or `get_partition`.
        # The original code used beta.get_partition_indices, so we keep same name.
        return self._tree.get_partition_indices(embeddings)


class PartitionerFactory:
    @staticmethod
    def get_partitioner(method: str, **kwargs):
        method = method.lower()
        if method in ("kmeans", "k-mean", "k-mean", "k_means"):
            return KMeansPartitioner()
        elif method in ("kdtree", "kd-tree", "kd_tree"):
            return KDTreePartitionerWrapper(**kwargs)
        else:
            raise ValueError(f"Unknown partitioner method: {method}")


# ----------------------------
# Data utils
# ----------------------------
def read_and_preprocess_data(cfg, emb: str, dimension: str, predict_dim: str):
    """
    Read CSV and embeddings pickle, filter, rename columns, and return (ds, embeddings_list)
    """
    df_path = os.path.join(cfg.processed_data_dir, f"{emb}_with_predictions.csv")
    emb_path = os.path.join(cfg.processed_data_dir, f"{cfg.dataset_name}_embeddings_{emb}.pickle")

    ds = pd.read_csv(df_path)
    with open(emb_path, "rb") as fh:
        embeddings_map = pickle.load(fh)

    # filter as original
    ds = ds[~ds[dimension].isna()]
    ds = ds[ds[dimension] >= 0]
    ds = ds.rename(columns={predict_dim: "confidence_score", dimension: "judgement_value"})

    # Extract embeddings in same order as ds rows
    embeddings = [embeddings_map[(row["Document"], str(row["Summary"]))] for _, row in ds.iterrows()]
    return ds.reset_index(drop=True), embeddings


def split_data(ds: pd.DataFrame, embeddings: List[np.ndarray], seed: int,
               kd_tree_split_size=0.2, calibration_train_split_size=0.6,
               calibration_val_split_size=0.1, calibration_test_split_size=0.1):
    """
    Split into four parts: build_kdtree (d1), train (d2), val (d3), test (d4)
    Returns a dict with dataframes and embedding lists.
    """
    SPLITS = [kd_tree_split_size, calibration_train_split_size, calibration_val_split_size, calibration_test_split_size]
    assert abs(sum(SPLITS) - 1.0) < 1e-6

    d1, d234, e1, e234 = train_test_split(ds, embeddings, train_size=kd_tree_split_size, random_state=seed)
    d2, d34, e2, e34 = train_test_split(d234, e234, train_size=calibration_train_split_size / sum(SPLITS[1:]), random_state=seed)
    d3, d4, e3, e4 = train_test_split(d34, e34, train_size=calibration_val_split_size / sum(SPLITS[2:]), random_state=seed)

    return {
        "data_kdtree_build": d1.reset_index(drop=True),
        "data_train": d2.reset_index(drop=True),
        "data_val": d3.reset_index(drop=True),
        "data_test": d4.reset_index(drop=True),
        "embeddings_kdtree_build": e1,
        "embeddings_train": e2,
        "embeddings_val": e3,
        "embeddings_test": e4,
    }


def keep_one_doc_ids(data_test_df: pd.DataFrame, embeddings_test: List[np.ndarray], doc_id_col: str = "Doc_id"):
    """
    Keep one random row per Doc_id, preserve alignment with embeddings_test.
    """
    df = data_test_df.copy()
    df["__orig_idx__"] = range(len(df))
    dedup = df.groupby(doc_id_col, group_keys=False).apply(lambda x: x.sample(1, random_state=42))
    kept_indices = dedup["__orig_idx__"].values
    dedup_embeddings = [embeddings_test[i] for i in kept_indices]
    dedup = dedup.drop(columns="__orig_idx__").reset_index(drop=True)
    return dedup, dedup_embeddings


# ----------------------------
# Calibration hyperparam utilities
# ----------------------------
def create_binning_hyperparameters(N: int, nu_ranges: List[float], eps_val=0.2, alpha=0.1, scaling_binning_split_ratio=0.5, b: int = 8):
    """
    Return a dict mapping calibrator_key -> list of param dicts.
    This closely follows your previous logic but returns a stable structure for experiments.
    """
    # simple deterministic choices (you can expand)
    b_binning_lst = [b]
    B_binning = max(1, N // b)

    b_scaling_binning_lst = [b]
    B_scaling_binning = max(1, int(N * scaling_binning_split_ratio) // b)

    params = {
        "binning": [{"n_bins": B_binning}],
        "scaling_binning": [{"n_bins": B_scaling_binning}],
        "beta_binning": [{"points_per_bin": x} for x in b_binning_lst],
        "scaling_beta_binning": [{"points_per_bin": x} for x in b_scaling_binning_lst],
        "hierarchical_scaling_beta_binning": [{"points_per_bin": x} for x in b_scaling_binning_lst],
    }

    # fill missing calibrators gracefully
    return params


# ----------------------------
# Metrics & plotting helpers
# ----------------------------
def compute_ece_and_fit_line(human_scores, model_scores, n_bins=10):
    human_scores = np.array(human_scores)
    model_scores = np.array(model_scores)

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(model_scores, quantiles)

    bin_centers = []
    avg_human_scores = []
    ece = 0.0
    for i in range(len(bin_edges) - 1):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            in_bin = (model_scores >= lo) & (model_scores <= hi)
        else:
            in_bin = (model_scores >= lo) & (model_scores < hi)
        if np.any(in_bin):
            avg_model = np.mean(model_scores[in_bin])
            avg_human = np.mean(human_scores[in_bin])
            prob = np.sum(in_bin) / len(model_scores)
            ece += abs(avg_model - avg_human) * prob
            bin_centers.append(avg_model)
            avg_human_scores.append(avg_human)

    if len(bin_centers) >= 2:
        slope, intercept, _, _, _ = linregress(bin_centers, avg_human_scores)
        pred_line = slope * np.asarray(bin_centers) + intercept
        mse = float(np.mean((np.asarray(avg_human_scores) - pred_line) ** 2))
    else:
        slope, intercept, mse = float("nan"), float("nan"), float("nan")

    return {"ece": ece, "slope": slope, "intercept": intercept, "mse": mse}


# ----------------------------
# Multi-objective / best param retrieval
# ----------------------------
def mo_best_param_retrieval(calibration_result: Dict):
    """
    Given calibration_result like {calibrator_key: [ {param_idx: {'validation_metrics':...}}, ... ] }
    return calibrator_key -> list of Pareto-optimal param idxs
    """
    multi_objective_metrics = ["beta_ece", "brier_score"]
    ret = {}
    for calibrator_key, param_maps in calibration_result.items():
        idx_to_m0 = defaultdict(list)
        idx_to_m1 = defaultdict(list)
        for pm in param_maps:
            for param_idx, metrics in pm.items():
                idx_to_m0[param_idx].append(metrics["validation_metrics"][multi_objective_metrics[0]])
                idx_to_m1[param_idx].append(metrics["validation_metrics"][multi_objective_metrics[1]])
        # average
        param_idx_to_metric_0 = {k: np.mean(v) for k, v in idx_to_m0.items()}
        param_idx_to_metric_1 = {k: np.mean(v) for k, v in idx_to_m1.items()}

        mo_metrics = pd.DataFrame({
            multi_objective_metrics[0]: list(param_idx_to_metric_0.values()),
            multi_objective_metrics[1]: list(param_idx_to_metric_1.values()),
        })

        mask = paretoset(mo_metrics, sense=["min", "min"])
        pareto_idxs = mo_metrics[mask].index.to_list()
        # map back to original param idx keys (assumes order preserved)
        param_keys = list(param_idx_to_metric_0.keys())
        pareto_param_keys = [param_keys[i] for i in pareto_idxs]
        ret[calibrator_key] = pareto_param_keys
    return ret


def best_model_retrieval(evaluation_results: Dict, metric_name: str, metric_direction=min):
    """
    Return mapping method -> (best_param_idx, avg_metric_value)
    evaluation_results: {seed: {method: [ {calibrator_param_idx:.., validation_metrics:..}, ...] } }
    """
    metric_sum_count = {}
    for seed, methods in evaluation_results.items():
        for method, calibrators in methods.items():
            for calibrator in calibrators:
                idx = calibrator["calibrator_param_idx"]
                mval = calibrator["validation_metrics"][metric_name]
                key = (method, idx)
                metric_sum_count.setdefault(key, {"sum": 0.0, "count": 0})
                metric_sum_count[key]["sum"] += mval
                metric_sum_count[key]["count"] += 1

    metric_avg_per_method = {k: v["sum"] / v["count"] for k, v in metric_sum_count.items()}
    best_per_method = {}
    for (method, idx), avg in metric_avg_per_method.items():
        if method not in best_per_method:
            best_per_method[method] = (idx, avg)
        else:
            current_val = best_per_method[method][1]
            if metric_direction == min:
                if avg < current_val:
                    best_per_method[method] = (idx, avg)
            else:
                if avg > current_val:
                    best_per_method[method] = (idx, avg)
    return best_per_method


# ----------------------------
# Core runner
# ----------------------------
class ExperimentRunner:
    def __init__(self, cfg, partitioner_method="kdtree", partitioner_kwargs=None):
        self.cfg = cfg
        self.partitioner_method = partitioner_method
        self.partitioner_kwargs = partitioner_kwargs or {}
        # controllable seeds and calibrators come from cfg in your original script

    def run_validation(self, seed: int, ds: pd.DataFrame, embeddings: List[np.ndarray],
                       calibrator_params_from_theory: Dict[str, List[Dict[str, Any]]],
                       calibrators_to_run: List[str], max_depth: int = 8):
        """
        Run validation across calibrators and hyperparams.
        Returns validation_results structure and trained models for each calibrator param.
        """
        data_splits = split_data(ds, embeddings, seed)
        # choose partitioner and fit on d1
        partitioner = PartitionerFactory.get_partitioner(self.partitioner_method, **self.partitioner_kwargs)
        partitioner.fit(data_splits["embeddings_kdtree_build"])

        validation_results = defaultdict(list)  # calibrator_key -> list of {param_idx: {...}}
        trained_models = {}  # (calibrator_key, param_idx) -> model

        for cal_key in calibrators_to_run:
            module = CALIBRATORS[cal_key]["module"]
            params_list = calibrator_params_from_theory.get(cal_key, [{}])

            for pidx, p in enumerate(params_list):
                logger.info(f"Validation seed={seed} calibrator={cal_key} param_idx={pidx} params={p}")
                # train model
                try:
                    beta_values = partitioner.get_partition_indices(data_splits["embeddings_train"])
                    model = module.train(dataset=self.cfg.dataset_name, prompt=None,
                                         data=data_splits["data_train"], calibrator_param=p,
                                         seed=seed, beta_values=beta_values)
                except Exception as exc:
                    logger.exception("Training failed", exc_info=exc)
                    model = None

                # validation test
                try:
                    beta_val = partitioner.get_partition_indices(data_splits["embeddings_val"])
                    conf_val = module.test(data=data_splits["data_val"], model=model, beta_values=beta_val)
                except Exception as exc:
                    logger.exception("Validation test failed", exc_info=exc)
                    conf_val = np.zeros(len(data_splits["data_val"]))  # fallback

                metrics = evaluate(data=data_splits["data_val"], embeddings=data_splits["embeddings_val"],
                                   confidence_scores=conf_val, beta_values=beta_val)

                validation_results[seed].append({
                    "calibrator_key": cal_key,
                    "calibrator_param_idx": pidx,
                    "model": model,
                    "validation_metrics": metrics
                })
                trained_models[(cal_key, pidx)] = model

        return data_splits, partitioner, validation_results, trained_models

    def run_test(self, seed: int, data_splits: Dict, partitioner, model, calibrator_module, calibrator_param, tag: str, dedup_by_docid=False):
        """
        Run test for a single trained model & calibrator setting.
        Returns metrics dicts and per-instance records.
        """
        # optionally dedup
        data_test = data_splits["data_test"]
        emb_test = data_splits["embeddings_test"]
        if dedup_by_docid and "Doc_id" in data_test.columns:
            data_test, emb_test = keep_one_doc_ids(data_test, emb_test, doc_id_col="Doc_id")

        beta_test = partitioner.get_partition_indices(emb_test)
        try:
            conf_test = calibrator_module.test(data=data_test, model=model, beta_values=beta_test)
        except Exception as exc:
            logger.exception("Test prediction failed", exc_info=exc)
            conf_test = np.zeros(len(data_test))

        eval_metrics = evaluate(data=data_test, embeddings=emb_test, confidence_scores=conf_test, beta_values=beta_test)

        # calibrated slope/mse
        slope_info = compute_ece_and_fit_line(data_test["judgement_value"].values, conf_test, n_bins=10)
        eval_metrics.update(slope_info)

        # original scores metrics
        orig_scores = data_test["confidence_score"].values
        ece_val = ece_continuous(y=data_test["judgement_value"].values, pred_prob=orig_scores, n_bins=10)
        brier = mean_squared_error(data_test["judgement_value"].values, orig_scores)
        orig_line = compute_ece_and_fit_line(data_test["judgement_value"].values, orig_scores, n_bins=10)
        ori_metrics = {
            "ece_value": ece_val,
            "brier_score": brier,
            "slope": orig_line["slope"],
            "intercept": orig_line["intercept"],
            "mse": orig_line["mse"],
        }

        # per-instance records
        per_instance = {
            "doc_id": data_test.get("Doc_id", pd.Series([None] * len(data_test))).values,
            "label": data_test["judgement_value"].values,
            "predict": orig_scores,
            "calibra_predict": conf_test,
            "prompt_name": getattr(self.cfg, "emb", None),
            "dimension": tag,
        }

        return eval_metrics, ori_metrics, per_instance

    def compare_binning_algorithms(self, cfg, emb: str, dimension: str, predict_dim: str,
                                   seeds: List[int], calibrators_to_compare: List[str],
                                   partitioner_method: Optional[str] = None, partitioner_kwargs: Optional[dict] = None,
                                   b: int = 8, max_depth: int = 8):
        """
        Run the full pipeline for the given calibrators list and return two DataFrames:
         - df_results : aggregated test metrics per calibrator
         - df_instance: per-sample predictions for all runs
        """
        if partitioner_method is not None:
            self.partitioner_method = partitioner_method
        if partitioner_kwargs is not None:
            self.partitioner_kwargs = partitioner_kwargs

        ds, embeddings = read_and_preprocess_data(cfg, emb, dimension, predict_dim)
        results_rows = []
        instance_rows = []

        for seed in seeds:
            logger.info(f"=== Starting seed {seed} ===")
            # prepare hyperparams from theory
            data_split_tmp = split_data(ds, embeddings, seed)
            calibrator_params = {seed: create_binning_hyperparameters(N=len(data_split_tmp["data_train"]), nu_ranges=[0, 0.001, 0.002], b=b)}

            # validation
            data_splits, partitioner, val_results, trained_models = self.run_validation(seed, ds, embeddings, calibrator_params[seed], calibrators_to_compare, max_depth=max_depth)

            # re-organize val_results into original structure for best selection
            # val_results: {seed: [ {calibrator_key, calibrator_param_idx, model, validation_metrics}, ... ] }
            # create mapping per calibrator_key
            per_calibrator = defaultdict(list)
            for item in val_results[seed]:
                per_calibrator[item["calibrator_key"]].append({item["calibrator_param_idx"]: {"validation_metrics": item["validation_metrics"], "model": item["model"]}})

            # pick best param per calibrator using ECE (min)
            best_map = {}
            # reuse best_model_retrieval but need to convert structure
            # convert to expected input structure for best_model_retrieval
            eval_results_for_best = {seed: defaultdict(list)}
            for key, lst in per_calibrator.items():
                for dic in lst:
                    for pidx, val in dic.items():
                        eval_results_for_best[seed][key].append({"calibrator_param_idx": pidx, "validation_metrics": val["validation_metrics"], "model": val["model"]})

            best_per_method = best_model_retrieval(eval_results_for_best, metric_name="ece", metric_direction=min)

            # test with chosen best param
            for cal_key in calibrators_to_compare:
                if cal_key not in best_per_method:
                    logger.warning(f"No best param found for {cal_key} in seed {seed}, skipping")
                    continue
                best_idx, _ = best_per_method[cal_key]
                model = None
                # find trained model for this (cal_key, best_idx)
                model = None
                for item in val_results[seed]:
                    if item["calibrator_key"] == cal_key and item["calibrator_param_idx"] == best_idx:
                        model = item["model"]
                        break

                # fallback: retrain if not found
                if model is None:
                    logger.info(f"Best model missing in cache for {cal_key} param {best_idx}, retraining.")
                    try:
                        beta_train = partitioner.get_partition_indices(data_splits["embeddings_train"])
                        model = CALIBRATORS[cal_key]["module"].train(dataset=cfg.dataset_name, prompt=emb, data=data_splits["data_train"], calibrator_param=calibrator_params[seed].get(cal_key, [{}])[best_idx], seed=seed, beta_values=beta_train)
                    except Exception as exc:
                        logger.exception("Retrain failed", exc_info=exc)
                        model = None

                # run test
                test_metrics, test_ori_metrics, test_value = self.run_test(seed, data_splits, partitioner, model, CALIBRATORS[cal_key]["module"], calibrator_params[seed].get(cal_key, [{}])[best_idx], tag=dimension, dedup_by_docid=("avg" in emb))
                # aggregate
                if cal_key=='none':
                  row = {
                        "seed": seed,
                        "calibrator": cal_key,
                        "ece_before": round(test_ori_metrics.get("ece_value"),3),
                        "brier_before": round(test_ori_metrics.get("brier_score"),3),
                        "slope_before": round(test_ori_metrics.get("slope"),3),
                        "intercept_before": round(test_ori_metrics.get("intercept"),3),
                  }
                else:
                  row = {
                        "seed": seed,
                        "calibrator": cal_key,
                        "ece_before": round(test_ori_metrics.get("ece_value"),3),
                        "brier_before": round(test_ori_metrics.get("brier_score"),3),
                        "slope_before": round(test_ori_metrics.get("slope"),3),
                        "intercept_before": round(test_ori_metrics.get("intercept"),3),
                        "ece_after": round(test_metrics.get("ece"),3),
                        "beta_ece": round(test_metrics.get("beta_ece"),3),
                        "brier_after": round(test_metrics.get("brier_score"),3),
                        "slope_after": round(test_metrics.get("slope"),3),
                        "intercept_after": round(test_metrics.get("intercept"),3),
                  }
                results_rows.append(row)

                # per-instance -> expand into dict rows
                ni = len(test_value["label"])
                for i in range(ni):
                    instance_rows.append({
                        "seed": seed,
                        "calibrator": cal_key,
                        "doc_id": test_value["doc_id"][i] if test_value.get("doc_id") is not None else None,
                        "label": test_value["label"][i],
                        "predict": test_value["predict"][i],
                        "calibra_predict": test_value["calibra_predict"][i],
                    })

        df_results = pd.DataFrame(results_rows)
        df_instance = pd.DataFrame(instance_rows)
        return df_results, df_instance


# ----------------------------
# Example CLI-like usage
# ----------------------------
if __name__ == "__main__":
    # Minimal local config object to mimic your previous Config class
    class SimpleCfg:
        processed_data_dir = "../CaliEvaSum_v7/processed_data"
        dataset_name = "FineSurE_test"
        emb = ["qwen8b_pair_prompt"]
        seeds = [42, 2345, 888, 124, 8181818, 92929, 11123, 99087]
      #   calibrators = ["none", "beta_binning", "beta_binning_ir"]  # default set; override below if needed

    cfg = SimpleCfg()

    runner = ExperimentRunner(cfg, partitioner_method="kdtree", partitioner_kwargs={"max_depth": 8})

    # pick which calibrators to compare (3 example binning algorithms)
    calibrators_to_compare = [
        "none",
        "binning",
        "binning_ir",
        # "scaling_v2",
      #   "scaling_binning",
        "beta_binning",
      #   "scaling_beta_binning",
      #   "hierarchical_scaling_beta_binning",
        "beta_binning_ir",
      #   "scaling_beta_binning_ir",
      #   "hierarchical_scaling_beta_binning_ir",
    ]  # <-- change to the three you want to compare

    all_results = []
    all_instances = []

    for emb in cfg.emb:
        df_res, df_inst = runner.compare_binning_algorithms(cfg=cfg, emb=emb,
                                                            dimension="Faithfulness",
                                                            predict_dim="predict_fai",
                                                            seeds=cfg.seeds,
                                                            calibrators_to_compare=calibrators_to_compare,
                                                            partitioner_method="kmeans",
                                                            partitioner_kwargs={"max_depth": 8},
                                                            b=8,
                                                            max_depth=8)
        # example save
        df_res.to_csv(f"comparison_results_{emb}.csv", index=False)
        df_inst.to_csv(f"comparison_instances_{emb}.csv", index=False)
        logger.info("Saved result CSVs")
