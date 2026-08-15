"""ML training engine — the core trainer function.

Ported from CLASSify-2's models.py (trainer_func, uns_trainer_func,
estimatorevaluation, write_report).  S3 → storage, ClearML → callbacks,
all modeling math preserved exactly.
"""

from __future__ import annotations

import io
import json
import os
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from ml.backends import tabpfn_available
from ml.evaluate import (
    clusteringstats,
    do_cross_validate,
    getmodelstats,
    hasclustermodel,
)
from ml.options import CLUSTERING_MODELS
from ml.shap_explain import compute_shap
from ml.tuning import run_tuning
from ml.visualize import label_file, visualize
from storage.base import Storage

ProgressCallback = Callable[[int, int, str], None]
LogCallback = Callable[[str], None]


def _log(msg: str, output_f: Any, log_cb: LogCallback | None = None) -> None:
    """Write a message to the output file and optional log callback."""
    if hasattr(output_f, "write"):
        output_f.write(msg + "\n")
        output_f.flush()
    if log_cb:
        log_cb(msg)


def _get_estimator(emethod: str, args: Any, params: dict[str, Any] | None = None) -> Any:
    """Create a model estimator with default or tuned parameters."""
    params = params or {}
    n_jobs = args.n_jobs if args.n_jobs > 0 else (os.cpu_count() or 1)

    if emethod == "randomforest":
        return RandomForestClassifier(random_state=args.random_state, n_jobs=n_jobs, **params)
    elif emethod == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            random_state=args.random_state, n_jobs=n_jobs, eval_metric="logloss", **params
        )
    elif emethod == "gradientboosting":
        return GradientBoostingClassifier(random_state=args.random_state, **params)
    elif emethod == "histgradientboosting":
        return HistGradientBoostingClassifier(random_state=args.random_state, **params)
    elif emethod == "bagging":
        return BaggingClassifier(random_state=args.random_state, n_jobs=n_jobs, **params)
    elif emethod == "sgdclassifier":
        return SGDClassifier(random_state=args.random_state, **params)
    elif emethod == "logisticregression":
        return LogisticRegression(random_state=args.random_state, max_iter=1000, **params)
    elif emethod == "kneighbors":
        return KNeighborsClassifier(n_jobs=n_jobs, **params)
    elif emethod == "neuralnetwork":
        return MLPClassifier(random_state=args.random_state, max_iter=500, **params)
    elif emethod == "tabpfn":
        from ml.backends import require

        tabpfn_module = require("tabpfn")
        TabPFNClassifier = tabpfn_module.TabPFNClassifier
        return TabPFNClassifier(random_state=args.random_state, **params)
    elif emethod == "spectralclustering":
        return SpectralClustering(random_state=args.random_state, **params)
    elif emethod == "kmeans":
        return KMeans(random_state=args.random_state, **params)
    elif emethod == "hdbscan":
        from hdbscan import HDBSCAN

        return HDBSCAN(**params)
    else:
        raise ValueError(f"Unknown model: {emethod}")


def estimator_evaluation(
    args: Any,
    emethod: str,
    X: pd.DataFrame,
    y: pd.Series,
    storage: Storage,
    output_f: Any,
    all_columns: list[str],
    filename: str,
    mappings: dict | None,
    test_X: pd.DataFrame | None = None,
    test_y: pd.Series | None = None,
    log_cb: LogCallback | None = None,
) -> tuple[dict, dict]:
    """Train and evaluate a single supervised model.

    Returns (results_dict, positive_rates_dict).
    Preserved from CLASSify-2's estimatorevaluation — only I/O changed.
    """
    results: dict[str, Any] = {"labels": y.unique().tolist()}
    positive_rates: dict[str, Any] = {}
    binary = len(set(y)) <= 2

    try:
        # Scaling
        scaler = StandardScaler() if args.standard_scaler else MinMaxScaler(feature_range=(0, 1))  # noqa: SIM108
        scaler.fit(X.values)
        X_scaled = scaler.transform(X.values)

        # Sampling
        if args.train_sample_type == 1:  # Undersample
            from imblearn.under_sampling import RandomUnderSampler

            X_scaled, y = RandomUnderSampler(random_state=args.random_state).fit_resample(
                X_scaled, y
            )
        elif args.train_sample_type == 2:  # Oversample
            from imblearn.over_sampling import RandomOverSampler

            X_scaled, y = RandomOverSampler(random_state=args.random_state).fit_resample(
                X_scaled, y
            )

        # Parameter tuning
        if args.parameter_tune and emethod not in CLUSTERING_MODELS:
            _log(f"Tuning enabled: Running for {emethod}", output_f, log_cb)
            params, best_score = run_tuning(emethod, args, X_scaled, y, output_f)
            _log(f"Best params: {params}, score: {best_score}", output_f, log_cb)
        else:
            params = {}
            best_score = 0.0

        # Train final model
        if emethod not in CLUSTERING_MODELS:
            estimator = _get_estimator(emethod, args, params)
            best_random = estimator.fit(X_scaled, y)
        else:
            estimator = _get_estimator(emethod, args, params)
            best_random = estimator.fit(X_scaled)

        # SHAP
        shap_importance_json = None
        if emethod not in CLUSTERING_MODELS and args.shap_feature_explainability:
            if emethod == "tabpfn":
                _log("SHAP not supported for TabPFN model", output_f, log_cb)
                shap_importance_json = json.dumps({key: -1 for key in all_columns})
            else:
                shap_importance_json = compute_shap(
                    best_random,
                    emethod,
                    scaler,
                    X_scaled,
                    scaler.transform(test_X.values) if test_X is not None else X_scaled,
                    all_columns,
                    binary,
                    args,
                    storage,
                    filename,
                )

        if emethod in CLUSTERING_MODELS:
            if args.shap_feature_explainability:
                _log("SHAP not supported for clustering models", output_f, log_cb)
            shap_importance_json = json.dumps({key: -1 for key in all_columns})

        # Odds ratio for logistic regression
        if emethod == "logisticregression":
            log_odds = best_random.coef_[0]
            odds_ratios = np.exp(log_odds)
            odds_ratios_df = pd.DataFrame({"Feature": all_columns, "Odds Ratio": odds_ratios})
            float_X = X.astype(float)
            se = np.sqrt(
                np.diag(np.linalg.pinv(np.dot(float_X.T, float_X)) * best_random.coef_.var())
            )
            z = 1.96
            lower_ci = np.exp(log_odds - z * se)
            upper_ci = np.exp(log_odds + z * se)
            odds_ratios_df["95% CI Lower"] = lower_ci
            odds_ratios_df["95% CI Upper"] = upper_ci
            odds_ratios_df.sort_values(by=["Odds Ratio"], ascending=False, inplace=True)
            buf = io.StringIO()
            odds_ratios_df.to_csv(buf, index=False)
            storage.put_text(f"{filename}/logisticregression_odds_ratio", buf.getvalue())

        # Save model
        if not args.disable_model_save:
            _log(f"Saving {emethod} model", output_f, log_cb)
            model_buf = io.BytesIO()
            dump(best_random, model_buf)
            if model_buf.tell() > 0 and model_buf.tell() < 1024 * 1024 * 1024:
                model_buf.seek(0)
                storage.put_bytes(f"{filename}/{emethod}_model.joblib", model_buf.getvalue())
            else:
                _log(f"Saving {emethod} model failed: invalid size", output_f, log_cb)

            scaler_buf = io.BytesIO()
            dump(scaler, scaler_buf)
            scaler_buf.seek(0)
            storage.put_bytes(f"{filename}/scaler.joblib", scaler_buf.getvalue())

            # Test set metrics
            if test_X is not None and test_y is not None and emethod not in CLUSTERING_MODELS:
                test_X_scaled = scaler.transform(test_X.values)
                test_results = getmodelstats(
                    best_random,
                    test_X_scaled,
                    test_y,
                    True,
                    emethod,
                    storage,
                    filename,
                    args,
                )
                (
                    test_sensitivity,
                    test_specificity,
                    test_auc,
                    test_acc,
                    test_kappa,
                    test_npv,
                    test_ppv,
                    fpr,
                    tpr,
                    test_f1score,
                ) = test_results
                positive_rates[emethod] = {"tpr": tpr.tolist(), "fpr": fpr.tolist()}
            else:
                test_sensitivity = test_specificity = test_auc = test_acc = test_kappa = -1
                test_npv = test_ppv = test_f1score = -1

            # Cross-validation
            can_cv = emethod not in CLUSTERING_MODELS
            if can_cv:
                try:
                    cvt_results = do_cross_validate(
                        X_scaled, y, best_random, args.folds, args.repeats
                    )
                    (
                        cvt_sensitivity,
                        cvt_sensitivity_moe,
                        cvt_specificity,
                        cvt_specificity_moe,
                        cvt_auc,
                        cvt_auc_moe,
                        cvt_acc,
                        cvt_acc_moe,
                        cvt_ppv,
                        cvt_ppv_moe,
                        cvt_npv,
                        cvt_npv_moe,
                        cvt_f1score,
                        cvt_f1score_moe,
                    ) = cvt_results
                except Exception as e:
                    _log(f"CV error: {e}", output_f, log_cb)
                    cvt_sensitivity = cvt_f1score = cvt_specificity = cvt_auc = cvt_acc = -1
                    cvt_ppv = cvt_npv = -1
                    cvt_sensitivity_moe = cvt_specificity_moe = cvt_auc_moe = cvt_acc_moe = -1
                    cvt_ppv_moe = cvt_npv_moe = cvt_f1score_moe = -1
            else:
                cvt_sensitivity = cvt_f1score = cvt_specificity = cvt_auc = cvt_acc = -1
                cvt_ppv = cvt_npv = -1
                cvt_sensitivity_moe = cvt_specificity_moe = cvt_auc_moe = cvt_acc_moe = -1
                cvt_ppv_moe = cvt_npv_moe = cvt_f1score_moe = -1

            # Training set metrics
            if emethod not in CLUSTERING_MODELS:
                try:
                    train_results = getmodelstats(
                        best_random,
                        X_scaled,
                        y,
                        False,
                        emethod,
                        storage,
                        filename,
                        args,
                    )
                    (
                        trt_sensitivity,
                        trt_specificity,
                        trt_auc,
                        trt_acc,
                        trt_kappa,
                        trt_npv,
                        trt_ppv,
                        trt_f1score,
                    ) = train_results
                except Exception as e:
                    _log(f"Train metrics error: {e}", output_f, log_cb)
                    trt_sensitivity = trt_specificity = trt_auc = trt_acc = -1
                    trt_kappa = trt_npv = trt_ppv = trt_f1score = -1
            else:
                trt_sensitivity = trt_specificity = trt_auc = trt_acc = -1
                trt_kappa = trt_npv = trt_ppv = trt_f1score = -1

            # Store results
            n_train = X_scaled.shape[0]
            n_test = test_X.shape[0] if test_X is not None else 0

            results.update(
                {
                    "test_auc": test_auc,
                    "test_acc": test_acc,
                    "test_sensitivity": test_sensitivity,
                    "test_specificity": test_specificity,
                    "test_npv": test_npv,
                    "test_ppv": test_ppv,
                    "test_f1score": test_f1score,
                    "test_kappa": test_kappa,
                    "trt_auc": trt_auc,
                    "trt_acc": trt_acc,
                    "trt_sensitivity": trt_sensitivity,
                    "trt_specificity": trt_specificity,
                    "trt_npv": trt_npv,
                    "trt_ppv": trt_ppv,
                    "trt_f1score": trt_f1score,
                    "trt_kappa": trt_kappa,
                    "best_score": best_score,
                    "n_train": n_train,
                    "n_test": n_test,
                    "cvt_auc": f"{cvt_auc} ± {cvt_auc_moe}",
                    "cvt_acc": f"{cvt_acc} ± {cvt_acc_moe}",
                    "cvt_sensitivity": f"{cvt_sensitivity} ± {cvt_sensitivity_moe}",
                    "cvt_specificity": f"{cvt_specificity} ± {cvt_specificity_moe}",
                    "cvt_ppv": f"{cvt_ppv} ± {cvt_ppv_moe}",
                    "cvt_npv": f"{cvt_npv} ± {cvt_npv_moe}",
                    "cvt_f1score": f"{cvt_f1score} ± {cvt_f1score_moe}",
                }
            )

            if args.shap_feature_explainability and shap_importance_json:
                for key, value in json.loads(shap_importance_json).items():
                    results["shap_" + key] = value

    except Exception as e:
        _log(f"Error in training {emethod}: {e}", output_f, log_cb)
        traceback.print_exc()
        results.update({key: -1 for key in results if key != "labels"})

    return results, positive_rates


def uns_estimator_evaluation(
    args: Any,
    emethod: str,
    X: pd.DataFrame,
    storage: Storage,
    output_f: Any,
    all_columns: list[str],
    filename: str,
    log_cb: LogCallback | None = None,
) -> dict:
    """Train and evaluate a single unsupervised (clustering) model.

    Preserved from CLASSify-2's uns_estimatorevaluation.
    """
    results: dict[str, Any] = {}

    try:
        scaler = StandardScaler() if args.standard_scaler else MinMaxScaler(feature_range=(0, 1))  # noqa: SIM108
        scaler.fit(X.values)
        X_scaled = scaler.transform(X.values)

        if args.parameter_tune:
            params, best_score = run_tuning(emethod, args, X_scaled, None, output_f)
            _log(f"Best params: {params}, score: {best_score}", output_f, log_cb)
        else:
            if emethod == "spectralclustering":
                params = {"n_clusters": args.num_clusters, "affinity": "nearest_neighbors"}
            elif emethod == "kmeans":
                params = {"n_clusters": args.num_clusters}
            elif emethod == "hdbscan":
                _log("num_clusters ignored, hdbscan autodetermines.", output_f, log_cb)
                params = {
                    "min_cluster_size": args.min_cluster_size_start,
                    "min_samples": args.min_samples_start,
                }
            else:
                params = {}
            best_score = 0

        estimator = _get_estimator(emethod, args, params)
        best_random = estimator.fit(X_scaled)

        labels = best_random.labels_
        if np.all(labels == -1):
            _log(
                "All points classified as noise. Try changing clustering parameters.",
                output_f,
                log_cb,
            )
        results["labels"] = labels.tolist()

        # Save model
        if not args.disable_model_save:
            model_buf = io.BytesIO()
            dump(best_random, model_buf)
            if model_buf.tell() > 0 and model_buf.tell() < 1024 * 1024 * 1024:
                model_buf.seek(0)
                storage.put_bytes(f"{filename}/{emethod}_model.joblib", model_buf.getvalue())

            scaler_buf = io.BytesIO()
            dump(scaler, scaler_buf)
            scaler_buf.seek(0)
            storage.put_bytes(f"{filename}/scaler.joblib", scaler_buf.getvalue())

        # Clustering metrics
        try:
            train_results = clusteringstats(
                best_random, X_scaled, None, False, emethod, storage, filename, None, args, True
            )
            silhouette_score, davies_bouldin_score, calinski_harabasz_score = train_results
        except Exception as e:
            _log(f"Clustering stats error: {e}", output_f, log_cb)
            silhouette_score = davies_bouldin_score = calinski_harabasz_score = -1

        results.update(
            {
                "silhouette_score": silhouette_score,
                "davies_bouldin_score": davies_bouldin_score,
                "calinski_harabasz_score": calinski_harabasz_score,
                "best_score": best_score,
            }
        )

    except Exception as e:
        _log(f"Error in clustering {emethod}: {e}", output_f, log_cb)
        traceback.print_exc()
        results.update({key: -1 for key in results})

    return results


def write_report(
    args: Any,
    model_results: list[dict],
    storage: Storage,
    filename: str,
    positive_rates: dict,
) -> None:
    """Generate the final report CSV and visualizations.

    Preserved from CLASSify-2's write_report.
    """
    hascluster, othermodels = hasclustermodel(args)

    models_to_skip = []
    for model in model_results:
        if model["results"].get("best_score") == -1:
            models_to_skip.append(model["model"])
        if not args.parameter_tune:
            model["results"].pop("best_score", None)

    results_list = [k for k in model_results[0]["results"] if k != "labels"]
    header = "model,features," + ",".join(results_list)
    labels = model_results[0]["results"].get("labels", [])
    multiclass = len(set(labels)) > 2

    columns = header.split(",")
    df = pd.DataFrame(columns=columns)

    for model_stats in model_results:
        if model_stats["model"] in models_to_skip:
            continue
        column_key = model_stats["column_key"]
        line = model_stats["model"] + "," + column_key
        for results_name in results_list:
            if results_name == "labels":
                result = str(model_stats["results"][results_name]).replace(",", "")
            elif results_name[:3] != "cvt":
                result = str(round(float(model_stats["results"][results_name]), 3))
            else:
                val = model_stats["results"][results_name]
                if val[:3] == "-1":
                    result = "-1 ± -1"
                else:
                    words = val.split(" ")
                    result = f"{round(float(words[0]), 3)} {words[1]} {round(float(words[2]), 3)}"
            if result == "-1" or result[:2] == "-1":
                result = "N/A"
            line += "," + result
        row = line.split(",")
        df = pd.concat([df, pd.DataFrame([row], columns=df.columns)], ignore_index=True)

    storage.write_csv(f"{filename}/results", df, index=False)

    if args.visualize:
        if not args.supervised:
            label_file(model_results, storage, filename)
        visualize(df, storage, filename, args, positive_rates, multiclass)


def trainer(
    args: Any,
    storage: Storage,
    full_dataset: pd.DataFrame,
    testset: pd.DataFrame | None,
    on_progress: ProgressCallback | None = None,
    log_cb: LogCallback | None = None,
    cancel_token: Any = None,
) -> None:
    """Main trainer entry point — replaces CLASSify-2's trainer().

    Args:
        args: TrainingArgs instance
        storage: Storage instance for reading/writing artifacts
        full_dataset: The processed training dataset
        testset: Optional separate testset (None = auto split)
        on_progress: Callback(completed, total, message) for progress updates
        log_cb: Optional callback for log messages
        cancel_token: Object with ``is_set()`` method for cancellation
    """
    filename = args.report_uuid
    output_buf = io.StringIO()

    def _check_cancel() -> bool:
        return (
            cancel_token is not None and hasattr(cancel_token, "is_set") and cancel_token.is_set()
        )

    if args.supervised:
        _supervised_trainer(
            args,
            storage,
            full_dataset,
            testset,
            output_buf,
            filename,
            on_progress,
            log_cb,
            _check_cancel,
        )
    else:
        _unsupervised_trainer(
            args,
            storage,
            full_dataset,
            testset,
            output_buf,
            filename,
            on_progress,
            log_cb,
            _check_cancel,
        )

    # Save output log
    storage.put_text(f"{filename}/output_log", output_buf.getvalue())


def _supervised_trainer(
    args: Any,
    storage: Storage,
    full_dataset: pd.DataFrame,
    testset: pd.DataFrame | None,
    output_f: Any,
    filename: str,
    on_progress: ProgressCallback | None,
    log_cb: LogCallback | None,
    check_cancel: Callable[[], bool],
) -> None:
    """Supervised training loop — ported from trainer_func."""
    hascluster, othermodels = hasclustermodel(args)

    # Extract mapping column if present
    if args.class_column and f"{args.class_column}_mapping" in full_dataset.columns:
        df_mappings = full_dataset[
            [args.class_column, f"{args.class_column}_mapping"]
        ].drop_duplicates()
        mappings = df_mappings.set_index(args.class_column)[
            f"{args.class_column}_mapping"
        ].to_dict()
        full_dataset = full_dataset.drop([f"{args.class_column}_mapping"], axis=1)
        if testset is not None:
            testset = testset.drop([f"{args.class_column}_mapping"], axis=1)
    else:
        mappings = None

    # Split into train/test
    _log(f"Model training list: {args.train_group}", output_f, log_cb)
    if testset is not None:
        test_dataset = testset
        train_dataset = full_dataset
    else:
        train_dataset, test_dataset = train_test_split(
            full_dataset,
            test_size=args.test_size,
            stratify=full_dataset[args.class_column],
            random_state=args.random_state,
        )

    # Drop rows with missing values
    if train_dataset.isna().any().any():
        _log("Dropping training set rows with missing values.", output_f, log_cb)
        train_dataset = train_dataset.dropna()
    if test_dataset.isna().any().any():
        _log("Dropping testset rows with missing values.", output_f, log_cb)
        test_dataset = test_dataset.dropna()

    # Encode class labels
    train_dataset[args.class_column] = LabelEncoder().fit_transform(
        train_dataset[args.class_column]
    )
    test_dataset[args.class_column] = LabelEncoder().fit_transform(test_dataset[args.class_column])

    # Synthetic data generation
    if args.synthesize_original or args.synthesize_new:
        try:
            from ml.synthesize import build_dataset

            train_dataset = build_dataset(args, train_dataset, storage, filename)
        except Exception as e:
            _log(f"Synthesis error: {e}", output_f, log_cb)
            return

    all_columns = train_dataset.columns.tolist()
    all_columns.remove(args.class_column)

    model_results: list[dict] = []
    best_score = 0.0
    overall_positive_rates: dict[str, Any] = {}

    _log("Starting Training", output_f, log_cb)
    X = train_dataset[all_columns]
    y = train_dataset[args.class_column]
    test_y = test_dataset[args.class_column]
    test_X = test_dataset[all_columns]
    _log(f"Current column_list: {all_columns}", output_f, log_cb)

    total_models = len(args.train_group)
    for i, model in enumerate(args.train_group):
        if check_cancel():
            _log("Training cancelled by user.", output_f, log_cb)
            break

        if model == "tabpfn" and len(all_columns) > 100:
            _log("Skipping TabPFN — supports max 100 features", output_f, log_cb)
        elif model == "tabpfn" and not tabpfn_available():
            _log("Skipping TabPFN — addon not installed", output_f, log_cb)
        else:
            _log(f"Evaluating model: {model}", output_f, log_cb)
            column_key = str(all_columns).replace(",", "-").replace("'", "").replace(" ", "")
            try:
                results, positive_rates = estimator_evaluation(
                    args,
                    model,
                    X,
                    y,
                    storage,
                    output_f,
                    all_columns,
                    filename,
                    mappings,
                    test_X=test_X,
                    test_y=test_y,
                    log_cb=log_cb,
                )
                overall_positive_rates.update(positive_rates)

                if on_progress:
                    on_progress(i + 1, total_models, f"{i + 1}/{total_models} Processed")

                if results and len(results) > 0:
                    model_stats = {
                        "model": model,
                        "column_key": column_key,
                        "results": results,
                    }
                    if "best_score" in results:
                        model_best = results["best_score"]
                        if best_score < model_best:
                            best_score = model_best
                    model_results.append(model_stats)
            except Exception as e:
                _log(f"Error in train/evaluation of {model}: {e}", output_f, log_cb)
                traceback.print_exc()

    # Write report
    if model_results:
        try:
            write_report(args, model_results, storage, filename, overall_positive_rates)
        except Exception as e:
            _log(f"Error writing report: {e}", output_f, log_cb)
    else:
        _log("No model results to report.", output_f, log_cb)


def _unsupervised_trainer(
    args: Any,
    storage: Storage,
    full_dataset: pd.DataFrame,
    testset: pd.DataFrame | None,
    output_f: Any,
    filename: str,
    on_progress: ProgressCallback | None,
    log_cb: LogCallback | None,
    check_cancel: Callable[[], bool],
) -> None:
    """Unsupervised training loop — ported from uns_trainer_func."""
    train_dataset = full_dataset

    if train_dataset.isna().any().any():
        _log("Dropping rows with missing values.", output_f, log_cb)
        train_dataset = train_dataset.dropna()

    all_columns = train_dataset.columns.tolist()
    if args.class_column and args.class_column in all_columns:
        all_columns.remove(args.class_column)

    model_results: list[dict] = []
    best_score = 0.0

    _log("Starting Training", output_f, log_cb)
    X = train_dataset[all_columns]
    _log(f"Current column_list: {all_columns}", output_f, log_cb)

    total_models = len(args.train_group)
    for i, model in enumerate(args.train_group):
        if check_cancel():
            _log("Training cancelled by user.", output_f, log_cb)
            break

        _log(f"Evaluating model: {model}", output_f, log_cb)
        column_key = str(all_columns).replace(",", "-").replace("'", "").replace(" ", "")
        try:
            results = uns_estimator_evaluation(
                args,
                model,
                X,
                storage,
                output_f,
                all_columns,
                filename,
                log_cb=log_cb,
            )

            if on_progress:
                on_progress(i + 1, total_models, f"{i + 1}/{total_models} Processed")

            if results and len(results) > 0:
                model_stats = {
                    "model": model,
                    "column_key": column_key,
                    "results": results,
                }
                model_best = results.get("best_score", 0)
                if best_score < model_best:
                    best_score = model_best
                model_results.append(model_stats)
        except Exception as e:
            _log(f"Error in train/evaluation of {model}: {e}", output_f, log_cb)
            traceback.print_exc()

    if model_results:
        try:
            write_report(args, model_results, storage, filename, {})
        except Exception as e:
            _log(f"Error writing report: {e}", output_f, log_cb)
    else:
        _log("No model results to report.", output_f, log_cb)
