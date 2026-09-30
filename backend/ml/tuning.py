"""Optuna hyperparameter tuning objective — ported from CLASSify-2's models.py.

The objective() function is preserved exactly; only the S3/ClearML
references are removed.  It builds model instances with sampled
hyperparameters and evaluates them using cross-validation.
"""

from __future__ import annotations

import contextlib
import time
from functools import partial
from typing import Any, Literal

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score

# Models whose estimators parallelize internally (joblib/OMP) — adding
# cross-validation parallelism on top would oversubscribe the CPU. For the
# rest, running folds in parallel is a large, results-identical speedup.
_INTERNAL_PARALLEL_MODELS = {"randomforest", "bagging", "kneighbors", "xgboost"}


def objective(
    trial: optuna.Trial,
    emethod: str,
    args: Any,
    X: Any,
    y: Any,
    output_f: Any,
    log_cb: Any = None,
) -> float:
    """Optuna objective function for hyperparameter tuning.

    Samples hyperparameters based on the model type and args ranges,
    then returns the cross-validated score to maximize (or minimize
    for davies_bouldin).
    """
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.ensemble import (
        BaggingClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    params: dict[str, Any] = {}

    if emethod == "randomforest":
        params["n_estimators"] = trial.suggest_int(
            "n_estimators",
            args.n_estimators_start,
            args.n_estimators_stop,
            step=args.n_estimators_step,
        )
        params["max_depth"] = trial.suggest_int(
            "max_depth", args.max_depth_start, args.max_depth_stop, step=args.max_depth_step
        )
        params["max_features"] = trial.suggest_categorical("max_features", args.max_features)
        params["min_samples_split"] = trial.suggest_categorical(
            "min_samples_split", args.min_samples_split
        )
        params["min_samples_leaf"] = trial.suggest_categorical(
            "min_samples_leaf", args.min_samples_leaf
        )
        params["bootstrap"] = trial.suggest_categorical("bootstrap", args.bootstrap)
        estimator = RandomForestClassifier(
            random_state=args.random_state, n_jobs=args.n_jobs, **params
        )

    elif emethod == "xgboost":
        import xgboost as xgb

        params["n_estimators"] = trial.suggest_int(
            "n_estimators",
            args.n_estimators_start,
            args.n_estimators_stop,
            step=args.n_estimators_step,
        )
        params["max_depth"] = trial.suggest_int(
            "max_depth", args.max_depth_start, args.max_depth_stop, step=args.max_depth_step
        )
        params["subsample"] = trial.suggest_float(
            "subsample", args.subsample_start, args.subsample_stop, step=args.subsample_step
        )
        estimator = xgb.XGBClassifier(
            random_state=args.random_state, n_jobs=args.n_jobs, eval_metric="logloss", **params
        )

    elif emethod == "gradientboosting":
        params["n_estimators"] = trial.suggest_int(
            "n_estimators",
            args.n_estimators_start,
            args.n_estimators_stop,
            step=args.n_estimators_step,
        )
        params["max_depth"] = trial.suggest_int(
            "max_depth", args.max_depth_start, args.max_depth_stop, step=args.max_depth_step
        )
        params["subsample"] = trial.suggest_float(
            "subsample", args.subsample_start, args.subsample_stop, step=args.subsample_step
        )
        params["validation_fraction"] = trial.suggest_float(
            "validation_fraction",
            args.validation_fraction_start,
            args.validation_fraction_stop,
            step=args.validation_fraction_step,
        )
        params["n_iter_no_change"] = trial.suggest_int(
            "n_iter_no_change",
            args.n_iter_no_change_start,
            args.n_iter_no_change_stop,
            step=args.n_iter_no_change_step,
        )
        estimator = GradientBoostingClassifier(random_state=args.random_state, **params)

    elif emethod == "histgradientboosting":
        params["learning_rate"] = trial.suggest_float(
            "learning_rate",
            args.learning_rate_start,
            args.learning_rate_stop,
            step=args.learning_rate_step,
        )
        params["max_iter"] = trial.suggest_int(
            "max_iter", args.n_estimators_start, args.n_estimators_stop, step=args.n_estimators_step
        )
        params["validation_fraction"] = trial.suggest_float(
            "validation_fraction",
            args.validation_fraction_start,
            args.validation_fraction_stop,
            step=args.validation_fraction_step,
        )
        params["n_iter_no_change"] = trial.suggest_int(
            "n_iter_no_change",
            args.n_iter_no_change_start,
            args.n_iter_no_change_stop,
            step=args.n_iter_no_change_step,
        )
        estimator = HistGradientBoostingClassifier(random_state=args.random_state, **params)

    elif emethod == "neuralnetwork":
        hidden_size = trial.suggest_int(
            "nn_hidden_layer_sizes",
            args.nn_hidden_layer_sizes_start,
            args.nn_hidden_layer_sizes_stop,
            step=args.nn_hidden_layer_sizes_step,
        )
        hidden_depth = trial.suggest_int(
            "nn_hidden_layer_depth",
            args.nn_hidden_layer_depth_start,
            args.nn_hidden_layer_depth_stop,
            step=args.nn_hidden_layer_depth_step,
        )
        params["hidden_layer_sizes"] = tuple([hidden_size] * hidden_depth)
        params["learning_rate_init"] = trial.suggest_float(
            "nn_learning_rate",
            args.nn_learning_rate_start,
            args.nn_learning_rate_stop,
            step=args.nn_learning_rate_step,
        )
        params["alpha"] = trial.suggest_float(
            "alpha", args.alpha_start, args.alpha_stop, step=args.alpha_step
        )
        estimator = MLPClassifier(random_state=args.random_state, max_iter=500, **params)

    elif emethod == "sgdclassifier":
        params["alpha"] = trial.suggest_float(
            "alpha", args.alpha_start, args.alpha_stop, step=args.alpha_step
        )
        params["validation_fraction"] = trial.suggest_float(
            "validation_fraction",
            args.validation_fraction_start,
            args.validation_fraction_stop,
            step=args.validation_fraction_step,
        )
        params["n_iter_no_change"] = trial.suggest_int(
            "n_iter_no_change",
            args.n_iter_no_change_start,
            args.n_iter_no_change_stop,
            step=args.n_iter_no_change_step,
        )
        estimator = SGDClassifier(random_state=args.random_state, **params)

    elif emethod == "logisticregression":
        params["C"] = trial.suggest_float("C", args.c_start, args.c_stop, step=args.c_step)
        estimator = LogisticRegression(random_state=args.random_state, max_iter=1000, **params)

    elif emethod == "bagging":
        params["n_estimators"] = trial.suggest_int(
            "n_estimators",
            args.n_estimators_start,
            args.n_estimators_stop,
            step=args.n_estimators_step,
        )
        params["max_samples"] = trial.suggest_float(
            "subsample", args.subsample_start, args.subsample_stop, step=args.subsample_step
        )
        estimator = BaggingClassifier(random_state=args.random_state, n_jobs=args.n_jobs, **params)

    elif emethod == "kneighbors":
        n_neighbors = trial.suggest_int("n_neighbors", 3, 50)
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=args.n_jobs)

    elif emethod == "spectralclustering":
        params["n_clusters"] = trial.suggest_int(
            "num_clusters",
            args.num_clusters_start,
            args.num_clusters_stop,
            step=args.num_clusters_step,
        )
        params["affinity"] = trial.suggest_categorical("affinity", args.affinity_method)
        if params["affinity"] == "nearest_neighbors":
            params["n_neighbors"] = trial.suggest_int(
                "nearest_neighbors",
                args.nearest_neighbors_start,
                args.nearest_neighbors_stop,
                step=args.nearest_neighbors_step,
            )
        estimator = SpectralClustering(random_state=args.random_state, **params)

    elif emethod == "kmeans":
        params["n_clusters"] = trial.suggest_int(
            "num_clusters",
            args.num_clusters_start,
            args.num_clusters_stop,
            step=args.num_clusters_step,
        )
        params["max_iter"] = trial.suggest_int(
            "max_iter", args.max_iter_start, args.max_iter_stop, step=args.max_iter_step
        )
        params["n_init"] = trial.suggest_int(
            "n_init", args.n_init_start, args.n_init_stop, step=args.n_init_step
        )
        estimator = KMeans(random_state=args.random_state, **params)

    elif emethod == "hdbscan":
        from hdbscan import HDBSCAN

        params["min_cluster_size"] = trial.suggest_int(
            "min_cluster_size",
            args.min_cluster_size_start,
            args.min_cluster_size_stop,
            step=args.min_cluster_size_step,
        )
        params["min_samples"] = trial.suggest_int(
            "min_samples", args.min_samples_start, args.min_samples_stop, step=args.min_samples_step
        )
        params["cluster_selection_epsilon"] = trial.suggest_float(
            "cluster_selection_epsilon",
            args.cluster_selection_epsilon_start,
            args.cluster_selection_epsilon_stop,
            step=args.cluster_selection_epsilon_step,
        )
        estimator = HDBSCAN(**params)
    else:
        raise ValueError(f"Unknown model: {emethod}")

    # For clustering, compute the optimization metric directly
    if emethod in ("spectralclustering", "kmeans", "hdbscan"):
        from sklearn import metrics as skmetrics

        # davies_bouldin is minimized (lower is better), so a failed trial
        # must score WORST (1000.0); the maximize-style metrics score -1.0.
        goal = args.clustering_parameter_goal[0] if args.clustering_parameter_goal else ""
        failure_score = 1000.0 if goal == "davies_bouldin_score" else -1.0
        try:
            labels = estimator.fit_predict(X)
            if len(set(labels)) <= 1:
                return failure_score
            if goal == "silhouette_score":
                return float(skmetrics.silhouette_score(X, labels))
            elif goal == "calinski_harabasz_score":
                return float(skmetrics.calinski_harabasz_score(X, labels))
            elif goal == "davies_bouldin_score":
                return float(skmetrics.davies_bouldin_score(X, labels))
            return failure_score
        except Exception:
            return failure_score

    # For supervised, use cross-validated score
    goal = args.parameter_goal[0] if hasattr(args, "parameter_goal") else "f1_macro"
    scoring_map = {
        "f1_macro": "f1_macro",
        "f1": "f1",
        "precision_macro": "precision_macro",
        "precision": "precision",
        "accuracy": "accuracy",
        "recall_macro": "recall_macro",
        "roc_auc": "roc_auc",
    }
    scoring = scoring_map.get(goal, "f1_macro")

    try:
        # Sequential-fold estimators get parallel folds — a large speedup
        # with identical results. (n_jobs is a scheduling knob, not math.)
        cv_n_jobs = 1 if emethod in _INTERNAL_PARALLEL_MODELS else args.n_jobs
        scores = cross_val_score(
            estimator,
            X,
            y,
            cv=min(args.folds, 5),
            scoring=scoring,
            error_score="raise",
            n_jobs=cv_n_jobs,
        )
        return float(np.mean(scores))
    except Exception as e:
        if hasattr(output_f, "write"):
            output_f.write(f"Tuning trial failed: {e}\n")
        if log_cb is not None:
            with contextlib.suppress(Exception):
                log_cb(f"Tuning trial failed for {emethod}: {e}")
        return 0.0


def _convert_best_params(emethod: str, best_params: dict[str, Any]) -> dict[str, Any]:
    """Convert Optuna trial param names to sklearn constructor param names."""
    if emethod == "neuralnetwork":
        hidden_size = best_params.pop("nn_hidden_layer_sizes")
        hidden_depth = best_params.pop("nn_hidden_layer_depth")
        best_params["hidden_layer_sizes"] = tuple([hidden_size] * hidden_depth)
        best_params["learning_rate_init"] = best_params.pop("nn_learning_rate")
    elif emethod == "bagging":
        best_params["max_samples"] = best_params.pop("subsample")
    elif emethod in ("spectralclustering", "kmeans"):
        best_params["n_clusters"] = best_params.pop("num_clusters")
    return best_params


def run_tuning(
    emethod: str,
    args: Any,
    X: Any,
    y: Any,
    output_f: Any,
    log_cb: Any = None,
) -> tuple[dict[str, Any], float]:
    """Run Optuna tuning and return (best_params, best_score)."""
    direction: Literal["minimize", "maximize"] = "maximize"
    if (
        hasattr(args, "clustering_parameter_goal")
        and args.clustering_parameter_goal
        and args.clustering_parameter_goal[0] == "davies_bouldin_score"
    ):
        direction = "minimize"

    def _log(message: str) -> None:
        with contextlib.suppress(Exception):
            if hasattr(output_f, "write"):
                output_f.write(message + "\n")
        if log_cb is not None:
            with contextlib.suppress(Exception):
                log_cb(message)

    study = optuna.create_study(
        direction=direction, sampler=optuna.samplers.TPESampler(seed=args.random_state)
    )
    objective_with_params = partial(
        objective, emethod=emethod, args=args, X=X, y=y, output_f=output_f, log_cb=log_cb
    )

    # Optuna swallows trial exceptions when catch is set; record them so an
    # all-trials-failed study can be reported with the real cause instead of
    # Optuna's cryptic "No trials are completed yet."
    trial_errors: list[str] = []
    started = time.monotonic()

    def tracked_objective(trial: optuna.Trial) -> float:
        trial_start = time.monotonic()
        try:
            value = objective_with_params(trial)
            _log(
                f"Tuning {emethod}: trial {trial.number + 1}/{args.n_iter} complete "
                f"— score {value:.3f} ({time.monotonic() - trial_start:.1f}s)"
            )
            return value
        except Exception as e:
            message = f"{type(e).__name__}: {e}"
            trial_errors.append(message)
            _log(f"Tuning trial failed: {message}")
            raise

    def _progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        elapsed_min = (time.monotonic() - started) / 60
        _log(
            f"Tuning {emethod}: {trial.number + 1}/{args.n_iter} trials ({elapsed_min:.1f} min elapsed)"
        )

    study.optimize(
        tracked_objective, n_trials=args.n_iter, catch=(Exception,), callbacks=[_progress_callback]
    )
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        last_error = trial_errors[-1] if trial_errors else "unknown error"
        raise ValueError(
            f"Parameter tuning failed for {emethod} — all {args.n_iter} trials failed. "
            f"Last error: {last_error}"
        )
    best_params = _convert_best_params(emethod, dict(study.best_trial.params))
    return best_params, float(study.best_value)
