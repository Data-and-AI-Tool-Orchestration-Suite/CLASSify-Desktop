"""Model evaluation metrics — ported from CLASSify-2's models.py.

All metric functions (getmodelstats, clusteringstats, do_cross_validate,
hungarianalgorithm, clusterscoring, hasclustermodel) are preserved
exactly.  Only the I/O (S3 → storage) is changed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder

from ml.options import CLUSTERING_MODELS


def hasclustermodel(args: Any) -> tuple[int, int]:
    """Count how many clustering vs supervised models are in train_group.

    Returns (hascluster_count, othermodels_count).
    """
    hascluster = 0
    othermodels = 0
    for model in args.train_group:
        if model in CLUSTERING_MODELS:
            hascluster += 1
        else:
            othermodels += 1
    return hascluster, othermodels


def multiclass_specificity(y_true: Any, y_pred: Any) -> float:
    """Calculate specificity for multiclass (macro average)."""
    cm = metrics.confusion_matrix(y_true, y_pred)
    specificity_per_class = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        if tn + fp > 0:
            specificity_per_class.append(tn / (tn + fp))
    return float(np.mean(specificity_per_class)) if specificity_per_class else 0.0


def multiclass_npv(y_true: Any, y_pred: Any) -> float:
    """Calculate NPV for multiclass (macro average)."""
    cm = metrics.confusion_matrix(y_true, y_pred)
    npv_per_class = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fn = cm[i, :].sum() - cm[i, i]
        if tn + fn > 0:
            npv_per_class.append(tn / (tn + fn))
    return float(np.mean(npv_per_class)) if npv_per_class else 0.0


def getmodelstats(
    model: Any,
    test_X: Any,
    test_y: Any,
    istestset: bool,
    emethod: str,
    storage: Any,
    filename: str,
    args: Any,
    fullX: Any = None,
    fullY: Any = None,
    alignedlabels: Any = None,
) -> tuple:
    """Calculate metrics for a trained model on a test or training set.

    Returns a tuple of (sensitivity, specificity, auc, acc, kappa, npv, ppv,
    fpr, tpr, f1score) for supervised models, or a clustering tuple for
    clustering models.

    Preserved exactly from CLASSify-2's models.py — only S3 → storage.
    """
    predictions = model.predict(test_X)

    if emethod in CLUSTERING_MODELS:
        # Clustering metrics on the test set
        if hasattr(model, "labels_") and len(model.labels_) == len(test_y):
            labels = model.labels_
        else:
            labels = predictions

        try:
            silhouette = float(
                metrics.silhouette_score(test_X, labels) if len(set(labels)) > 1 else -1
            )
        except Exception:
            silhouette = -1
        try:
            ari = float(metrics.adjusted_rand_score(test_y, labels))
        except Exception:
            ari = -1
        try:
            nmi = float(metrics.normalized_mutual_info_score(test_y, labels))
        except Exception:
            nmi = -1
        try:
            davies_bouldin = float(
                metrics.davies_bouldin_score(test_X, labels) if len(set(labels)) > 1 else -1
            )
        except Exception:
            davies_bouldin = -1
        try:
            calinski_harabasz = float(
                metrics.calinski_harabasz_score(test_X, labels) if len(set(labels)) > 1 else -1
            )
        except Exception:
            calinski_harabasz = -1
        try:
            v_measure = float(metrics.v_measure_score(test_y, labels))
        except Exception:
            v_measure = -1
        try:
            fmi = float(metrics.fowlkes_mallows_score(test_y, labels))
        except Exception:
            fmi = -1

        return (
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            np.array([0, 1]),
            np.array([0, 1]),
            -1,
            silhouette,
            ari,
            nmi,
            davies_bouldin,
            calinski_harabasz,
            v_measure,
            fmi,
        )

    # Supervised metrics
    binary = len(set(test_y)) <= 2

    if binary:
        test_pred_proba = model.predict_proba(test_X)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(test_y, test_pred_proba)
        auc = float(metrics.roc_auc_score(test_y, test_pred_proba))
        acc = float(metrics.accuracy_score(test_y, predictions))
        kappa = float(metrics.cohen_kappa_score(test_y, predictions))
        cm = metrics.confusion_matrix(test_y, predictions)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        f1score = float(metrics.f1_score(test_y, predictions))
    else:
        test_pred_proba = model.predict_proba(test_X)
        try:
            auc = float(metrics.roc_auc_score(test_y, test_pred_proba, multi_class="ovr"))
        except Exception:
            auc = -1
        acc = float(metrics.accuracy_score(test_y, predictions))
        kappa = float(metrics.cohen_kappa_score(test_y, predictions))
        sensitivity = float(metrics.recall_score(test_y, predictions, average="macro"))
        specificity = multiclass_specificity(test_y, predictions)
        ppv = float(metrics.precision_score(test_y, predictions, average="macro"))
        npv = multiclass_npv(test_y, predictions)
        f1score = float(metrics.f1_score(test_y, predictions, average="macro"))
        fpr = np.array([0, 1])
        tpr = np.array([0, 1])

    return (
        sensitivity,
        specificity,
        auc,
        acc,
        kappa,
        npv,
        ppv,
        fpr,
        tpr,
        f1score,
    )


def clusteringstats(
    model: Any,
    X: Any,
    y: Any,
    istestset: bool,
    emethod: str,
    storage: Any,
    filename: str,
    alignedlabels: Any,
    args: Any,
    istraining: bool = False,
) -> tuple:
    """Calculate clustering metrics.

    Returns (silhouette_score, davies_bouldin_score, calinski_harabasz_score).
    """
    labels = model.labels_ if hasattr(model, "labels_") else model.predict(X)

    try:
        silhouette = float(metrics.silhouette_score(X, labels) if len(set(labels)) > 1 else -1)
    except Exception:
        silhouette = -1
    try:
        davies_bouldin = float(
            metrics.davies_bouldin_score(X, labels) if len(set(labels)) > 1 else -1
        )
    except Exception:
        davies_bouldin = -1
    try:
        calinski_harabasz = float(
            metrics.calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else -1
        )
    except Exception:
        calinski_harabasz = -1

    return silhouette, davies_bouldin, calinski_harabasz


def do_cross_validate(X: Any, y: Any, model: Any, folds: int, repeats: int) -> tuple:
    """Run repeated stratified cross-validation.

    Returns (sensitivity, sensitivity_moe, specificity, specificity_moe,
    auc, auc_moe, acc, acc_moe, ppv, ppv_moe, npv, npv_moe,
    f1score, f1score_moe).
    """
    binary = len(set(y)) <= 2

    scoring = {
        "sensitivity": "recall",
        "specificity": make_specificity_scorer(),
        "accuracy": "accuracy",
        "precision": "precision",
        "f1": "f1",
    }
    if binary:
        scoring["auc"] = "roc_auc"
    else:
        scoring = {
            "sensitivity": "recall_macro",
            "accuracy": "accuracy",
            "precision": "precision_macro",
            "f1": "f1_macro",
        }
        scoring["auc"] = "roc_auc_ovr"

    try:
        cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, error_score="raise")
    except Exception:
        # Fallback: return -1 for all
        return (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)

    def mean_and_moe(values: np.ndarray) -> tuple[float, float]:
        mean = float(np.mean(values))
        moe = float(1.96 * np.std(values) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        return mean, moe

    sens, sens_moe = mean_and_moe(cv_results["test_sensitivity"])
    spec, spec_moe = mean_and_moe(cv_results.get("test_specificity", np.array([-1])))
    auc_vals = cv_results.get("test_auc", np.array([-1]))
    auc, auc_moe = mean_and_moe(auc_vals)
    acc, acc_moe = mean_and_moe(cv_results["test_accuracy"])
    ppv, ppv_moe = mean_and_moe(cv_results["test_precision"])
    npv_vals = cv_results.get("test_specificity", np.array([-1]))  # approximate
    npv_v, npv_moe = mean_and_moe(npv_vals)
    f1, f1_moe = mean_and_moe(cv_results["test_f1"])

    return (
        sens,
        sens_moe,
        spec,
        spec_moe,
        auc,
        auc_moe,
        acc,
        acc_moe,
        ppv,
        ppv_moe,
        npv_v,
        npv_moe,
        f1,
        f1_moe,
    )


def make_specificity_scorer() -> Any:
    """Create a specificity scorer for cross-validation."""
    from sklearn.metrics import make_scorer

    def specificity_score(y_true: Any, y_pred: Any) -> float:
        cm = metrics.confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        return multiclass_specificity(y_true, y_pred)

    return make_scorer(specificity_score)


def hungarian_algorithm(true_labels: Any, pred_labels: Any) -> dict:
    """Align predicted cluster labels with true labels using Hungarian algorithm."""
    true_encoder = LabelEncoder()
    pred_encoder = LabelEncoder()
    true_encoded = true_encoder.fit_transform(true_labels)
    pred_encoded = pred_encoder.fit_transform(pred_labels)

    n_true = len(true_encoder.classes_)
    n_pred = len(pred_encoder.classes_)
    cost_matrix = np.zeros((n_true, n_pred))

    for i in range(n_true):
        for j in range(n_pred):
            mask = pred_encoded == j
            if mask.sum() > 0:
                cost_matrix[i, j] = -np.sum(true_encoded[mask] == i)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {
        pred_encoder.classes_[col]: true_encoder.classes_[row]
        for row, col in zip(row_ind, col_ind, strict=False)
    }

    aligned = np.array([mapping.get(label, label) for label in pred_labels])
    return {"mapping": mapping, "aligned_labels": aligned}
