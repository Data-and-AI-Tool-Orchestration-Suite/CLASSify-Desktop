"""Re-test saved models on a new testset — ported from CLASSify-2's retest_file.py.

Loads joblib models from storage, applies the saved scaler, and computes
metrics on the new testset.  S3 calls replaced with storage calls.
"""

from __future__ import annotations

import io
from typing import Any

import joblib
import pandas as pd
from sklearn import metrics

from ml.column_types import detect_encoding
from ml.evaluate import multiclass_npv, multiclass_specificity
from storage.base import Storage


def get_testset(storage: Storage, filepath: str, class_column: str) -> tuple:
    """Load a testset from storage, split into X and y."""
    try:
        raw = storage.get_bytes(filepath)
        encoding = detect_encoding(raw)
        text = raw.decode(encoding)
        testset = pd.read_csv(io.StringIO(text))

        if "index" in testset.columns:
            testset = testset.drop(["index"], axis=1)
        if "Index" in testset.columns:
            testset = testset.drop(["Index"], axis=1)
        if class_column in testset.columns:
            test_y = testset[class_column]
            test_X = testset.drop([class_column], axis=1)
            return test_X, test_y
        return testset, []
    except Exception:
        return None, None


def read_model(
    storage: Storage,
    model_key: str,
    scaler_key: str,
    test_X: pd.DataFrame,
    test_y: Any,
    model_name: str,
) -> dict | None:
    """Load a model + scaler from storage and evaluate on the test set."""
    try:
        model_bytes = storage.get_bytes(model_key)
        model = joblib.load(io.BytesIO(model_bytes))
        scaler_bytes = storage.get_bytes(scaler_key)
        scaler = joblib.load(io.BytesIO(scaler_bytes))
    except Exception:
        print(f"Failed to read {model_name} from storage")
        return None

    test_X_scaled = scaler.transform(test_X.values)
    results: dict[str, Any] = {}

    if len(test_y) == 0:
        predictions = model.predict(test_X_scaled)
        results[model_name] = predictions
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(test_X_scaled)
            if proba.ndim == 1:
                results[f"{model_name}_proba"] = proba
            else:
                num_classes = proba.shape[1]
                if num_classes == 2:
                    results[f"{model_name}_proba"] = proba[:, 1]
                else:
                    for class_index in range(num_classes):
                        results[f"{model_name}_proba_class{class_index}"] = proba[:, class_index]
    else:
        predictions = model.predict(test_X_scaled)
        binary = len(set(test_y)) <= 2
        if binary:
            test_pred_proba = model.predict_proba(test_X_scaled)[:, 1]
            auc = float(metrics.roc_auc_score(test_y, test_pred_proba))
        else:
            test_pred_proba = model.predict_proba(test_X_scaled)
            try:
                auc = float(metrics.roc_auc_score(test_y, test_pred_proba, multi_class="ovr"))
            except Exception:
                auc = -1

        acc = float(metrics.accuracy_score(test_y, predictions))
        if binary:
            cm = metrics.confusion_matrix(test_y, predictions)
            tn, fp, fn, tp = cm.ravel()
            sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
            f1score = float(metrics.f1_score(test_y, predictions))
        else:
            sensitivity = float(metrics.recall_score(test_y, predictions, average="macro"))
            specificity = multiclass_specificity(test_y, predictions)
            ppv = float(metrics.precision_score(test_y, predictions, average="macro"))
            npv = multiclass_npv(test_y, predictions)
            f1score = float(metrics.f1_score(test_y, predictions, average="macro"))

        results["model"] = model_name
        results["test_auc"] = auc
        results["test_acc"] = acc
        results["test_sensitivity"] = sensitivity
        results["test_specificity"] = specificity
        results["test_npv"] = npv
        results["test_ppv"] = ppv
        results["test_f1score"] = f1score

    return results


def retest(
    storage: Storage,
    model_names: list[str],
    testset_key: str,
    class_column: str,
    dataset_prefix: str,
) -> dict:
    """Re-test multiple models on a new testset.

    Args:
        storage: Storage instance
        model_names: list of model names to re-test
        testset_key: storage key for the testset CSV
        class_column: name of the class column
        dataset_prefix: storage prefix (report_id) for models/scaler

    Returns: {"success": bool, "message": str}
    """
    test_X, test_y = get_testset(storage, testset_key, class_column)
    saving_summary = len(test_y) > 0

    if test_X is None:
        return {"success": False, "message": "Failed to retrieve testset from storage"}

    model_results: list[dict] = []
    scaler_key = f"{dataset_prefix}/scaler.joblib"

    for model_name in model_names:
        model_key = f"{dataset_prefix}/{model_name}_model.joblib"
        results = read_model(storage, model_key, scaler_key, test_X, test_y, model_name)
        if results is not None:
            model_results.append(results)

    if not saving_summary:
        model_results = [{k: v for d in model_results for k, v in d.items()}]

    try:
        results_df = pd.DataFrame(model_results)
        buf = io.StringIO()
        if saving_summary:
            results_df.to_csv(buf, index=False)
        else:
            results_df.to_csv(buf, index_label="testset_index")
        storage.put_text(f"{dataset_prefix}/retest_results", buf.getvalue())
        return {"success": True, "message": "Re-test completed successfully"}
    except Exception as e:
        return {"success": False, "message": str(e)}
