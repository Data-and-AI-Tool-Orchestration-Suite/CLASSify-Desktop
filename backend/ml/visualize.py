"""Visualization generators — ported from CLASSify-2's visualization.py.

All chart functions are preserved exactly; only the S3 save calls are
replaced with ``storage.put_bytes``.
"""

from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from sklearn import metrics as skmetrics  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from ml.options import CLUSTERING_MODELS  # noqa: E402
from storage.base import Storage  # noqa: E402


def _save_png(storage: Storage, key: str, fig: Any) -> None:
    """Save a matplotlib figure to storage as PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    storage.put_bytes(key, buf.getvalue())
    plt.close(fig)


def plot_clusters(
    storage: Storage,
    filename: str,
    X: Any,
    labels: Any,
    model_name: str,
    hasnoise: bool = True,
    istestset: bool = True,
    title: str = "Cluster Visualization",
) -> None:
    """PCA-reduced cluster visualization."""
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if -1 in unique_labels:
        n_clusters -= 1
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))  # type: ignore[attr-defined]

    for i, label in enumerate(unique_labels):
        if label == -1:
            if hasnoise:
                ax.scatter(
                    pca_data[labels == label, 0],
                    pca_data[labels == label, 1],
                    color="gray",
                    s=50,
                    alpha=0.6,
                    edgecolors="k",
                    label="Noise",
                )
        elif -1 in unique_labels:
            ax.scatter(
                pca_data[labels == label, 0],
                pca_data[labels == label, 1],
                color=colors[i - 1],
                s=50,
                alpha=0.6,
                edgecolors="k",
                label=f"Cluster {label}",
            )
        else:
            ax.scatter(
                pca_data[labels == label, 0],
                pca_data[labels == label, 1],
                color=colors[i],
                s=50,
                alpha=0.6,
                edgecolors="k",
                label=f"Cluster {label}",
            )

    ax.set_title(title)
    ax.set_xlabel(f"PCA Feature 1 ({pca.explained_variance_ratio_[0]:.2%} Variance Explained)")
    ax.set_ylabel(f"PCA Feature 2 ({pca.explained_variance_ratio_[1]:.2%} Variance Explained)")
    ax.legend()
    ax.grid(True)

    viz_key = f"{filename}/viz/Model{'Test' if istestset else 'Train'}Cluster_{model_name}"
    _save_png(storage, viz_key, fig)


def model_metric_heatmap(
    df: pd.DataFrame, storage: Storage, filename: str, supervised: bool, choice: int = 1
) -> None:
    """Generate the model metric heatmap."""
    try:
        if supervised:
            if choice == 2:
                subset = df[
                    [
                        "model",
                        "trt_auc",
                        "trt_acc",
                        "trt_sensitivity",
                        "trt_specificity",
                        "trt_npv",
                        "trt_ppv",
                        "trt_f1score",
                    ]
                ]
            else:
                subset = df[
                    [
                        "model",
                        "test_auc",
                        "test_acc",
                        "test_sensitivity",
                        "test_specificity",
                        "test_npv",
                        "test_ppv",
                        "test_f1score",
                    ]
                ]

            if subset["test_auc"].notnull().any():
                custom_labels = [
                    "AUC",
                    "Accuracy",
                    "Sensitivity",
                    "Specificity",
                    "NPV",
                    "PPV",
                    "F1 Score",
                ]
            else:
                subset = subset.drop(["test_auc"], axis=1)
                custom_labels = ["Accuracy", "Sensitivity", "Specificity", "NPV", "PPV", "F1 Score"]
        else:
            subset = df[
                ["model", "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]
            ]
            custom_labels = ["Silhouette Score", "Calinski Harabasz Score", "Davies Bouldin Score"]

        subset = subset.set_index("model")
        g1 = subset.T
        inverted = ["davies_bouldin_score"]
        fig, ax = plt.subplots(figsize=(12, 7))

        if not supervised:
            scaled_g1 = g1.copy()
            for i in range(scaled_g1.shape[0]):
                row = scaled_g1.iloc[i]
                min_val = row.min()
                max_val = row.max()
                if max_val - min_val != 0:
                    scaled_g1.iloc[i] = (row - min_val) / (max_val - min_val)
                else:
                    scaled_g1.iloc[i] = 0
            for row_index in inverted:
                if row_index in scaled_g1.index:
                    scaled_g1.loc[row_index] = 1 - scaled_g1.loc[row_index]
            ax = sns.heatmap(
                scaled_g1,
                annot=g1,
                cmap="RdYlGn",
                fmt=".3f",
                linewidths=0.5,
                yticklabels=custom_labels,
                cbar_kws={"ticks": [0, 1]},
            )
            ax.collections[0].colorbar.set_ticklabels(["Worse", "Better"])
        else:
            ax = sns.heatmap(
                g1, annot=True, cmap="RdYlGn", fmt=".3f", linewidths=0.5, yticklabels=custom_labels
            )

        row_max = g1.idxmax(axis=1)
        row_min = g1.idxmin(axis=1)
        for index, row in enumerate(row_max.index):
            value = g1.loc[row][row_min[row]] if row in inverted else g1.loc[row][row_max[row]]  # noqa: SIM108
            column_names = g1.columns[g1.loc[row] == value].tolist()
            for column in column_names:
                position = g1.columns.get_loc(column)
                ax.add_patch(
                    Rectangle(
                        (position + 0.05, index + 0.05),
                        0.9,
                        0.9,
                        fill=False,
                        edgecolor="black",
                        lw=3,
                    )
                )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_title("Accuracy Metrics and Models")
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy Metric")
        ax.tick_params(axis="y", rotation=0)

        _save_png(storage, f"{filename}/viz/ModelMetricHeatmap", fig)
    except Exception as e:
        print(f"Heatmap error: {e}")


def model_metric_heatmap_multiclass(df: pd.DataFrame, storage: Storage, filename: str) -> None:
    """Generate the multiclass model metric heatmap."""
    try:
        subset = df[
            [
                "model",
                "test_auc",
                "test_acc",
                "test_kappa",
                "test_sensitivity",
                "test_specificity",
                "test_f1score",
                "trt_auc",
                "trt_acc",
                "trt_kappa",
                "trt_sensitivity",
                "trt_specificity",
                "trt_f1score",
            ]
        ]
        if subset["test_auc"].notnull().any():
            custom_labels = [
                "Test AUC",
                "Test Accuracy",
                "Test Kappa",
                "Test Sensitivity",
                "Test Specificity",
                "Test F1 Score",
                "Training AUC",
                "Training Accuracy",
                "Training Kappa",
                "Training Sensitivity",
                "Training Specificity",
                "Training F1 Score",
            ]
        else:
            subset = subset.drop(["test_auc", "trt_auc"], axis=1)
            custom_labels = [
                "Test Accuracy",
                "Test Kappa",
                "Test Precision",
                "Test Recall",
                "Test F1 Score",
                "Training Accuracy",
                "Training Kappa",
                "Training Precision",
                "Training Recall",
                "Training F1 Score",
            ]
        subset = subset.set_index("model")
        g1 = subset.T
        fig, ax = plt.subplots(figsize=(12, 7))
        ax = sns.heatmap(
            g1, annot=True, cmap="RdYlGn", fmt=".3f", linewidths=0.5, yticklabels=custom_labels
        )
        row_max = g1.idxmax(axis=1)
        for index, row in enumerate(row_max.index):
            value = g1.loc[row][row_max[row]]
            column_names = g1.columns[g1.loc[row] == value].tolist()
            for column in column_names:
                position = g1.columns.get_loc(column)
                ax.add_patch(
                    Rectangle(
                        (position + 0.05, index + 0.05),
                        0.9,
                        0.9,
                        fill=False,
                        edgecolor="black",
                        lw=3,
                    )
                )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_title("Accuracy Metrics and Models")
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy Metric")
        ax.tick_params(axis="y", rotation=0)

        _save_png(storage, f"{filename}/viz/ModelMetricHeatmapMulticlass", fig)
    except Exception as e:
        print(f"Multiclass heatmap error: {e}")


def true_false_rates(df: pd.DataFrame, storage: Storage, filename: str, choice: int = 1) -> None:
    """Generate the true/false positive/negative rates bar chart."""
    if choice == 2:
        data = df[["model", "trt_sensitivity", "trt_specificity"]]
        g1 = data.groupby(["model"]).mean()
        g1["tpr"] = g1["trt_sensitivity"]
        g1["fpr"] = 1 - g1["trt_specificity"]
        g1["tnr"] = g1["trt_specificity"]
        g1["fnr"] = 1 - g1["trt_sensitivity"]
        g1 = g1.drop("trt_sensitivity", axis=1).drop("trt_specificity", axis=1)
    else:
        data = df[["model", "test_sensitivity", "test_specificity"]]
        g1 = data.groupby(["model"]).mean()
        g1["tpr"] = g1["test_sensitivity"]
        g1["fpr"] = 1 - g1["test_specificity"]
        g1["tnr"] = g1["test_specificity"]
        g1["fnr"] = 1 - g1["test_sensitivity"]
        g1 = g1.drop("test_sensitivity", axis=1).drop("test_specificity", axis=1)

    bar_width = 0.15
    index = np.arange(len(g1))
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(index, g1["tpr"], width=bar_width, label="True Positive", color="blue")
    ax.bar(index, g1["fnr"], bottom=g1["tpr"], width=bar_width, label="False Negative", color="red")
    ax.bar(index + 2 * bar_width, g1["tnr"], width=bar_width, label="True Negative", color="green")
    ax.bar(
        index + 2 * bar_width,
        g1["fpr"],
        bottom=g1["tnr"],
        width=bar_width,
        label="False Positive",
        color="orange",
    )
    ax.set_xlabel("Models")
    ax.set_ylabel("Rate")
    ax.set_title("Rates for Each Model")
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(g1.index, rotation=20)
    ax.legend(bbox_to_anchor=(1, 1.11), loc="best", ncol=2)

    _save_png(storage, f"{filename}/viz/TrueFalseRates", fig)


def new_bars(df: pd.DataFrame, storage: Storage, filename: str, multiclass: bool) -> None:
    """Generate the AUC comparison bar chart (test/train/CV)."""
    df = df.copy()
    if multiclass:
        columns = ["cvt_acc", "cvt_auc", "cvt_sensitivity", "cvt_specificity", "cvt_f1score"]
    else:
        columns = [
            "cvt_acc",
            "cvt_sensitivity",
            "cvt_specificity",
            "cvt_auc",
            "cvt_ppv",
            "cvt_npv",
            "cvt_f1score",
        ]

    for col in columns:
        df[col] = df[col].replace("N/A", "0 ± 0")
        split_values = df[col].astype(str).str.split(" ± ", expand=True)
        df[col] = split_values[0].astype(float)
        df[f"{col}_err"] = split_values[1].astype(float)

    data = df[["model", "test_auc", "trt_auc", "cvt_auc", "cvt_auc_err"]]
    g1 = data.groupby(["model"]).mean()

    if "spectralclustering" in g1.index:
        g1 = g1.drop("spectralclustering", axis=0)
    if "kmeans" in g1.index:
        g1 = g1.drop("kmeans", axis=0)

    bar_width = 0.15
    index = np.arange(len(g1))
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(index - 0.5 * bar_width, g1["test_auc"], width=bar_width, label="Test AUC", color="blue")
    ax.bar(index + 1 * bar_width, g1["trt_auc"], width=bar_width, label="Train AUC", color="green")
    ax.bar(
        index + 2.5 * bar_width,
        g1["cvt_auc"],
        width=bar_width,
        label="Cross Validation AUC",
        color="purple",
        yerr=g1["cvt_auc_err"],
        capsize=5,
    )
    ax.set_xlabel("Models")
    ax.set_ylabel("Rate")
    ax.set_title("AUC for Each Model")
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(g1.index, rotation=20)
    ax.legend(bbox_to_anchor=(1, 1.11), loc="best", ncol=2)

    _save_png(storage, f"{filename}/viz/TrainTestCV", fig)


def create_roc(positive_rates: dict, storage: Storage, filename: str) -> None:
    """Generate the overall ROC curve comparing all models."""
    positive_rates = {k: v for k, v in positive_rates.items() if k not in CLUSTERING_MODELS}
    if not positive_rates:
        return

    model_aucs = [
        (model, skmetrics.auc(pr["fpr"], pr["tpr"])) for model, pr in positive_rates.items()
    ]
    model_aucs = sorted(model_aucs, key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot([0, 1], [0, 1], "k--")
    for i, (model, auc) in enumerate(model_aucs):
        ax.plot(
            positive_rates[model]["fpr"],
            positive_rates[model]["tpr"],
            label=f"{model} (area = {auc:.3f})",
            color=f"C{i}",
        )
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Overall")
    ax.legend(loc="lower right")

    _save_png(storage, f"{filename}/viz/ROC_curve", fig)


def best_score_chart(df: pd.DataFrame, storage: Storage, filename: str) -> None:
    """Generate the best parameter tuning score bar chart."""
    df = df[["model", "best_score"]]
    g1 = (
        df.groupby(["model"])["best_score"]
        .mean()
        .reset_index(name="mean")
        .sort_values(["mean"], ascending=False)
    )
    g1 = g1.set_index("model")
    fig, ax = plt.subplots(figsize=(12, 7))
    g1.plot(
        kind="bar", title="Best Score for Each Model", ylabel="Best Score", xlabel="Model", ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

    _save_png(storage, f"{filename}/viz/BestScore", fig)


def visualize(
    df: pd.DataFrame,
    storage: Storage,
    filename: str,
    args: Any,
    positive_rates: dict,
    multiclass: bool,
) -> None:
    """Generate all visualizations for a completed report.

    Mirrors CLASSify-2's visualize() function — calls the individual
    chart generators based on supervised/unsupervised mode.
    """
    columns_to_skip = ["model", "features"]
    for col in df.columns:
        if col[:3] == "cvt":
            columns_to_skip.append(col)

    columns_to_convert = [col for col in df.columns if col not in columns_to_skip]
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors="coerce")

    if args.supervised:
        true_false_rates(df, storage, filename)
        if not multiclass:
            model_metric_heatmap(df, storage, filename, True)
        else:
            model_metric_heatmap_multiclass(df, storage, filename)

        from ml.evaluate import hasclustermodel

        hascluster, othermodels = hasclustermodel(args)

        if othermodels != 0:
            if multiclass:
                new_bars(df, storage, filename, True)
            else:
                new_bars(df, storage, filename, False)

            if args.parameter_tune:
                best_score_chart(df, storage, filename)
            if len(df[df["test_auc"].notnull()]) >= 2:
                create_roc(positive_rates, storage, filename)
    else:
        model_metric_heatmap(df, storage, filename, False)


def label_file(model_results: list[dict], storage: Storage, filename: str) -> None:
    """Save cluster labels for unsupervised models."""
    result_df = pd.DataFrame()
    for result in model_results:
        model_name = result["model"]
        labels = result["results"].get("labels", None)
        if labels is not None:
            col_name = f"{model_name} labels"
            result_df[col_name] = labels

    result_df.reset_index(inplace=True)
    storage.write_csv(f"{filename}/labeled", result_df, index=False)
