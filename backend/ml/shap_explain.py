"""SHAP feature explainability — ported from CLASSify-2's models.py.

Generates SHAP beeswarm plots and per-row SHAP CSV files.
Preserved exactly; S3 calls replaced with storage calls.
"""

from __future__ import annotations

import io
import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from storage.base import Storage  # noqa: E402


def compute_shap(
    model: Any,
    model_name: str,
    scaler: Any,
    train_X: Any,
    test_X: Any,
    all_columns: list[str],
    binary: bool,
    args: Any,
    storage: Storage,
    filename: str,
) -> str | None:
    """Compute SHAP values, save beeswarm plot + per-row CSV.

    Returns the SHAP importance JSON string, or None if SHAP is not
    supported for this model type.
    """
    tree_models = ["randomforest", "xgboost"]
    tree_models_binary_only = ["histgradientboosting", "gradientboosting"]
    linear_models = ["logisticregression", "sgdclassifier"]

    try:
        if (
            model_name in tree_models
            or model_name in linear_models
            or (model_name in tree_models_binary_only and binary)
        ):
            ex = shap.Explainer(model, train_X)
        else:
            ex = shap.Explainer(model.predict_proba, train_X)

        if model_name in tree_models or (model_name in tree_models_binary_only and binary):
            shap_all = ex(test_X, check_additivity=False)
        else:
            shap_all = ex(test_X)

        # Per-row SHAP CSV
        if len(shap_all.values.shape) == 3 and binary:
            shap_per_row = shap_all.values[:, :, 1]
        else:
            shap_per_row = shap_all.values

        shap_all.feature_names = all_columns if not binary else all_columns

        if not binary and len(shap_all.values.shape) == 3:
            new_columns = [
                f"{feature_name}_class_{class_name}"
                for feature_name in shap_all.feature_names
                for class_name in model.classes_
            ]
            shap_per_row = shap_per_row.reshape(
                shap_per_row.shape[0], shap_per_row.shape[1] * shap_per_row.shape[2]
            )
            shap_per_row_df = pd.DataFrame(shap_per_row, columns=new_columns)
        else:
            shap_per_row_df = pd.DataFrame(shap_per_row, columns=shap_all.feature_names)

        shap_per_row_df = shap_per_row_df.round(4)
        shap_per_row_df.index.name = "Row Number"
        shap_per_row_df["dataset"] = "test"

        buf = io.StringIO()
        shap_per_row_df.to_csv(buf)
        storage.put_text(f"{filename}/shap_rows_{model_name}", buf.getvalue())

        # Beeswarm plot
        shap_values = shap_all
        if len(shap_all.values.shape) == 3 and binary:
            shap_values = shap_all[:, :, 1]

        fig, ax = plt.subplots(figsize=(10, 8))
        max_display = min(args.shap_diagram_features, len(all_columns))
        shap.plots.beeswarm(shap_values, max_display=max_display, show=False, ax=ax, plot_size=None)
        plt.title(model_name)
        buf2 = io.BytesIO()
        fig.savefig(buf2, format="png", bbox_inches="tight")
        buf2.seek(0)
        storage.put_bytes(f"{filename}/viz/SHAP_{model_name}", buf2.getvalue())
        plt.close(fig)

        # SHAP importance JSON
        if len(shap_all.values.shape) == 3 and binary:
            vals = shap_all.values[:, :, 1]
        else:
            vals = shap_all.values

        if len(vals.shape) == 2:
            mean_abs = np.mean(np.abs(vals), axis=0)
        else:
            mean_abs = np.mean(np.abs(vals.reshape(vals.shape[0], -1)), axis=0)

        columns = all_columns[: len(mean_abs)]
        columns_not_present = [item for item in all_columns if item not in columns]

        shap_dict = dict(zip(columns, [abs(float(item)) for item in mean_abs], strict=False))
        shap_dict = dict(sorted(shap_dict.items(), key=lambda item: item[1], reverse=True))
        for column in columns_not_present:
            shap_dict[column] = -1

        return json.dumps(shap_dict)

    except Exception as e:
        print(f"SHAP error for {model_name}: {e}")
        shap_dict = {key: -1 for key in all_columns}
        return json.dumps(shap_dict)


def get_shap_row_graph(
    storage: Storage,
    filename: str,
    model_name: str,
    row_num: int,
    train_test: str,
    class_column: str,
) -> bytes | None:
    """Generate a per-row SHAP impact bar chart.

    Ported from CLASSify-2's download-shap-row-graph endpoint.
    Returns PNG bytes, or None on failure.
    """
    try:
        df: pd.DataFrame = storage.read_csv(
            f"{filename}/shap_rows_{model_name}", index_col="Row Number"
        )
        df = df[df["dataset"] == train_test]
        row: pd.Series = df.loc[int(row_num)]

        original_dataset: pd.DataFrame = storage.read_csv(f"{filename}/file", index_col="index")
        original_dataset_row: pd.Series = original_dataset.loc[int(row_num)]

        predicted_columns = df.columns.str.startswith("predicted_")
        predicted_column = df.columns[predicted_columns][-1] if predicted_columns.any() else None

        multiclass = len(df[class_column].unique()) > 2 if class_column in df.columns else False

        if predicted_column:
            class_label = row.get(predicted_column, row.get(class_column))
            row = row.drop(labels=[class_column, predicted_column, "dataset"], errors="ignore")
        else:
            row = row.drop(labels=["dataset"], errors="ignore")

        row_series: pd.Series = row if isinstance(row, pd.Series) else pd.Series(row)
        plot_df = row_series.to_frame(name="shap_value")
        plot_df = pd.merge(
            plot_df,
            original_dataset_row.to_frame(name="feature_value"),
            left_index=True,
            right_index=True,
            how="left",
        )

        plot_df["abs_shap"] = plot_df["shap_value"].abs()
        plot_df = plot_df.sort_values(by="abs_shap", ascending=False).reset_index()

        max_display = 15
        if len(plot_df) > max_display:
            plot_df_extra = plot_df.iloc[max_display:]
            numeric_means = plot_df_extra.select_dtypes(include=np.number).mean()
            numeric_means["index"] = f"Avg. of {len(plot_df_extra)} other columns"
            plot_df = plot_df[~plot_df["index"].isin(plot_df_extra["index"])]
            plot_df.loc[max(plot_df.index) + 1] = numeric_means

        plot_df = plot_df.iloc[::-1].reset_index(drop=True)

        positive_color = "#28A745"
        negative_color = "#DC3545"
        y_pos = np.arange(len(plot_df))

        fig, ax = plt.subplots(figsize=(8, len(plot_df) * 0.5))
        ax.barh(
            y_pos,
            plot_df["shap_value"],
            color=[positive_color if x > 0 else negative_color for x in plot_df["shap_value"]],
        )

        max_abs_shap = plot_df["shap_value"].abs().max()
        text_offset = max_abs_shap * 0.02
        text_placement_threshold = max_abs_shap * 0.2

        for i, (_, r) in plot_df.iterrows():
            shap_val = r["shap_value"]
            display_text = f"{shap_val:+.3f}".replace("+0.000", "<0.001").replace(
                "-0.000", ">-0.001"
            )
            if abs(shap_val) > text_placement_threshold:
                x_pos = shap_val - (np.sign(shap_val) * text_offset)
                ha = "right" if shap_val > 0 else "left"
                text_color = "white"
            else:
                x_pos = shap_val + (np.sign(shap_val) * text_offset)
                if shap_val == 0:
                    x_pos = text_offset
                ha = "left" if shap_val >= 0 else "right"
                text_color = "black"
            ax.text(
                x_pos,
                y_pos[i],
                display_text,
                va="center",
                ha=ha,
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

        y_labels = [
            f"{r['index']}"
            if pd.isna(r["feature_value"]) or (str(r["index"]).startswith("Avg. of") and i == 0)
            else f"{r['index']} ({r['feature_value']})"
            for i, (_, r) in enumerate(plot_df.iterrows())
        ]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=11)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.8)
        ax.tick_params(axis="y", length=0)

        min_shap = plot_df["shap_value"].min()
        max_shap = plot_df["shap_value"].max()
        data_range = max_shap - min_shap
        padding = data_range * 0.15
        ax.set_xlim(min_shap - padding, max_shap + padding)

        ax.set_xlabel(f"Impact on {model_name} output (SHAP value)", fontsize=12)
        title = f"Feature Impact on Prediction - Row {row_num}"
        if multiclass:
            title += f" (for Class {class_label})"
        ax.set_title(title)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()

    except Exception as e:
        print(f"SHAP row graph error: {e}")
        return None
