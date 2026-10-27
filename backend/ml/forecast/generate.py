"""Future forecast generation — refit saved models and predict ahead.

Ported from the Forecaster web app's backend (generate_new_forecasts.py).
Loads the pickled models from a completed forecasting run, optionally
extends future covariates from an uploaded CSV, refits on the full series,
and writes predictions to ``<report_id>/forecasts``.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ml.args import ForecastArgs
from ml.column_types import read_csv_from_storage
from ml.forecast.engine import _log, get_predictions, read_in_data, setUpCovariates
from storage.base import KeyNotFound, Storage

LogCallback = Callable[[str], None]


def _load_model(storage: Storage, report_id: str, model_name: str) -> Any:
    from joblib import load

    model_buf = io.BytesIO(storage.get_bytes(f"{report_id}/{model_name}_model.joblib"))
    return load(model_buf)


def run_forecast_generation(
    args: ForecastArgs,
    storage: Storage,
    on_progress: Any = None,
    log_cb: LogCallback | None = None,
    cancel_token: Any = None,
) -> None:
    """Main forecast generation entry point (mode="forecast-generation" jobs)."""
    output_buf = io.StringIO()

    def log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    try:
        _run_generation(args, storage, output_buf, log, on_progress, cancel_token)
    finally:
        storage.put_text(f"{args.report_uuid}/forecast_generation_log", output_buf.getvalue())


def _run_generation(
    args: ForecastArgs,
    storage: Storage,
    output_f: Any,
    log: LogCallback,
    on_progress: Any,
    cancel_token: Any,
) -> None:
    from darts import TimeSeries
    from darts.dataprocessing.transformers import MissingValuesFiller

    report_id = args.report_uuid

    df = read_in_data(args, storage, log, output_f)
    if df is None:
        raise ValueError("Failed to read the dataset for forecast generation.")
    series, past_covariates, future_covariates, scalers, groups, future_covariate_scalers = (
        setUpCovariates(df, args, log, output_f, retrain=True)
    )
    preds: dict[str, Any] = {}

    future_covariates_key = f"{report_id}/future_covariates"
    try:
        future_covariates_df = read_csv_from_storage(storage, future_covariates_key)
        if future_covariates is None:
            _log(
                "Future covariates were uploaded but were not used for training, so they will "
                "be ignored.",
                output_f,
                log,
            )
        else:
            if (
                isinstance(future_covariates, list)
                and args.grouping[0] not in future_covariates_df.columns
            ):
                _log(
                    "No grouping column was included in uploaded future covariates, so they "
                    "will be ignored.",
                    output_f,
                    log,
                )
            else:
                if isinstance(future_covariates, list):
                    future_covariates_components = future_covariates[0].components.tolist()
                else:
                    future_covariates_components = future_covariates.components.tolist()
                components_in_columns = [
                    component
                    for component in future_covariates_components
                    if component in future_covariates_df.columns
                ]
                if len(components_in_columns) == 0:
                    _log(
                        "Future covariates were uploaded but do not match any covariate names "
                        "used for training, so they will be ignored.",
                        output_f,
                        log,
                    )
                else:
                    if isinstance(future_covariates, list):
                        last_future_covariate_date = future_covariates[0].time_index[-1]
                        series_freq = future_covariates[0].freq
                        series_freq_str = future_covariates[0].freq_str
                        grouping_col = args.grouping[0]
                        all_new_future_covariates = []
                        for i, grouping in enumerate(groups):
                            future_covariates_grouping_df = future_covariates_df[
                                future_covariates_df[grouping_col] == grouping
                            ]
                            if len(future_covariates_grouping_df) == 0:
                                all_new_future_covariates.append(future_covariates[i])
                            else:
                                new_future_covariate_dates = pd.date_range(
                                    start=last_future_covariate_date + series_freq,
                                    periods=len(future_covariates_grouping_df),
                                    freq=series_freq_str,
                                )
                                new_base_date_column_name = "Date"
                                new_date_column_name = new_base_date_column_name
                                new_date_column_name_counter = 1
                                while new_date_column_name in future_covariates_grouping_df.columns:
                                    new_date_column_name = f"{new_base_date_column_name}_{new_date_column_name_counter}"
                                    new_date_column_name_counter += 1
                                future_covariates_grouping_df.insert(
                                    0, new_date_column_name, new_future_covariate_dates
                                )
                                new_future_covariates = TimeSeries.from_dataframe(
                                    future_covariates_grouping_df,
                                    time_col=new_date_column_name,
                                    value_cols=components_in_columns,
                                    freq=new_future_covariate_dates.freqstr,
                                )
                                new_future_covariates = future_covariate_scalers[i].transform(
                                    new_future_covariates
                                )
                                new_future_covariates = future_covariates[i].concatenate(
                                    new_future_covariates, ignore_time_axis=False
                                )
                                all_new_future_covariates.append(new_future_covariates)
                        future_covariates = all_new_future_covariates
                    else:
                        last_future_covariate_date = future_covariates.time_index[-1]
                        series_freq = future_covariates.freq
                        series_freq_str = future_covariates.freq_str
                        new_future_covariate_dates = pd.date_range(
                            start=last_future_covariate_date + series_freq,
                            periods=len(future_covariates_df),
                            freq=series_freq_str,
                        )
                        new_base_date_column_name = "Date"
                        new_date_column_name = new_base_date_column_name
                        new_date_column_name_counter = 1
                        while new_date_column_name in future_covariates_df.columns:
                            new_date_column_name = (
                                f"{new_base_date_column_name}_{new_date_column_name_counter}"
                            )
                            new_date_column_name_counter += 1
                        future_covariates_df.insert(
                            0, new_date_column_name, new_future_covariate_dates
                        )
                        new_future_covariates = TimeSeries.from_dataframe(
                            future_covariates_df,
                            time_col=new_date_column_name,
                            value_cols=components_in_columns,
                        )
                        new_future_covariates = future_covariate_scalers[0].transform(
                            new_future_covariates
                        )
                        future_covariates = future_covariates.concatenate(
                            new_future_covariates, ignore_time_axis=False
                        )
    except KeyNotFound:
        _log("No uploaded future covariates found — continuing without them.", output_f, log)
    except Exception as e:
        _log(f"Failed to load uploaded future covariates: {e}", output_f, log)

    if future_covariates is not None:
        if isinstance(series, list):
            new_future_covariates = []
            message_logged = False
            for i, future_covariate_series in enumerate(future_covariates):
                last_series_date = series[i].time_index[-1]
                last_future_covariate_date = future_covariate_series.time_index[-1]
                series_freq = series[i].freq
                series_freq_str = series[i].freq_str

                series_forecast_dates = pd.date_range(
                    start=last_series_date + series_freq,
                    periods=args.forecast_horizon,
                    freq=series_freq_str,
                )
                future_covariates_needed_dates = pd.date_range(
                    start=last_future_covariate_date + series_freq,
                    end=series_forecast_dates[-1],
                    freq=series_freq_str,
                )
                num_future_covariates_dates_to_fill = len(future_covariates_needed_dates)
                if num_future_covariates_dates_to_fill > 0:
                    if not message_logged:
                        _log(
                            "Not enough future covariate values are provided, so imputation "
                            "will be used to fill in these dates.",
                            output_f,
                            log,
                        )
                        message_logged = True
                    nan_values = np.full(
                        (num_future_covariates_dates_to_fill, future_covariate_series.n_components),
                        np.nan,
                    )
                    future_covariate_filled_series = TimeSeries.from_times_and_values(
                        times=future_covariates_needed_dates,
                        values=nan_values,
                        columns=future_covariate_series.components,
                    )
                    future_covariate_series = future_covariate_series.concatenate(
                        future_covariate_filled_series, ignore_time_axis=False
                    )
                    filler = MissingValuesFiller(fill="auto")
                    future_covariate_series = filler.transform(future_covariate_series)
                    new_future_covariates.append(future_covariate_series)
                else:
                    new_future_covariates.append(future_covariate_series)
            future_covariates = new_future_covariates

            new_future_covariates = []
            for future_covariate_series in future_covariates:
                if np.any(np.isnan(future_covariate_series.values())):
                    filler = MissingValuesFiller(fill="auto")
                    future_covariate_series = filler.transform(future_covariate_series)
                new_future_covariates.append(future_covariate_series)
            future_covariates = new_future_covariates
        else:
            last_series_date = series.time_index[-1]
            last_future_covariate_date = future_covariates.time_index[-1]
            series_freq = series.freq
            series_freq_str = series.freq_str
            series_forecast_dates = pd.date_range(
                start=last_series_date + series_freq,
                periods=args.forecast_horizon,
                freq=series_freq_str,
            )
            future_covariates_needed_dates = pd.date_range(
                start=last_future_covariate_date + series_freq,
                end=series_forecast_dates[-1],
                freq=series_freq_str,
            )
            num_future_covariates_dates_to_fill = len(future_covariates_needed_dates)
            if num_future_covariates_dates_to_fill > 0:
                _log(
                    "Not enough future covariate values are provided, so imputation will be "
                    "used to fill in these dates.",
                    output_f,
                    log,
                )
                nan_values = np.full(
                    (num_future_covariates_dates_to_fill, future_covariates.n_components), np.nan
                )
                future_covariate_filled_series = TimeSeries.from_times_and_values(
                    times=future_covariates_needed_dates,
                    values=nan_values,
                    columns=future_covariates.components,
                )
                future_covariates = future_covariates.concatenate(
                    future_covariate_filled_series, ignore_time_axis=False
                )
                filler = MissingValuesFiller(fill="auto")
                future_covariates = filler.transform(future_covariates)

            if np.any(np.isnan(future_covariates.values())):
                filler = MissingValuesFiller(fill="auto")
                future_covariates = filler.transform(future_covariates)

    target_is_integer = []
    for target_column in args.target:
        target_is_integer.append(
            df[target_column].dropna().astype(float).apply(float.is_integer).all()
        )

    total_models = len(args.train_group)
    for i, model_name in enumerate(args.train_group):
        if cancel_token is not None and hasattr(cancel_token, "is_set") and cancel_token.is_set():
            _log("Forecast generation cancelled by user.", output_f, log)
            break
        _log(f"Forecasting model: {model_name}", output_f, log)
        try:
            model = _load_model(storage, report_id, model_name)
        except Exception:
            _log(f"Skipping {model_name} — no saved model found.", output_f, log)
            if on_progress:
                on_progress(i + 1, total_models, f"Forecasting ({i + 1}/{total_models})")
            continue

        if model_name == "tft" and future_covariates is None:
            _log("Skipping tft — requires future covariates.", output_f, log)
            if on_progress:
                on_progress(i + 1, total_models, f"Forecasting ({i + 1}/{total_models})")
            continue

        try:
            if past_covariates is not None and future_covariates is not None:
                if model_name == "arima":
                    _log(
                        "ARIMA model can only utilize future covariates, not past covariates.",
                        output_f,
                        log,
                    )
                    model.fit(series, future_covariates=future_covariates)
                elif model_name == "exponential-smoothing":
                    _log(
                        "Exponential Smoothing model cannot use covariates, so these will be ignored.",
                        output_f,
                        log,
                    )
                    model.fit(series)
                else:
                    model.fit(
                        series, past_covariates=past_covariates, future_covariates=future_covariates
                    )
            elif past_covariates is not None:
                if model_name == "arima" or model_name == "exponential-smoothing":
                    _log(
                        f"{model_name} model cannot use past covariates.",
                        output_f,
                        log,
                    )
                    model.fit(series)
                else:
                    model.fit(series, past_covariates=past_covariates)
            elif future_covariates is not None:
                if model_name == "exponential-smoothing":
                    _log(
                        "Exponential Smoothing model cannot use covariates, so these will be ignored.",
                        output_f,
                        log,
                    )
                    model.fit(series)
                else:
                    model.fit(series, future_covariates=future_covariates)
            else:
                model.fit(series)

            model_params = getattr(model, "model_params", {})
            if "output_chunk_length" in model_params:
                output_chunk_length = model_params["output_chunk_length"]
                if past_covariates is not None and output_chunk_length < args.forecast_horizon:
                    _log(
                        f"Cannot forecast past output chunk length ({output_chunk_length}) "
                        f"without past covariates. Setting forecast horizon to "
                        f"{output_chunk_length}.",
                        output_f,
                        log,
                    )
                    args.forecast_horizon = output_chunk_length

            predict_params: dict[str, Any] = {"n": args.forecast_horizon}
            if future_covariates is not None and model_name != "exponential-smoothing":
                predict_params["future_covariates"] = future_covariates
            if isinstance(series, list):
                predict_params["series"] = series
                if past_covariates is not None:
                    predict_params["past_covariates"] = past_covariates
            future_pred = model.predict(**predict_params)

            pred_dict = get_predictions(future_pred, scalers, groups, target_is_integer)
            preds[model_name] = pred_dict
            _log(f"{model_name} forecast complete.", output_f, log)
        except Exception as e:
            _log(f"Error in forecasting {model_name}: {e}", output_f, log)
        if on_progress:
            on_progress(i + 1, total_models, f"Forecasting ({i + 1}/{total_models})")

    if not preds:
        raise ValueError(
            "No forecasts were generated — none of the requested models had a saved model. "
            "Check the output log for details."
        )

    _log("==============FINAL FORECASTS=================", output_f, log)
    _log(str(preds), output_f, log)

    try:
        storage.put_text(f"{report_id}/forecasts", json.dumps(preds, indent=4))
    except Exception:
        _log("Failed to save forecasts to storage", output_f, log)
