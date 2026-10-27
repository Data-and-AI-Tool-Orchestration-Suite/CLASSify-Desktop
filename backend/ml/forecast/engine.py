"""Forecasting engine — Darts-based time-series trainer.

Ported from the Forecaster web app's backend (backend/python/main.py).
S3 → LocalStorage, HTTP status updates → progress callbacks, ClearML-style
config → ForecastArgs dataclass. The modeling math and defaults are
preserved exactly.
"""

from __future__ import annotations

import io
import json
import math
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from ml.args import ForecastArgs
from ml.column_types import read_csv_from_storage
from storage.base import Storage

ProgressCallback = Callable[[int, int, str], None]
LogCallback = Callable[[str], None]

freq_map = {
    "H": 24,
    "D": 7,
    "W-MON": 52,
    "W-TUE": 52,
    "MS": 12,
    "M": 12,
    "QS": 4,
    "Q": 4,
    "AS": 1,
    "A": 1,
}
retrain_row_name = "retrain_row"


def _log(msg: str, output_f: Any, log_cb: LogCallback | None = None) -> None:
    if hasattr(output_f, "write"):
        output_f.write(msg + "\n")
        output_f.flush()
    if log_cb:
        log_cb(msg)


def _parse_date_column(
    df: pd.DataFrame, date_column: str, logger_msg: Callable[[str], None]
) -> bool:
    """Parse the time component column in place. Returns False on failure."""
    if pd.api.types.is_numeric_dtype(df[date_column].dtype):
        logger_msg(
            "Since time component column consists of numbers, assuming that values represent "
            "seconds since epoch. If this is not the case, please reformat the time component "
            "column and re-run."
        )
        if pd.api.types.is_integer_dtype(df[date_column].dtype):
            df[date_column] = pd.to_datetime(df[date_column], unit="s", origin="1970-01-01")
        else:
            df[date_column] = pd.to_datetime(
                np.round(df[date_column] * 1000000).astype(int), unit="us", origin="1970-01-01"
            )
    elif pd.api.types.is_datetime64_any_dtype(
        df[date_column].dtype
    ) or pd.api.types.is_object_dtype(df[date_column].dtype):
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    else:
        logger_msg(
            "Unrecognized format for time component column. Please ensure your column follows "
            "a date or numerical structure."
        )
        return False
    return True


def read_in_data(
    args: ForecastArgs, storage: Storage, log: LogCallback, output_f: Any
) -> pd.DataFrame | None:
    global retrain_row_name
    date_column = args.time_component

    def logger_msg(msg: str) -> None:
        _log(msg, output_f, log)

    try:
        df = read_csv_from_storage(storage, f"{args.report_uuid}/original_file")
        if not _parse_date_column(df, date_column, logger_msg):
            return None
    except Exception:
        _log("Failed to retrieve dataset from storage", output_f, log)
        return None

    if args.retrain:
        try:
            df_new = read_csv_from_storage(storage, f"{args.report_uuid}/retrain_file")
            if not _parse_date_column(df_new, date_column, logger_msg):
                return None
        except Exception:
            _log("Failed to retrieve retraining dataset from storage", output_f, log)
            return None
        if retrain_row_name not in df.columns:
            df[retrain_row_name] = 0
            df_new[retrain_row_name] = 1
        else:
            while True:
                retrain_row_name = retrain_row_name + "0"
                if retrain_row_name not in df.columns:
                    df[retrain_row_name] = 0
                    df_new[retrain_row_name] = 1
                    break
        df = pd.concat([df, df_new])

    if df[date_column].isna().any():
        _log("Missing dates found! Dropping these rows", output_f, log)
        df = df[~df[date_column].isna()]
        if len(df) == 0:
            _log("No valid dates found. Ensure the correct date column is selected.", output_f, log)
            return None

    if len(args.grouping) == 0:
        df = df.sort_values(by=[date_column])
        if df[date_column].duplicated().any():
            _log("Duplicate dates found! Dropping all duplicate rows", output_f, log)
            if args.retrain:
                df = df.drop_duplicates(subset=[date_column], keep="last")
            else:
                df = df.drop_duplicates(subset=[date_column], keep="first")
    else:
        df = df.sort_values(by=[date_column, args.grouping[0]])
    if args.retrain:
        csv_buffer = io.StringIO()
        df.drop(columns=[retrain_row_name]).to_csv(csv_buffer, index=False)
        storage.put_text(f"{args.report_uuid}/original_file", csv_buffer.getvalue())
    return df


def _concatenate_extension(
    series: Any, extension_range: Any, columns: Any, prepend: bool = False
) -> Any:
    from darts import TimeSeries

    n_timesteps_to_add = len(extension_range)
    nan_values = np.full((n_timesteps_to_add, series.n_components), np.nan)
    extension_series = TimeSeries.from_times_and_values(
        times=extension_range, values=nan_values, columns=columns
    )
    if prepend:
        return extension_series.concatenate(series, ignore_time_axis=False)
    return series.concatenate(extension_series, ignore_time_axis=False)


def setUpCovariates(  # noqa: N802
    df: pd.DataFrame, args: ForecastArgs, log: LogCallback, output_f: Any, retrain: bool = False
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Build the Darts series, covariates, scalers and splits.

    Ported verbatim from the Forecaster backend — grouping alignment,
    covariate one-hot encoding, missing-value filling and scaling.
    """
    from darts import TimeSeries
    from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
    from darts.utils.model_selection import train_test_split

    date_column = args.time_component
    if len(args.grouping) > 0:
        grouping_column = args.grouping[0]
        groups = df[grouping_column].unique()
        earliest_start_time = None
        latest_end_time = None
        for group in groups:
            dataset_subset = df[df[grouping_column] == group]
            start_time = dataset_subset[date_column].min()
            end_time = dataset_subset[date_column].max()
            if earliest_start_time is None or start_time < earliest_start_time:
                earliest_start_time = start_time
            if latest_end_time is None or end_time > latest_end_time:
                latest_end_time = end_time
    else:
        groups = [""]

    all_train: list[Any] = []
    all_val: list[Any] = []
    all_series: list[Any] = []
    all_past_covariates: list[Any] = []
    all_future_covariates: list[Any] = []
    all_scalers: list[Any] = []
    future_covariate_scalers: list[Any] = []
    if len(args.static_covariates) > 0:
        missing_static_covariates: dict[str, Any] = {}
        for group in groups:
            dataset = df if group == "" else df[df[grouping_column] == group]
            for static_covariate in args.static_covariates:
                non_nan_series = dataset[static_covariate].dropna()
                if non_nan_series.empty and static_covariate not in missing_static_covariates:
                    missing_static_covariates[static_covariate] = df[static_covariate].dtype

    logged_covariate_messages: list[str] = []
    for group in groups:
        dataset = df if group == "" else df[df[grouping_column] == group].reset_index(drop=True)
        for target_column in args.target:
            if dataset[target_column].isna().all():
                _log(
                    f"The grouping {group} has no valid values for prediction on column "
                    f"{target_column}. Skipping group.",
                    output_f,
                    log,
                )
                continue
        if args.generate_real_predictions:
            args.output_chunk_length = args.real_prediction_length
            missing_values_at_end = dataset[args.target[0]].isna()
            last_non_nan_index = missing_values_at_end[::-1].idxmin()
            if not missing_values_at_end.any():
                num_to_extend = args.real_prediction_length
            else:
                num_missing_values_at_end = len(dataset[args.target]) - (last_non_nan_index + 1)
                if num_missing_values_at_end < args.real_prediction_length:
                    num_to_extend = args.real_prediction_length - num_missing_values_at_end
                elif num_missing_values_at_end > args.real_prediction_length:
                    dataset = dataset[: args.real_prediction_length - num_missing_values_at_end]
                    num_to_extend = 0
                else:
                    num_to_extend = 0

        series = TimeSeries.from_dataframe(
            dataset, date_column, args.target, fill_missing_dates=True
        )

        start_time = series.start_time()
        end_time = series.end_time()
        if len(args.grouping) > 0:
            if start_time > earliest_start_time:
                extension_end_time = start_time - series.freq
                extension_range = pd.date_range(
                    start=earliest_start_time, end=extension_end_time, freq=series.freq
                )
                _log(
                    f"Extending beginning of target series for grouping {group} by "
                    f"{len(extension_range)} timesteps to align groupings together.",
                    output_f,
                    log,
                )
                series = _concatenate_extension(
                    series, extension_range, series.components, prepend=True
                )
            if end_time < latest_end_time:
                extension_start_time = end_time + series.freq
                extension_range = pd.date_range(
                    start=extension_start_time, end=latest_end_time, freq=series.freq
                )
                _log(
                    f"Extending end of target series for grouping {group} by "
                    f"{len(extension_range)} timesteps to align groupings together.",
                    output_f,
                    log,
                )
                series = _concatenate_extension(series, extension_range, series.components)

        filler = MissingValuesFiller()
        series = filler.transform(series)

        if args.generate_real_predictions:
            args.output_chunk_length = args.real_prediction_length

            series_freq = series.freq
            original_end_time = series.end_time()
            new_start_time = original_end_time + to_offset(series_freq)
            new_time_index = pd.date_range(
                start=new_start_time, periods=num_to_extend, freq=series_freq
            )
            nan_values = np.full((num_to_extend, series.n_components), np.nan)
            val = TimeSeries.from_times_and_values(
                times=new_time_index, values=nan_values, columns=series.components
            )
            series = series.concatenate(val, ignore_time_axis=False)

            existing_columns = dataset.columns.tolist()
            new_rows_data: dict[str, Any] = {}
            for col in existing_columns:
                if col == date_column:
                    new_rows_data[col] = new_time_index
                else:
                    new_rows_data[col] = [np.nan] * len(new_time_index)
            df_to_add = pd.DataFrame(new_rows_data)
            dataset = pd.concat([dataset, df_to_add])
        future_covariates: Any = None
        past_covariates: Any = None

        if len(args.future_covariates) > 0:
            for _i, future_covariate in enumerate(args.future_covariates):
                if dataset[future_covariate].dtype == "object":
                    _log(
                        f"One hot encoding {future_covariate} column because it is categorical.",
                        output_f,
                        log,
                    )
                    original_columns = set(dataset.columns)
                    one_hot_encoded = pd.get_dummies(
                        dataset, columns=[future_covariate], prefix=future_covariate
                    )
                    new_columns = set(one_hot_encoded.columns)
                    future_covariate_columns = list(new_columns - original_columns)
                    for future_covariate_column in future_covariate_columns:
                        covariate = TimeSeries.from_dataframe(
                            one_hot_encoded,
                            date_column,
                            future_covariate_column,
                            fill_missing_dates=True,
                        )
                        if (
                            args.generate_real_predictions
                            and np.all(np.isnan(covariate.values()[-1]))
                            and not args.fill_future_covariates
                        ):
                            if future_covariate not in logged_covariate_messages:
                                _log(
                                    f"Treating {future_covariate} as past covariate for predictions "
                                    "because the fill-future-covariates option was set to false",
                                    output_f,
                                    log,
                                )
                                logged_covariate_messages.append(future_covariate)
                                args.past_covariates.append(future_covariate_column)
                            continue
                        filler = MissingValuesFiller()
                        covariate = filler.transform(covariate)
                        if future_covariates is None:
                            future_covariates = covariate
                        else:
                            future_covariates = future_covariates.stack(covariate)
                else:
                    covariate = TimeSeries.from_dataframe(
                        dataset, date_column, future_covariate, fill_missing_dates=True
                    )
                    if (
                        args.generate_real_predictions
                        and np.all(np.isnan(covariate.values()[-1]))
                        and not args.fill_future_covariates
                    ):
                        if future_covariate not in logged_covariate_messages:
                            _log(
                                f"Treating {future_covariate} as past covariate for predictions because "
                                "the fill-future-covariates option was set to false",
                                output_f,
                                log,
                            )
                            logged_covariate_messages.append(future_covariate)
                            args.past_covariates.append(future_covariate)
                        continue
                    filler = MissingValuesFiller()
                    covariate = filler.transform(covariate)
                    if future_covariates is None:
                        future_covariates = covariate
                    else:
                        future_covariates = future_covariates.stack(covariate)
        if len(args.past_covariates) > 0:
            for _i, past_covariate in enumerate(args.past_covariates):
                if dataset[past_covariate].dtype == "object":
                    _log(
                        f"One hot encoding {past_covariate} column because it is categorical.",
                        output_f,
                        log,
                    )
                    original_columns = set(dataset.columns)
                    one_hot_encoded = pd.get_dummies(
                        dataset, columns=[past_covariate], prefix=past_covariate
                    )
                    new_columns = set(one_hot_encoded.columns)
                    past_covariate_columns = list(new_columns - original_columns)
                    for past_covariate_column in past_covariate_columns:
                        covariate = TimeSeries.from_dataframe(
                            one_hot_encoded,
                            date_column,
                            past_covariate_column,
                            fill_missing_dates=True,
                        )
                        filler = MissingValuesFiller()
                        covariate = filler.transform(covariate)
                        if past_covariates is None:
                            past_covariates = covariate
                        else:
                            past_covariates = past_covariates.stack(covariate)
                else:
                    covariate = TimeSeries.from_dataframe(
                        dataset, date_column, past_covariate, fill_missing_dates=True
                    )
                    filler = MissingValuesFiller()
                    covariate = filler.transform(covariate)
                    if past_covariates is None:
                        past_covariates = covariate
                    else:
                        past_covariates = past_covariates.stack(covariate)

        if future_covariates is not None:
            start_time = future_covariates.start_time()
            end_time = future_covariates.end_time()
            if len(args.grouping) > 0:
                if start_time > earliest_start_time:
                    extension_end_time = start_time - future_covariates.freq
                    extension_range = pd.date_range(
                        start=earliest_start_time,
                        end=extension_end_time,
                        freq=future_covariates.freq,
                    )
                    if group == "":
                        _log(
                            f"Extending beginning of future covariate series by {len(extension_range)} "
                            "timesteps to align with target series.",
                            output_f,
                            log,
                        )
                    else:
                        _log(
                            f"Extending beginning of future covariate series for grouping {group} by "
                            f"{len(extension_range)} timesteps to align groupings together.",
                            output_f,
                            log,
                        )
                    future_covariates = _concatenate_extension(
                        future_covariates, extension_range, future_covariates.components
                    )
                    filler = MissingValuesFiller()
                    future_covariates = filler.transform(future_covariates)
                if end_time < latest_end_time:
                    extension_start_time = end_time + future_covariates.freq
                    extension_range = pd.date_range(
                        start=extension_start_time, end=latest_end_time, freq=future_covariates.freq
                    )
                    if group == "":
                        _log(
                            f"Extending end of future covariate series by {len(extension_range)} "
                            "timesteps to align with target series.",
                            output_f,
                            log,
                        )
                    else:
                        _log(
                            f"Extending end of future covariate series for grouping {group} by "
                            f"{len(extension_range)} timesteps to align groupings together.",
                            output_f,
                            log,
                        )
                    future_covariates = _concatenate_extension(
                        future_covariates, extension_range, future_covariates.components
                    )
                    filler = MissingValuesFiller()
                    future_covariates = filler.transform(future_covariates)

            scaler = Scaler()
            future_covariates = scaler.fit_transform(future_covariates)
            future_covariate_scalers.append(scaler)
        if past_covariates is not None:
            start_time = past_covariates.start_time()
            end_time = past_covariates.end_time()
            if len(args.grouping) > 0:
                if start_time > earliest_start_time:
                    extension_end_time = start_time - past_covariates.freq
                    extension_range = pd.date_range(
                        start=earliest_start_time, end=extension_end_time, freq=past_covariates.freq
                    )
                    if group == "":
                        _log(
                            f"Extending beginning of past covariate series by {len(extension_range)} "
                            "timesteps to align with target series.",
                            output_f,
                            log,
                        )
                    else:
                        _log(
                            f"Extending beginning of past covariate series for grouping {group} by "
                            f"{len(extension_range)} timesteps to align groupings together.",
                            output_f,
                            log,
                        )
                    past_covariates = _concatenate_extension(
                        past_covariates, extension_range, past_covariates.components
                    )
                    filler = MissingValuesFiller()
                    past_covariates = filler.transform(past_covariates)
                if end_time < latest_end_time:
                    extension_start_time = end_time + past_covariates.freq
                    extension_range = pd.date_range(
                        start=extension_start_time, end=latest_end_time, freq=past_covariates.freq
                    )
                    if group == "":
                        _log(
                            f"Extending end of past covariate series by {len(extension_range)} "
                            "timesteps to align with target series.",
                            output_f,
                            log,
                        )
                    else:
                        _log(
                            f"Extending end of past covariate series for grouping {group} by "
                            f"{len(extension_range)} timesteps to align groupings together.",
                            output_f,
                            log,
                        )
                    past_covariates = _concatenate_extension(
                        past_covariates, extension_range, past_covariates.components
                    )
                    filler = MissingValuesFiller()
                    past_covariates = filler.transform(past_covariates)

            scaler = Scaler()
            past_covariates = scaler.fit_transform(past_covariates)

        if len(args.static_covariates) > 0:
            static_dict = dataset[args.static_covariates].iloc[0].to_dict()
            static_dict = {key: [value] for key, value in static_dict.items()}
            static_df = pd.DataFrame(data=static_dict)
            for col in static_df.columns:
                if col in missing_static_covariates:
                    if pd.isna(static_df[col].iloc[0]):
                        static_df[col + "_missing"] = [1]
                        if pd.api.types.is_numeric_dtype(missing_static_covariates[col]):
                            static_df[col] = [0]
                        else:
                            static_df[col] = ["Missing"]
                    else:
                        static_df[col + "_missing"] = [0]
            series = series.with_static_covariates(static_df)
        if args.test_size > len(series):
            _log(
                "Test size is larger than timeseries length, setting to 0.2 default.", output_f, log
            )
            args.test_size = 0.2
        elif args.test_size <= 0:
            _log("Test size cannot be <=0, setting to 0.2 default.", output_f, log)
            args.test_size = 0.2
        if not retrain:
            if args.generate_real_predictions:
                train, val = train_test_split(series, args.real_prediction_length)
            else:
                test_size = args.test_size
                if args.retrain and test_size < 1:
                    prop_new_data = (df[retrain_row_name] == 1).mean()
                    test_size = test_size * prop_new_data
                train, val = train_test_split(series, test_size)
            scaler = Scaler()
            scaler.fit(train)
            train = scaler.transform(train)
            val = scaler.transform(val)
            all_train.append(train)
            all_val.append(val)
        else:
            scaler = Scaler()
            series = scaler.fit_transform(series)
            all_series.append(series)
        if past_covariates is not None:
            all_past_covariates.append(past_covariates)
        if future_covariates is not None:
            all_future_covariates.append(future_covariates)
        all_scalers.append(scaler)

    if len(all_train) == 1 or len(all_series) == 1:
        if len(all_past_covariates) == 0:
            all_past_covariates = [None]
        if len(all_future_covariates) == 0:
            all_future_covariates = [None]
        if retrain:
            return (
                all_series[0],
                all_past_covariates[0],
                all_future_covariates[0],
                all_scalers[0],
                groups,
                future_covariate_scalers,
            )
        return (
            all_train[0],
            all_val[0],
            all_past_covariates[0],
            all_future_covariates[0],
            all_scalers[0],
            groups,
        )
    past_out: Any = all_past_covariates if all_past_covariates else None
    future_out: Any = all_future_covariates if all_future_covariates else None
    if retrain:
        return all_series, past_out, future_out, all_scalers, groups, future_covariate_scalers
    return all_train, all_val, past_out, future_out, all_scalers, groups


def _historical_forecast_pred(
    model: Any,
    full_series: Any,
    series_end_time: Any,
    args: ForecastArgs,
    log: LogCallback,
    output_f: Any,
    model_label: str,
    past_covariates: Any = None,
    future_covariates: Any = None,
    components: Any = None,
) -> Any:
    """Run historical_forecasts and merge overlapping windows (shared helper)."""
    from darts import TimeSeries

    historical_params: dict[str, Any] = {
        "series": full_series,
        "forecast_horizon": args.expanding_window_forecast_horizon,
        "stride": args.expanding_window_stride,
        "last_points_only": False,
        "retrain": True,
        "verbose": False,
        "overlap_end": True,
    }
    if past_covariates is not None:
        historical_params["past_covariates"] = past_covariates
    if future_covariates is not None:
        historical_params["future_covariates"] = future_covariates
    pred = model.historical_forecasts(**historical_params)
    all_preds_df = pd.concat([ts.to_dataframe() for ts in pred], axis=0)
    final_preds_df = all_preds_df.groupby(level=0).last()
    pred = TimeSeries.from_dataframe(
        final_preds_df,
        time_col=None,
        value_cols=(components if components is not None else full_series.components).tolist(),
        fill_missing_dates=True,
    )
    try:
        pred, _ = pred.split_after(series_end_time)
    except Exception:
        _log(
            f"Historical predictions end early at {pred.end_time()} for {model_label} model due "
            "to mismatch of forecast horizon and series length.",
            output_f,
            log,
        )
    return pred


def train_arima_model(
    train: Any,
    val: Any,
    future_covariates: Any,
    args: ForecastArgs,
    storage: Storage,
    log: LogCallback,
    output_f: Any,
) -> Any:
    from darts.models import ARIMA, VARIMA

    min_start_length = 31
    model_params: dict[str, Any] = {"p": args.arima_p, "d": args.arima_d, "q": args.arima_q}
    if isinstance(train, list):
        _log(
            "Trained ARIMA models will not be saved to storage because separate models must be "
            "created for each grouping.",
            output_f,
            log,
        )
        preds = []
        for i, train_series in enumerate(train):
            fit_params: dict[str, Any] = {"series": train_series}
            predict_params: dict[str, Any] = {"n": len(val[i])}
            if future_covariates is not None:
                fit_params["future_covariates"] = future_covariates[i]
                predict_params["future_covariates"] = future_covariates[i]
            if train_series.n_components > 1:
                _log(
                    "Using VARIMA model instead of ARIMA to perform multivariate forecasting",
                    output_f,
                    log,
                )
                model = VARIMA(**model_params)
            else:
                model = ARIMA(**model_params)
            if args.expanding_window:
                full_series = train_series.concatenate(val[i])
                series_end_time = full_series.end_time()
                safe_start_time = full_series.time_index[min_start_length]
                historical_params: dict[str, Any] = {
                    "series": full_series,
                    "forecast_horizon": args.expanding_window_forecast_horizon,
                    "stride": args.expanding_window_stride,
                    "last_points_only": False,
                    "retrain": True,
                    "verbose": False,
                    "start": safe_start_time,
                    "overlap_end": True,
                }
                if future_covariates is not None:
                    required_range = pd.date_range(
                        start=full_series.end_time() + full_series.freq,
                        periods=args.expanding_window_forecast_horizon,
                        freq=full_series.freq,
                    )
                    required_end_date = required_range[-1]
                    extension_range = pd.date_range(
                        start=future_covariates[i].end_time() + future_covariates[i].freq,
                        end=required_end_date,
                        freq=future_covariates[i].freq,
                    )
                    if len(extension_range) > 0:
                        from darts import TimeSeries
                        from darts.dataprocessing.transformers import MissingValuesFiller

                        nan_values = np.full(
                            (len(extension_range), future_covariates[i].n_components), np.nan
                        )
                        extension_series = TimeSeries.from_times_and_values(
                            times=extension_range,
                            values=nan_values,
                            columns=future_covariates[i].components,
                        )
                        extended_future_covariates = future_covariates[i].concatenate(
                            extension_series, ignore_time_axis=False
                        )
                        filler = MissingValuesFiller()
                        extended_future_covariates = filler.transform(extended_future_covariates)
                        historical_params["future_covariates"] = extended_future_covariates
                    else:
                        historical_params["future_covariates"] = future_covariates[i]
                pred = model.historical_forecasts(**historical_params)
                all_preds_df = pd.concat([ts.to_dataframe() for ts in pred], axis=0)
                final_preds_df = all_preds_df.groupby(level=0).last()
                pred = TimeSeries.from_dataframe(
                    final_preds_df,
                    time_col=None,
                    value_cols=train_series.components.tolist(),
                    fill_missing_dates=True,
                )
                try:
                    pred, _ = pred.split_after(series_end_time)
                except Exception:
                    _log(
                        f"Historical predictions end early at {pred.end_time()} for ARIMA model due "
                        "to mismatch of forecast horizon and series length.",
                        output_f,
                        log,
                    )
            else:
                model.fit(**fit_params)
                pred = model.predict(**predict_params)

            preds.append(pred)
        return preds
    fit_params = {"series": train}
    predict_params = {"n": len(val)}
    if future_covariates is not None:
        fit_params["future_covariates"] = future_covariates
        predict_params["future_covariates"] = future_covariates
    if train.n_components > 1:
        _log(
            "Using VARIMA model instead of ARIMA to perform multivariate forecasting", output_f, log
        )
        model = VARIMA(**model_params)
    else:
        model = ARIMA(**model_params)
    if args.expanding_window:
        full_series = train.concatenate(val)
        series_end_time = full_series.end_time()
        safe_start_time = full_series.time_index[min_start_length]
        historical_params = {
            "series": full_series,
            "forecast_horizon": args.expanding_window_forecast_horizon,
            "stride": args.expanding_window_stride,
            "last_points_only": False,
            "retrain": True,
            "verbose": False,
            "start": safe_start_time,
            "overlap_end": True,
        }
        if future_covariates is not None:
            required_range = pd.date_range(
                start=full_series.end_time() + full_series.freq,
                periods=args.expanding_window_forecast_horizon,
                freq=full_series.freq,
            )
            required_end_date = required_range[-1]
            extension_range = pd.date_range(
                start=future_covariates.end_time() + future_covariates.freq,
                end=required_end_date,
                freq=future_covariates.freq,
            )
            if len(extension_range) > 0:
                from darts import TimeSeries
                from darts.dataprocessing.transformers import MissingValuesFiller

                nan_values = np.full((len(extension_range), future_covariates.n_components), np.nan)
                extension_series = TimeSeries.from_times_and_values(
                    times=extension_range,
                    values=nan_values,
                    columns=future_covariates.components,
                )
                extended_future_covariates = future_covariates.concatenate(
                    extension_series, ignore_time_axis=False
                )
                filler = MissingValuesFiller()
                extended_future_covariates = filler.transform(extended_future_covariates)
                historical_params["future_covariates"] = extended_future_covariates
            else:
                historical_params["future_covariates"] = future_covariates
        pred = model.historical_forecasts(**historical_params)
        all_preds_df = pd.concat([ts.to_dataframe() for ts in pred], axis=0)
        final_preds_df = all_preds_df.groupby(level=0).last()
        pred = TimeSeries.from_dataframe(
            final_preds_df,
            time_col=None,
            value_cols=train.components.tolist(),
            fill_missing_dates=True,
        )
        try:
            pred, _ = pred.split_after(series_end_time)
        except Exception:
            _log(
                f"Historical predictions end early at {pred.end_time()} for ARIMA model due to "
                "mismatch of forecast horizon and series length.",
                output_f,
                log,
            )
    else:
        model.fit(**fit_params)
        pred = model.predict(**predict_params)

    _save_forecast_model(storage, args.report_uuid, "arima", model, log, output_f)

    return pred


def _save_forecast_model(
    storage: Storage,
    report_id: str,
    model_name: str,
    model: Any,
    log: LogCallback,
    output_f: Any,
) -> None:
    from joblib import dump

    model_buf = io.BytesIO()
    dump(model, model_buf)
    try:
        model_buf.seek(0)
        storage.put_bytes(f"{report_id}/{model_name}_model.joblib", model_buf.getvalue())
    except Exception:
        _log(f"Failed to save {model_name} model to storage", output_f, log)


def train_exponential_smoothing_model(
    train: Any, val: Any, args: ForecastArgs, storage: Storage, log: LogCallback, output_f: Any
) -> Any:
    from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing

    if isinstance(train, list):
        _log(
            "Trained exponential smoothing models will not be saved to storage because separate "
            "models must be created for each grouping.",
            output_f,
            log,
        )
        preds = []
        for i, train_series in enumerate(train):
            predict_params: dict[str, Any] = {"n": len(val[i])}
            if train_series.n_components > 1:
                all_predictions = []
                _log(
                    "Building separate Exponential Smoothing models for each component of "
                    "multivariate time series",
                    output_f,
                    log,
                )
                for component in train_series.components:
                    model = ExponentialSmoothing()
                    if args.expanding_window:
                        full_series = train_series[component].concatenate(val[i][component])
                        series_end_time = full_series.end_time()
                        freq = full_series.freq_str
                        seasonal_period = freq_map.get(freq, 1)
                        min_training_length = 2 * seasonal_period + 1
                        safe_start_time = full_series.time_index[min_training_length]
                        historical_params = {
                            "series": full_series,
                            "forecast_horizon": args.expanding_window_forecast_horizon,
                            "stride": args.expanding_window_stride,
                            "last_points_only": False,
                            "retrain": True,
                            "verbose": False,
                            "start": safe_start_time,
                            "overlap_end": True,
                        }
                        pred = model.historical_forecasts(**historical_params)
                        all_preds_df = pd.concat([ts.to_dataframe() for ts in pred], axis=0)
                        final_preds_df = all_preds_df.groupby(level=0).last()
                        pred = TimeSeries_from_dataframe(final_preds_df, value_cols=[component])
                        try:
                            pred, _ = pred.split_after(series_end_time)
                        except Exception:
                            _log(
                                f"Historical predictions end early at {pred.end_time()} for "
                                "Exponential Smoothing model due to mismatch of forecast horizon "
                                "and series length.",
                                output_f,
                                log,
                            )
                    else:
                        model.fit(train_series[component])
                        pred = model.predict(**predict_params)
                    all_predictions.append(pred)
                pred = all_predictions[0]
                for j in range(1, len(all_predictions)):
                    pred = pred.stack(all_predictions[j])
            else:
                model = ExponentialSmoothing()
                if args.expanding_window:
                    full_series = train_series.concatenate(val[i])
                    series_end_time = full_series.end_time()
                    freq = full_series.freq_str
                    seasonal_period = freq_map.get(freq, 1)
                    min_training_length = 2 * seasonal_period + 1
                    safe_start_time = full_series.time_index[min_training_length]
                    historical_params = {
                        "series": full_series,
                        "forecast_horizon": args.expanding_window_forecast_horizon,
                        "stride": args.expanding_window_stride,
                        "last_points_only": False,
                        "retrain": True,
                        "verbose": False,
                        "start": safe_start_time,
                        "overlap_end": True,
                    }
                    pred = model.historical_forecasts(**historical_params)
                    all_preds_df = pd.concat([ts.to_dataframe() for ts in pred], axis=0)
                    final_preds_df = all_preds_df.groupby(level=0).last()
                    pred = TimeSeries_from_dataframe(
                        final_preds_df, value_cols=train_series.components.tolist()
                    )
                    try:
                        pred, _ = pred.split_after(series_end_time)
                    except Exception:
                        _log(
                            f"Historical predictions end early at {pred.end_time()} for Exponential "
                            "Smoothing model due to mismatch of forecast horizon and series length.",
                            output_f,
                            log,
                        )
                else:
                    model.fit(train_series)
                    pred = model.predict(**predict_params)

            preds.append(pred)
        return preds
    predict_params = {"n": len(val)}
    if train.n_components > 1:
        all_predictions = []
        _log(
            "Building separate Exponential Smoothing models for each component of multivariate "
            "time series",
            output_f,
            log,
        )
        _log(
            "Trained exponential smoothing models will not be saved to storage because separate "
            "models must be created for each grouping.",
            output_f,
            log,
        )
        for component in train.components:
            model = ExponentialSmoothing()
            if args.expanding_window:
                full_series = train[component].concatenate(val[component])
                series_end_time = full_series.end_time()
                freq = full_series.freq_str
                seasonal_period = freq_map.get(freq, 1)
                min_training_length = 2 * seasonal_period + 1
                safe_start_time = full_series.time_index[min_training_length]
                historical_params = {
                    "series": full_series,
                    "forecast_horizon": args.expanding_window_forecast_horizon,
                    "stride": args.expanding_window_stride,
                    "last_points_only": False,
                    "retrain": True,
                    "verbose": False,
                    "start": safe_start_time,
                    "overlap_end": True,
                }
                pred = model.historical_forecasts(**historical_params)
                all_preds_df = pd.concat([ts.to_dataframe() for ts in pred], axis=0)
                final_preds_df = all_preds_df.groupby(level=0).last()
                pred = TimeSeries_from_dataframe(final_preds_df, value_cols=[component])
                try:
                    pred, _ = pred.split_after(series_end_time)
                except Exception:
                    _log(
                        f"Historical predictions end early at {pred.end_time()} for Exponential "
                        "Smoothing model due to mismatch of forecast horizon and series length.",
                        output_f,
                        log,
                    )
            else:
                model.fit(train[component])
                pred = model.predict(**predict_params)
            all_predictions.append(pred)
        pred = all_predictions[0]
        for j in range(1, len(all_predictions)):
            pred = pred.stack(all_predictions[j])
    else:
        model = ExponentialSmoothing()
        if args.expanding_window:
            full_series = train.concatenate(val)
            series_end_time = full_series.end_time()
            freq = full_series.freq_str
            seasonal_period = freq_map.get(freq, 1)
            min_training_length = 2 * seasonal_period + 1
            safe_start_time = full_series.time_index[min_training_length]
            historical_params = {
                "series": full_series,
                "forecast_horizon": args.expanding_window_forecast_horizon,
                "stride": args.expanding_window_stride,
                "last_points_only": False,
                "retrain": True,
                "verbose": False,
                "start": safe_start_time,
                "overlap_end": True,
            }
            pred = model.historical_forecasts(**historical_params)
            all_preds_df = pd.concat([ts.to_dataframe() for ts in pred], axis=0)
            final_preds_df = all_preds_df.groupby(level=0).last()
            pred = TimeSeries_from_dataframe(final_preds_df, value_cols=train.components.tolist())
            try:
                pred, _ = pred.split_after(series_end_time)
            except Exception:
                _log(
                    f"Historical predictions end early at {pred.end_time()} for Exponential "
                    "Smoothing model due to mismatch of forecast horizon and series length.",
                    output_f,
                    log,
                )
        else:
            model.fit(train)
            pred = model.predict(**predict_params)
        _save_forecast_model(
            storage, args.report_uuid, "exponential-smoothing", model, log, output_f
        )

    return pred


def TimeSeries_from_dataframe(df: Any, value_cols: Any) -> Any:
    from darts import TimeSeries

    return TimeSeries.from_dataframe(
        df, time_col=None, value_cols=value_cols, fill_missing_dates=True
    )


def train_lagged_covariate_model(
    train: Any,
    val: Any,
    past_covariates: Any,
    future_covariates: Any,
    args: ForecastArgs,
    storage: Storage,
    model_name: str,
    model_cls: Any,
    log: LogCallback,
    output_f: Any,
) -> Any:
    """Shared trainer for the regression-style models (linreg, RF, LGBM, XGB)."""
    model_params: dict[str, Any] = {
        "lags": args.lags,
        "output_chunk_length": args.output_chunk_length,
    }
    fit_params: dict[str, Any] = {"series": train}
    n = len(val[0]) if isinstance(val, list) else len(val)
    predict_params: dict[str, Any] = {"n": n}
    if past_covariates is not None:
        model_params["lags_past_covariates"] = args.lags
        fit_params["past_covariates"] = past_covariates
        predict_params["past_covariates"] = past_covariates
    if future_covariates is not None:
        model_params["lags_future_covariates"] = [args.lags_future]
        fit_params["future_covariates"] = future_covariates
        predict_params["future_covariates"] = future_covariates
    if args.probabilistic and model_name in ("linear-regression", "lgbm", "xgboost"):
        model_params["likelihood"] = "quantile"
        predict_params["num_samples"] = args.probabilistic_num_samples
    if isinstance(train, list):
        predict_params["series"] = train
    model = model_cls(**model_params)

    if args.expanding_window:
        if isinstance(train, list):
            full_series = [
                train_series.concatenate(val_series)
                for train_series, val_series in zip(train, val, strict=False)
            ]
            series_end_time = full_series[0].end_time()
        else:
            full_series = train.concatenate(val)
            series_end_time = full_series.end_time()
        historical_params = {
            "series": full_series,
            "forecast_horizon": args.expanding_window_forecast_horizon,
            "stride": args.expanding_window_stride,
            "last_points_only": False,
            "retrain": True,
            "verbose": False,
            "overlap_end": True,
        }
        if past_covariates is not None:
            historical_params["past_covariates"] = past_covariates
        if future_covariates is not None:
            historical_params["future_covariates"] = future_covariates
        pred = model.historical_forecasts(**historical_params)
        if isinstance(train, list):
            new_preds = []
            for pred_series in pred:
                all_preds_df = pd.concat([ts.to_dataframe() for ts in pred_series], axis=0)
                final_preds_df = all_preds_df.groupby(level=0).last()
                new_pred = TimeSeries_from_dataframe(
                    final_preds_df, value_cols=train[0].components.tolist()
                )
                try:
                    new_pred, _ = new_pred.split_after(series_end_time)
                except Exception:
                    _log(
                        f"Historical predictions end early at {new_pred.end_time()} for {model_name} "
                        "model due to mismatch of forecast horizon and series length.",
                        output_f,
                        log,
                    )
                new_preds.append(new_pred)
            pred = new_preds
        else:
            all_preds_df = pd.concat([ts.to_dataframe() for ts in pred], axis=0)
            final_preds_df = all_preds_df.groupby(level=0).last()
            pred = TimeSeries_from_dataframe(final_preds_df, value_cols=train.components.tolist())
            try:
                pred, _ = pred.split_after(series_end_time)
            except Exception:
                _log(
                    f"Historical predictions end early at {pred.end_time()} for {model_name} model "
                    "due to mismatch of forecast horizon and series length.",
                    output_f,
                    log,
                )
    else:
        model.fit(**fit_params)
        pred = model.predict(**predict_params)

    _save_forecast_model(storage, args.report_uuid, model_name, model, log, output_f)

    return pred


def train_linear_regression_model(
    train: Any,
    val: Any,
    past_covariates: Any,
    future_covariates: Any,
    args: ForecastArgs,
    storage: Storage,
    log: LogCallback,
    output_f: Any,
) -> Any:
    from darts.models.forecasting.linear_regression_model import LinearRegressionModel

    return train_lagged_covariate_model(
        train,
        val,
        past_covariates,
        future_covariates,
        args,
        storage,
        "linear-regression",
        LinearRegressionModel,
        log,
        output_f,
    )


def train_random_forest_model(
    train: Any,
    val: Any,
    past_covariates: Any,
    future_covariates: Any,
    args: ForecastArgs,
    storage: Storage,
    log: LogCallback,
    output_f: Any,
) -> Any:
    from darts.models.forecasting.random_forest import RandomForestModel

    if args.probabilistic:
        _log("Probabilistic forecasting not supported for random forest model!", output_f, log)

    return train_lagged_covariate_model(
        train,
        val,
        past_covariates,
        future_covariates,
        args,
        storage,
        "random-forest",
        RandomForestModel,
        log,
        output_f,
    )


def train_lgbm_model(
    train: Any,
    val: Any,
    past_covariates: Any,
    future_covariates: Any,
    args: ForecastArgs,
    storage: Storage,
    log: LogCallback,
    output_f: Any,
) -> Any:
    from darts.models.forecasting.lgbm import LightGBMModel

    return train_lagged_covariate_model(
        train,
        val,
        past_covariates,
        future_covariates,
        args,
        storage,
        "lgbm",
        LightGBMModel,
        log,
        output_f,
    )


def train_xgboost_model(
    train: Any,
    val: Any,
    past_covariates: Any,
    future_covariates: Any,
    args: ForecastArgs,
    storage: Storage,
    log: LogCallback,
    output_f: Any,
) -> Any:
    from darts.models.forecasting.xgboost import XGBModel

    return train_lagged_covariate_model(
        train,
        val,
        past_covariates,
        future_covariates,
        args,
        storage,
        "xgboost",
        XGBModel,
        log,
        output_f,
    )


def train_torch_model(
    train: Any,
    val: Any,
    past_covariates: Any,
    future_covariates: Any,
    args: ForecastArgs,
    storage: Storage,
    model_name: str,
    model_cls: Any,
    probabilistic_model: Any,
    log: LogCallback,
    output_f: Any,
) -> Any:
    """Shared trainer for the torch-based models (TFT, N-Linear)."""
    try:
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping
    except ImportError:
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    my_stopper = EarlyStopping(
        monitor="train_loss",
        min_delta=0.001,
        patience=5,
        verbose=False,
        mode="min",
    )
    import torch

    model_params: dict[str, Any] = {
        "input_chunk_length": args.lags,
        "output_chunk_length": args.output_chunk_length,
        "n_epochs": args.epochs,
        "pl_trainer_kwargs": {
            "callbacks": [my_stopper],
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "enable_progress_bar": False,
            "logger": False,
            "enable_model_summary": False,
        },
    }
    fit_params: dict[str, Any] = {"series": train, "verbose": False}
    n = len(val[0]) if isinstance(val, list) else len(val)
    predict_params: dict[str, Any] = {"n": n}
    if past_covariates is not None:
        fit_params["past_covariates"] = past_covariates
        predict_params["past_covariates"] = past_covariates
    if future_covariates is not None:
        fit_params["future_covariates"] = future_covariates
        predict_params["future_covariates"] = future_covariates
    if args.probabilistic:
        model_params["likelihood"] = probabilistic_model
        if model_name == "nlinear":
            model_params["normalize"] = False
        predict_params["num_samples"] = args.probabilistic_num_samples
    if isinstance(train, list):
        predict_params["series"] = train
    model = model_cls(**model_params)

    if args.expanding_window:
        if isinstance(train, list):
            full_series = [
                train_series.concatenate(val_series)
                for train_series, val_series in zip(train, val, strict=False)
            ]
            series_end_time = full_series[0].end_time()
        else:
            full_series = train.concatenate(val)
            series_end_time = full_series.end_time()
        historical_params = {
            "series": full_series,
            "forecast_horizon": args.expanding_window_forecast_horizon,
            "stride": args.expanding_window_stride,
            "last_points_only": False,
            "retrain": True,
            "verbose": False,
            "overlap_end": True,
        }
        if past_covariates is not None:
            historical_params["past_covariates"] = past_covariates
        if future_covariates is not None:
            historical_params["future_covariates"] = future_covariates
        pred = model.historical_forecasts(**historical_params)
        if isinstance(train, list):
            new_preds = []
            for pred_series in pred:
                all_preds_df = pd.concat([ts.to_dataframe() for ts in pred_series], axis=0)
                final_preds_df = all_preds_df.groupby(level=0).last()
                new_pred = TimeSeries_from_dataframe(
                    final_preds_df, value_cols=train[0].components.tolist()
                )
                try:
                    new_pred, _ = new_pred.split_after(series_end_time)
                except Exception:
                    _log(
                        f"Historical predictions end early at {new_pred.end_time()} for {model_name} "
                        "model due to mismatch of forecast horizon and series length.",
                        output_f,
                        log,
                    )
                new_preds.append(new_pred)
            pred = new_preds
        else:
            all_preds_df = pd.concat([ts.to_dataframe() for ts in pred], axis=0)
            final_preds_df = all_preds_df.groupby(level=0).last()
            pred = TimeSeries_from_dataframe(final_preds_df, value_cols=train.components.tolist())
            try:
                pred, _ = pred.split_after(series_end_time)
            except Exception:
                _log(
                    f"Historical predictions end early at {pred.end_time()} for {model_name} model "
                    "due to mismatch of forecast horizon and series length.",
                    output_f,
                    log,
                )
    else:
        model.fit(**fit_params)
        pred = model.predict(**predict_params)

    _save_forecast_model(storage, args.report_uuid, model_name, model, log, output_f)

    return pred


def train_tft_model(
    train: Any,
    val: Any,
    past_covariates: Any,
    future_covariates: Any,
    args: ForecastArgs,
    probabilistic_model: Any,
    storage: Storage,
    log: LogCallback,
    output_f: Any,
) -> Any:
    from darts.models.forecasting.tft_model import TFTModel

    return train_torch_model(
        train,
        val,
        past_covariates,
        future_covariates,
        args,
        storage,
        "tft",
        TFTModel,
        probabilistic_model,
        log,
        output_f,
    )


def train_nlinear_model(
    train: Any,
    val: Any,
    past_covariates: Any,
    future_covariates: Any,
    args: ForecastArgs,
    probabilistic_model: Any,
    storage: Storage,
    log: LogCallback,
    output_f: Any,
) -> Any:
    from darts.models.forecasting.nlinear import NLinearModel

    return train_torch_model(
        train,
        val,
        past_covariates,
        future_covariates,
        args,
        storage,
        "nlinear",
        NLinearModel,
        probabilistic_model,
        log,
        output_f,
    )


def train_baseline_model(
    train: Any, val: Any, args: ForecastArgs, log: LogCallback, output_f: Any
) -> Any:
    from darts.models import NaiveSeasonal

    if isinstance(train, list):
        freq = train[0].freq_str
        n = len(val[0])
    else:
        freq = train.freq_str
        n = len(val)
    seasonal_period = freq_map.get(freq, 1)
    model_params = {"K": seasonal_period}
    model = NaiveSeasonal(**model_params)

    if args.expanding_window:
        if isinstance(train, list):
            all_preds = []
            full_series_list = [
                train_series.concatenate(val_series)
                for train_series, val_series in zip(train, val, strict=False)
            ]
            for series in full_series_list:
                series_end_time = series.end_time()
                historical_params = {
                    "series": series,
                    "forecast_horizon": args.expanding_window_forecast_horizon,
                    "stride": args.expanding_window_stride,
                    "last_points_only": False,
                    "retrain": True,
                    "verbose": False,
                    "overlap_end": True,
                }
                pred = model.historical_forecasts(**historical_params)
                all_preds_df = pd.concat([ts.to_dataframe() for ts in pred], axis=0)
                final_preds_df = all_preds_df.groupby(level=0).last()
                new_pred = TimeSeries_from_dataframe(
                    final_preds_df, value_cols=train[0].components.tolist()
                )
                try:
                    new_pred, _ = new_pred.split_after(series_end_time)
                except Exception:
                    _log(
                        f"Historical predictions end early at {new_pred.end_time()} for baseline "
                        "model due to mismatch of forecast horizon and series length.",
                        output_f,
                        log,
                    )
                all_preds.append(new_pred)

            return all_preds
        full_series = train.concatenate(val)
        series_end_time = full_series.end_time()
        historical_params = {
            "series": full_series,
            "forecast_horizon": args.expanding_window_forecast_horizon,
            "stride": args.expanding_window_stride,
            "last_points_only": False,
            "retrain": True,
            "verbose": False,
            "overlap_end": True,
        }
        pred = model.historical_forecasts(**historical_params)
        all_preds_df = pd.concat([ts.to_dataframe() for ts in pred], axis=0)
        final_preds_df = all_preds_df.groupby(level=0).last()
        pred = TimeSeries_from_dataframe(final_preds_df, value_cols=train.components.tolist())
        try:
            pred, _ = pred.split_after(series_end_time)
        except Exception:
            _log(
                f"Historical predictions end early at {pred.end_time()} for baseline model due to "
                "mismatch of forecast horizon and series length.",
                output_f,
                log,
            )
    else:
        if isinstance(train, list):
            preds = []
            for train_series in train:
                model = NaiveSeasonal(**model_params)
                model.fit(train_series)
                pred = model.predict(n=n)
                preds.append(pred)
            return preds
        model.fit(train)
        pred = model.predict(n=n)
    return pred


def nan_to_none_recursive(data_dict: dict) -> dict:
    cleaned_dict: dict[str, Any] = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            cleaned_dict[key] = nan_to_none_recursive(value)
        elif isinstance(value, list):
            cleaned_list: list[Any] = []
            for item in value:
                if isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
                    cleaned_list.append(None)
                else:
                    cleaned_list.append(item)
            cleaned_dict[key] = cleaned_list
        elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            cleaned_dict[key] = None
        else:
            cleaned_dict[key] = value
    return cleaned_dict


def evaluate(
    val: Any,
    pred: Any,
    train: Any,
    scaler: Any,
    groupings: Any,
    log: LogCallback,
    output_f: Any,
) -> dict:
    """Compute forecast metrics — ported verbatim from the Forecaster backend."""
    from darts.metrics import mae, mape, mase, mse, r2_score, rmse, smape

    if len(groupings) > 1:
        group_results: dict[str, Any] = {}
        logged_mape_message = False
        logged_mase_message = False
        for i, group in enumerate(groupings):
            pred_mae = mae(val[i], pred[i], component_reduction=None)
            pred_mse = mse(val[i], pred[i], component_reduction=None)
            pred_rmse = rmse(val[i], pred[i], component_reduction=None)
            if (val[i].values() > 0).all():
                pred_mape = mape(val[i], pred[i], component_reduction=None)
                pred_smape = smape(val[i], pred[i], component_reduction=None)
            else:
                if not logged_mape_message:
                    _log(
                        "Could not calculate MAPE and SMAPE accurately due to zero values in time series.",
                        output_f,
                        log,
                    )
                    logged_mape_message = True
                if val[i].n_components > 1:
                    pred_mape = [0] * val[i].n_components
                    pred_smape = [0] * val[i].n_components
                else:
                    pred_mape = 0
                    pred_smape = 0
            pred_r_squared = r2_score(val[i], pred[i], component_reduction=None)
            try:
                pred_mase = mase(val[i], pred[i], insample=train[i], component_reduction=None)
            except Exception as e:
                if not logged_mase_message:
                    _log(f"Could not calculate MASE accurately: {str(e)}", output_f, log)
                    logged_mase_message = True
                pred_mase = [1] * val[i].n_components if val[i].n_components > 1 else 1
            de_normalized_val = scaler[i].inverse_transform(val[i])
            de_normalized_pred = scaler[i].inverse_transform(pred[i])

            dn_pred_mae = mae(de_normalized_val, de_normalized_pred, component_reduction=None)
            dn_pred_mse = mse(de_normalized_val, de_normalized_pred, component_reduction=None)
            dn_pred_rmse = rmse(de_normalized_val, de_normalized_pred, component_reduction=None)

            if (de_normalized_val.values() > 0).all():
                dn_pred_mape = mape(de_normalized_val, de_normalized_pred, component_reduction=None)
                dn_pred_smape = smape(
                    de_normalized_val, de_normalized_pred, component_reduction=None
                )
            else:
                if val[i].n_components > 1:
                    dn_pred_mape = [0] * val[i].n_components
                    dn_pred_smape = [0] * val[i].n_components
                else:
                    dn_pred_mape = 0
                    dn_pred_smape = 0

            if isinstance(pred_mae, np.ndarray):
                component_results = {}
                for j, component in enumerate(val[i].components):
                    component_results[component] = {
                        "Normalized MAE": round(float(pred_mae[j]), 4),
                        "Normalized MSE": round(float(pred_mse[j]), 4),
                        "Normalized RMSE": round(float(pred_rmse[j]), 4),
                        "Normalized MAPE": round(float(pred_mape[j]), 4),
                        "Normalized SMAPE": round(float(pred_smape[j]), 4),
                        "De-Normalized MAE": round(float(dn_pred_mae[j]), 4),
                        "De-Normalized MSE": round(float(dn_pred_mse[j]), 4),
                        "De-Normalized RMSE": round(float(dn_pred_rmse[j]), 4),
                        "De-Normalized MAPE": round(float(dn_pred_mape[j]), 4),
                        "De-Normalized SMAPE": round(float(dn_pred_smape[j]), 4),
                        "R-squared": round(float(pred_r_squared[j]), 4),
                        "MASE": round(float(pred_mase[j]), 4),
                    }
                results = {
                    "Normalized MAE": round(
                        float(np.mean(np.array(pred_mae)[np.isfinite(pred_mae)])), 4
                    ),
                    "Normalized MSE": round(
                        float(np.mean(np.array(pred_mse)[np.isfinite(pred_mse)])), 4
                    ),
                    "Normalized RMSE": round(
                        float(np.mean(np.array(pred_rmse)[np.isfinite(pred_rmse)])), 4
                    ),
                    "Normalized MAPE": round(
                        float(np.mean(np.array(pred_mape)[np.isfinite(pred_mape)])), 4
                    ),
                    "Normalized SMAPE": round(
                        float(np.mean(np.array(pred_smape)[np.isfinite(pred_smape)])), 4
                    ),
                    "De-Normalized MAE": round(
                        float(np.mean(np.array(dn_pred_mae)[np.isfinite(dn_pred_mae)])), 4
                    ),
                    "De-Normalized MSE": round(
                        float(np.mean(np.array(dn_pred_mse)[np.isfinite(dn_pred_mse)])), 4
                    ),
                    "De-Normalized RMSE": round(
                        float(np.mean(np.array(dn_pred_rmse)[np.isfinite(dn_pred_rmse)])), 4
                    ),
                    "De-Normalized MAPE": round(
                        float(np.mean(np.array(dn_pred_mape)[np.isfinite(dn_pred_mape)])), 4
                    ),
                    "De-Normalized SMAPE": round(
                        float(np.mean(np.array(dn_pred_smape)[np.isfinite(dn_pred_smape)])), 4
                    ),
                    "R-squared": round(
                        float(np.mean(np.array(pred_r_squared)[np.isfinite(pred_r_squared)])), 4
                    ),
                    "MASE": round(float(np.mean(np.array(pred_mase)[np.isfinite(pred_mase)])), 4),
                    "components": component_results,
                }
            else:
                results = {
                    "Normalized MAE": round(float(pred_mae), 4),
                    "Normalized MSE": round(float(pred_mse), 4),
                    "Normalized RMSE": round(float(pred_rmse), 4),
                    "Normalized MAPE": round(float(pred_mape), 4),
                    "Normalized SMAPE": round(float(pred_smape), 4),
                    "De-Normalized MAE": round(float(dn_pred_mae), 4),
                    "De-Normalized MSE": round(float(dn_pred_mse), 4),
                    "De-Normalized RMSE": round(float(dn_pred_rmse), 4),
                    "De-Normalized MAPE": round(float(dn_pred_mape), 4),
                    "De-Normalized SMAPE": round(float(dn_pred_smape), 4),
                    "R-squared": round(float(pred_r_squared), 4),
                    "MASE": round(float(pred_mase), 4),
                    "components": {},
                }
            group_results[group] = results

        sum_dict: dict[str, float] = defaultdict(float)
        count_dict: dict[str, int] = defaultdict(int)
        for d in group_results.values():
            for key, value in d.items():
                if (
                    isinstance(value, (int, float))
                    and not math.isnan(value)
                    and not math.isinf(value)
                ):
                    sum_dict[key] += value
                    count_dict[key] += 1
        results = {}
        for key, total_sum in sum_dict.items():
            count = count_dict.get(key, 0)
            if count > 0:
                results[key] = round(total_sum / count, 4)
            else:
                results[key] = None

        component_averages: dict[str, Any] = {}
        for _grouping, group_dict in group_results.items():
            if len(group_dict["components"]) > 0:
                component_dict = group_dict["components"]
                if len(component_averages) == 0:
                    for component in component_dict:
                        component_averages[component] = {
                            key: [] for key in component_dict[component]
                        }
                for component in component_dict:
                    for measurement, measurement_value in component_dict[component].items():
                        component_averages[component][measurement].append(measurement_value)
            else:
                break

        if len(component_averages) > 0:
            new_component_results: dict[str, Any] = {}
            for component, measurement_lists in component_averages.items():
                averaged_component_dict: dict[str, Any] = {}
                for measurement, measurement_list in measurement_lists.items():
                    valid_numbers = [
                        x
                        for x in measurement_list
                        if isinstance(x, (int, float)) and not math.isnan(x) and not math.isinf(x)
                    ]
                    averaged_component_dict[measurement] = (
                        sum(valid_numbers) / len(valid_numbers) if valid_numbers else 0
                    )
                new_component_results[component] = averaged_component_dict
            results["components"] = new_component_results
        else:
            results["components"] = component_averages
        results["groupings"] = group_results
    else:
        pred_mae = mae(val, pred, component_reduction=None)
        pred_mse = mse(val, pred, component_reduction=None)
        pred_rmse = rmse(val, pred, component_reduction=None)
        if (val.values() > 0).all():
            pred_mape = mape(val, pred, component_reduction=None)
            pred_smape = smape(val, pred, component_reduction=None)
        else:
            _log(
                "Could not calculate MAPE and SMAPE accurately due to zero values in time series.",
                output_f,
                log,
            )
            if val.n_components > 1:
                pred_mape = [0] * val.n_components
                pred_smape = [0] * val.n_components
            else:
                pred_mape = 0
                pred_smape = 0

        pred_r_squared = r2_score(val, pred, component_reduction=None)
        try:
            pred_mase = mase(val, pred, insample=train, component_reduction=None)
        except Exception as e:
            _log(f"Could not calculate MASE: {str(e)}", output_f, log)
            pred_mase = [1] * val.n_components if val.n_components > 1 else 1

        de_normalized_val = scaler.inverse_transform(val)
        de_normalized_pred = scaler.inverse_transform(pred)

        dn_pred_mae = mae(de_normalized_val, de_normalized_pred, component_reduction=None)
        dn_pred_mse = mse(de_normalized_val, de_normalized_pred, component_reduction=None)
        dn_pred_rmse = rmse(de_normalized_val, de_normalized_pred, component_reduction=None)
        if (de_normalized_val.values() > 0).all():
            dn_pred_mape = mape(de_normalized_val, de_normalized_pred, component_reduction=None)
            dn_pred_smape = smape(de_normalized_val, de_normalized_pred, component_reduction=None)
        else:
            if val.n_components > 1:
                dn_pred_mape = [0] * val.n_components
                dn_pred_smape = [0] * val.n_components
            else:
                dn_pred_mape = 0
                dn_pred_smape = 0

        if isinstance(pred_mae, np.ndarray):
            group_results = {}
            for i, component in enumerate(val.components):
                group_results[component] = {
                    "Normalized MAE": round(float(pred_mae[i]), 4),
                    "Normalized MSE": round(float(pred_mse[i]), 4),
                    "Normalized RMSE": round(float(pred_rmse[i]), 4),
                    "Normalized MAPE": round(float(pred_mape[i]), 4),
                    "Normalized SMAPE": round(float(pred_smape[i]), 4),
                    "De-Normalized MAE": round(float(dn_pred_mae[i]), 4),
                    "De-Normalized MSE": round(float(dn_pred_mse[i]), 4),
                    "De-Normalized RMSE": round(float(dn_pred_rmse[i]), 4),
                    "De-Normalized MAPE": round(float(dn_pred_mape[i]), 4),
                    "De-Normalized SMAPE": round(float(dn_pred_smape[i]), 4),
                    "R-squared": round(float(pred_r_squared[i]), 4),
                    "MASE": round(float(pred_mase[i]), 4),
                }
            results = {
                "Normalized MAE": round(
                    float(np.mean(np.array(pred_mae)[np.isfinite(pred_mae)])), 4
                ),
                "Normalized MSE": round(
                    float(np.mean(np.array(pred_mse)[np.isfinite(pred_mse)])), 4
                ),
                "Normalized RMSE": round(
                    float(np.mean(np.array(pred_rmse)[np.isfinite(pred_rmse)])), 4
                ),
                "Normalized MAPE": round(
                    float(np.mean(np.array(pred_mape)[np.isfinite(pred_mape)])), 4
                ),
                "Normalized SMAPE": round(
                    float(np.mean(np.array(pred_smape)[np.isfinite(pred_smape)])), 4
                ),
                "De-Normalized MAE": round(
                    float(np.mean(np.array(dn_pred_mae)[np.isfinite(dn_pred_mae)])), 4
                ),
                "De-Normalized MSE": round(
                    float(np.mean(np.array(dn_pred_mse)[np.isfinite(dn_pred_mse)])), 4
                ),
                "De-Normalized RMSE": round(
                    float(np.mean(np.array(dn_pred_rmse)[np.isfinite(dn_pred_rmse)])), 4
                ),
                "De-Normalized MAPE": round(
                    float(np.mean(np.array(dn_pred_mape)[np.isfinite(dn_pred_mape)])), 4
                ),
                "De-Normalized SMAPE": round(
                    float(np.mean(np.array(dn_pred_smape)[np.isfinite(dn_pred_smape)])), 4
                ),
                "R-squared": round(
                    float(np.mean(np.array(pred_r_squared)[np.isfinite(pred_r_squared)])), 4
                ),
                "MASE": round(float(np.mean(np.array(pred_mase)[np.isfinite(pred_mase)])), 4),
                "groupings": {},
                "components": group_results,
            }
        else:
            results = {
                "Normalized MAE": round(float(pred_mae), 4),
                "Normalized MSE": round(float(pred_mse), 4),
                "Normalized RMSE": round(float(pred_rmse), 4),
                "Normalized MAPE": round(float(pred_mape), 4),
                "Normalized SMAPE": round(float(pred_smape), 4),
                "De-Normalized MAE": round(float(dn_pred_mae), 4),
                "De-Normalized MSE": round(float(dn_pred_mse), 4),
                "De-Normalized RMSE": round(float(dn_pred_rmse), 4),
                "De-Normalized MAPE": round(float(dn_pred_mape), 4),
                "De-Normalized SMAPE": round(float(dn_pred_smape), 4),
                "R-squared": round(float(pred_r_squared), 4),
                "MASE": round(float(pred_mase), 4),
                "groupings": {},
                "components": {},
            }
    results = nan_to_none_recursive(results)
    return results


def get_predictions(pred: Any, scaler: Any, groupings: Any, target_is_integer: Any) -> dict:
    """De-normalize predictions into the nested {groupings: {components: {date: value}}} shape."""
    results: dict[str, Any] = {}
    if len(groupings) > 1:
        grouping_results: dict[str, Any] = {}
        for i, grouping in enumerate(groupings):
            denormalized_pred = scaler[i].inverse_transform(pred[i])
            if denormalized_pred.n_components > 1:
                component_results = {}
                for j, component in enumerate(denormalized_pred.components):
                    if denormalized_pred[component].is_probabilistic:
                        denormalized_pred_dict = (
                            denormalized_pred[component].mean().to_series().to_dict()
                        )
                    else:
                        denormalized_pred_dict = denormalized_pred[component].to_series().to_dict()
                    if target_is_integer[j]:
                        denormalized_pred_dict = {
                            str(key): int(value) for key, value in denormalized_pred_dict.items()
                        }
                    else:
                        denormalized_pred_dict = {
                            str(key): round(value, 4)
                            for key, value in denormalized_pred_dict.items()
                        }
                    component_results[component] = denormalized_pred_dict
                grouping_results[grouping] = {"components": component_results}
            else:
                if denormalized_pred.is_probabilistic:
                    denormalized_pred_dict = denormalized_pred.mean().to_series().to_dict()
                else:
                    denormalized_pred_dict = denormalized_pred.to_series().to_dict()
                if target_is_integer:
                    denormalized_pred_dict = {
                        str(key): int(value) for key, value in denormalized_pred_dict.items()
                    }
                else:
                    denormalized_pred_dict = {
                        str(key): round(value, 4) for key, value in denormalized_pred_dict.items()
                    }
                grouping_results[grouping] = {
                    "components": {denormalized_pred.components[0]: denormalized_pred_dict}
                }
        results["groupings"] = grouping_results
    else:
        components_results: dict[str, Any] = {}
        denormalized_pred = scaler.inverse_transform(pred)
        if denormalized_pred.n_components > 1:
            for i, component in enumerate(denormalized_pred.components):
                if denormalized_pred[component].is_probabilistic:
                    denormalized_pred_dict = (
                        denormalized_pred[component].mean().to_series().to_dict()
                    )
                else:
                    denormalized_pred_dict = denormalized_pred[component].to_series().to_dict()
                if target_is_integer[i]:
                    denormalized_pred_dict = {
                        str(key): int(value) for key, value in denormalized_pred_dict.items()
                    }
                else:
                    denormalized_pred_dict = {
                        str(key): round(value, 4) for key, value in denormalized_pred_dict.items()
                    }
                components_results[component] = denormalized_pred_dict
        else:
            if denormalized_pred.is_probabilistic:
                denormalized_pred_dict = denormalized_pred.mean().to_series().to_dict()
            else:
                denormalized_pred_dict = denormalized_pred.to_series().to_dict()
            if target_is_integer:
                denormalized_pred_dict = {
                    str(key): int(value) for key, value in denormalized_pred_dict.items()
                }
            else:
                denormalized_pred_dict = {
                    str(key): round(value, 4) for key, value in denormalized_pred_dict.items()
                }
            components_results[denormalized_pred.components[0]] = denormalized_pred_dict
        results = {"components": components_results}
    return results


def visualize_forecast(
    val: Any,
    pred: Any,
    train: Any,
    scaler: Any,
    groupings: Any,
    args: ForecastArgs,
    model_name: str,
    storage: Storage,
    log: LogCallback,
    output_f: Any,
    target_is_integer: Any,
) -> None:
    """Write the raw chart data as JSON so the interactive plot can be rendered client-side."""
    if len(groupings) == 1:
        components_dict: dict[str, Any] = {}
        de_normalized_pred = scaler.inverse_transform(pred)
        de_normalized_train = scaler.inverse_transform(train)
        if args.expanding_window and not args.generate_real_predictions:
            de_normalized_val = scaler.inverse_transform(val)
            de_normalized_full = de_normalized_train.concatenate(
                de_normalized_val, ignore_time_axis=False
            )
        else:
            de_normalized_full = de_normalized_train
        for i, component_name in enumerate(de_normalized_pred.components):
            resolution = de_normalized_full[component_name].time_index.resolution
            if resolution in ["microsecond", "millisecond", "nanosecond"]:
                datetime_format = "%Y-%m-%d %H:%M:%S.%f"
            else:
                datetime_format = "%Y-%m-%d %H:%M:%S"
            train_x = [
                t.strftime(datetime_format)
                for t in de_normalized_full[component_name].time_index.tolist()
            ]
            if target_is_integer[i]:
                train_y = [
                    int(x) for x in de_normalized_full[component_name].values().flatten().tolist()
                ]
            else:
                train_y = [
                    round(x, 4)
                    for x in de_normalized_full[component_name].values().flatten().tolist()
                ]
            component_dict: dict[str, Any] = {"train_x": train_x, "train_y": train_y}
            if not args.generate_real_predictions and not args.expanding_window:
                de_normalized_val = scaler.inverse_transform(val)
                val_x = [
                    t.strftime(datetime_format)
                    for t in de_normalized_val[component_name].time_index.tolist()
                ]
                if target_is_integer[i]:
                    val_y = [
                        int(x)
                        for x in de_normalized_val[component_name].values().flatten().tolist()
                    ]
                else:
                    val_y = [
                        round(x, 4)
                        for x in de_normalized_val[component_name].values().flatten().tolist()
                    ]
                component_dict["val_x"] = val_x
                component_dict["val_y"] = val_y

            pred_array = de_normalized_pred[component_name].values().flatten()
            not_nan_mask = ~np.isnan(pred_array)
            valid_times = de_normalized_pred.time_index[not_nan_mask]
            valid_values = pred_array[not_nan_mask]

            pred_x = [t.strftime(datetime_format) for t in valid_times.tolist()]
            if target_is_integer[i]:
                pred_y = [int(x) for x in valid_values.tolist()]
            else:
                pred_y = [round(x, 4) for x in valid_values.tolist()]
            component_dict["pred_x"] = pred_x
            component_dict["pred_y"] = pred_y
            components_dict[component_name] = component_dict
        final_dict: dict[str, Any] = {"": components_dict}
    else:
        final_dict = {}
        for i, grouping in enumerate(groupings):
            components_dict = {}
            de_normalized_pred = scaler[i].inverse_transform(pred[i])
            de_normalized_train = scaler[i].inverse_transform(train[i])
            if args.expanding_window and not args.generate_real_predictions:
                de_normalized_val = scaler[i].inverse_transform(val[i])
                de_normalized_full = de_normalized_train.concatenate(
                    de_normalized_val, ignore_time_axis=False
                )
            else:
                de_normalized_full = de_normalized_train
            for _j, component_name in enumerate(de_normalized_pred.components):
                resolution = de_normalized_full[component_name].time_index.resolution
                if resolution in ["microsecond", "millisecond", "nanosecond"]:
                    datetime_format = "%Y-%m-%d %H:%M:%S.%f"
                else:
                    datetime_format = "%Y-%m-%d %H:%M:%S"
                train_x = [
                    t.strftime(datetime_format)
                    for t in de_normalized_full[component_name].time_index.tolist()
                ]
                train_y = [
                    round(x, 4)
                    for x in de_normalized_full[component_name].values().flatten().tolist()
                ]
                component_dict = {"train_x": train_x, "train_y": train_y}
                if not args.generate_real_predictions and not args.expanding_window:
                    de_normalized_val = scaler[i].inverse_transform(val[i])
                    val_x = [
                        t.strftime(datetime_format)
                        for t in de_normalized_val[component_name].time_index.tolist()
                    ]
                    val_y = [
                        round(x, 4)
                        for x in de_normalized_val[component_name].values().flatten().tolist()
                    ]
                    component_dict["val_x"] = val_x
                    component_dict["val_y"] = val_y
                pred_x = [
                    t.strftime(datetime_format)
                    for t in de_normalized_pred[component_name].time_index.tolist()
                ]
                pred_y = [
                    round(x, 4)
                    for x in de_normalized_pred[component_name].values().flatten().tolist()
                ]
                component_dict["pred_x"] = pred_x
                component_dict["pred_y"] = pred_y
                components_dict[component_name] = component_dict
            final_dict[grouping] = components_dict
    final_json = json.dumps(final_dict, indent=4)
    try:
        storage.put_text(f"{args.report_uuid}/forecast_data_{model_name}", final_json)
    except Exception:
        _log(f"Failed to save {model_name} visualization data to storage", output_f, log)


METRIC_KEYS = [
    "Normalized MAE",
    "Normalized MSE",
    "Normalized RMSE",
    "Normalized MAPE",
    "Normalized SMAPE",
    "De-Normalized MAE",
    "De-Normalized MSE",
    "De-Normalized RMSE",
    "De-Normalized MAPE",
    "De-Normalized SMAPE",
    "R-squared",
    "MASE",
]


def write_results(
    all_results: dict[str, Any],
    args: ForecastArgs,
    storage: Storage,
    output_f: Any,
    log: LogCallback,
) -> int:
    """Write the flattened metrics report CSV + the nested results JSON.

    CSV rows: one per model (overall), plus one row per grouping and per
    component/grouping combination. Empty grouping/component cells mean
    "averaged across all".
    """
    rows: list[dict[str, Any]] = []
    for model_name, results in all_results.items():
        overall = {k: v for k, v in results.items() if k not in ("components", "groupings")}
        rows.append({"model": model_name, "grouping": "", "component": "", **overall})
        groupings = results.get("groupings") or {}
        for group_name, group_metrics in groupings.items():
            scalar = {
                k: v for k, v in group_metrics.items() if k not in ("components", "groupings")
            }
            rows.append({"model": model_name, "grouping": group_name, "component": "", **scalar})
            for component_name, comp_metrics in (group_metrics.get("components") or {}).items():
                rows.append(
                    {
                        "model": model_name,
                        "grouping": group_name,
                        "component": component_name,
                        **comp_metrics,
                    }
                )
        for component_name, comp_metrics in (results.get("components") or {}).items():
            rows.append(
                {"model": model_name, "grouping": "", "component": component_name, **comp_metrics}
            )

    columns = ["model", "grouping", "component"] + METRIC_KEYS
    df = pd.DataFrame(rows, columns=columns)
    storage.write_csv(f"{args.report_uuid}/results", df, index=False)

    try:
        storage.put_text(f"{args.report_uuid}/results.json", json.dumps(all_results, indent=4))
    except Exception:
        _log("Failed to save results JSON to storage", output_f, log)

    return len(rows)


def check_args(args: ForecastArgs, log: LogCallback, output_f: Any) -> ForecastArgs:
    if args.lags <= 0:
        _log("Lags argument cannot have value <=0. Setting to default 10.", output_f, log)
        args.lags = 10
    if args.output_chunk_length <= 0:
        _log(
            "Output chunk length argument cannot have value <=0. Setting to default 10.",
            output_f,
            log,
        )
        args.output_chunk_length = 10
    if args.epochs <= 0:
        _log("Epochs argument cannot have value <=0. Setting to default 10.", output_f, log)
        args.epochs = 10
    if args.probabilistic and args.probabilistic_num_samples <= 0:
        _log(
            "Number of samples for probabilistic forecasting cannot have value <=0. Setting to default 100.",
            output_f,
            log,
        )
        args.probabilistic_num_samples = 100
    return args


def forecast_trainer(
    args: ForecastArgs,
    storage: Storage,
    on_progress: ProgressCallback | None = None,
    log_cb: LogCallback | None = None,
    cancel_token: Any = None,
) -> None:
    """Main forecasting trainer entry point (mode="forecasting" jobs)."""
    output_buf = io.StringIO()

    def log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    def check_cancel() -> bool:
        return (
            cancel_token is not None and hasattr(cancel_token, "is_set") and cancel_token.is_set()
        )

    try:
        _run_forecast_training(args, storage, output_buf, log, on_progress, check_cancel)
    finally:
        storage.put_text(f"{args.report_uuid}/output_log", output_buf.getvalue())


def _run_forecast_training(
    args: ForecastArgs,
    storage: Storage,
    output_f: Any,
    log: LogCallback,
    on_progress: ProgressCallback | None,
    check_cancel: Callable[[], bool],
) -> None:
    from darts.utils.likelihood_models import (
        GaussianLikelihood,
        LaplaceLikelihood,
        QuantileRegression,
    )

    needs_torch_likelihood = args.probabilistic and any(
        m in ("tft", "nlinear") for m in args.train_group
    )
    if needs_torch_likelihood:
        if args.probabilistic_likelihood == "gaussian":
            probabilistic_model: Any = GaussianLikelihood()
        elif args.probabilistic_likelihood == "quantile":
            probabilistic_model = QuantileRegression([0.05, 0.25, 0.5, 0.75, 0.95])
        elif args.probabilistic_likelihood == "laplace":
            probabilistic_model = LaplaceLikelihood()
        else:
            probabilistic_model = GaussianLikelihood()
    else:
        probabilistic_model = None

    all_results: dict[str, Any] = {}

    args = check_args(args, log, output_f)
    _log(f"Model training list: {args.train_group}", output_f, log)
    df = read_in_data(args, storage, log, output_f)
    if df is None:
        raise ValueError(
            "Failed to read the dataset for forecasting — check the output log for details."
        )
    train, val, past_covariates, future_covariates, scaler, groupings = setUpCovariates(
        df, args, log, output_f
    )
    if train is None:
        raise ValueError("Failed to prepare the time series — check the output log for details.")

    target_is_integer = []
    for target_column in args.target:
        target_is_integer.append(
            df[target_column].dropna().astype(float).apply(float.is_integer).all()
        )

    total_models = len(args.train_group) + 1

    _log("Starting Training", output_f, log)
    try:
        baseline_pred = train_baseline_model(train, val, args, log, output_f)
        if not args.generate_real_predictions:
            baseline_results = evaluate(val, baseline_pred, train, scaler, groupings, log, output_f)
        else:
            baseline_results = get_predictions(baseline_pred, scaler, groupings, target_is_integer)
        all_results["Baseline Average Prediction"] = baseline_results
        _log("Baseline Average Prediction training complete.", output_f, log)
    except Exception as e:
        _log(f"Error in training Baseline Average Prediction: {str(e)}", output_f, log)
    if args.visualize and "Baseline Average Prediction" in all_results:
        visualize_forecast(
            val,
            baseline_pred,
            train,
            scaler,
            groupings,
            args,
            "baseline",
            storage,
            log,
            output_f,
            target_is_integer,
        )
    if on_progress:
        on_progress(1, total_models, "Training (1/" + str(total_models) + ")")

    for i, model in enumerate(args.train_group):
        if check_cancel():
            _log("Training cancelled by user.", output_f, log)
            break

        pred = None
        _log(f"Evaluating model: {model}", output_f, log)
        if model == "linear-regression":
            try:
                pred = train_linear_regression_model(
                    train, val, past_covariates, future_covariates, args, storage, log, output_f
                )
            except Exception as e:
                pred = None
                _log(f"Error in training {model}: {e}", output_f, log)
        elif model == "random-forest":
            try:
                pred = train_random_forest_model(
                    train, val, past_covariates, future_covariates, args, storage, log, output_f
                )
            except Exception as e:
                pred = None
                _log(f"Error in training {model}: {e}", output_f, log)
        elif model == "lgbm":
            try:
                pred = train_lgbm_model(
                    train, val, past_covariates, future_covariates, args, storage, log, output_f
                )
            except Exception as e:
                pred = None
                _log(f"Error in training {model}: {e}", output_f, log)
        elif model == "xgboost":
            try:
                pred = train_xgboost_model(
                    train, val, past_covariates, future_covariates, args, storage, log, output_f
                )
            except Exception as e:
                pred = None
                _log(f"Error in training {model}: {e}", output_f, log)
        elif model == "tft":
            if future_covariates is None:
                _log("Skipping tft — requires future covariates to be included.", output_f, log)
                pred = None
            else:
                try:
                    pred = train_tft_model(
                        train,
                        val,
                        past_covariates,
                        future_covariates,
                        args,
                        probabilistic_model,
                        storage,
                        log,
                        output_f,
                    )
                except Exception as e:
                    pred = None
                    _log(f"Error in training {model}: {e}", output_f, log)
        elif model == "nlinear":
            try:
                pred = train_nlinear_model(
                    train,
                    val,
                    past_covariates,
                    future_covariates,
                    args,
                    probabilistic_model,
                    storage,
                    log,
                    output_f,
                )
            except Exception as e:
                pred = None
                _log(f"Error in training {model}: {e}", output_f, log)
        elif model == "arima":
            if past_covariates is not None:
                _log(
                    "ARIMA model can only utilize future covariates, not past covariates.",
                    output_f,
                    log,
                )
            try:
                pred = train_arima_model(
                    train, val, future_covariates, args, storage, log, output_f
                )
            except Exception as e:
                pred = None
                _log(f"Error in training {model}: {e}", output_f, log)
        elif model == "exponential-smoothing":
            if past_covariates is not None or future_covariates is not None:
                _log(
                    "Exponential Smoothing model cannot use covariates, so these will be ignored.",
                    output_f,
                    log,
                )
            try:
                pred = train_exponential_smoothing_model(train, val, args, storage, log, output_f)
            except Exception as e:
                pred = None
                _log(f"Error in training {model}: {e}", output_f, log)
        else:
            _log(f"Skipping {model} — unknown forecasting model.", output_f, log)

        if pred is not None:
            if not args.generate_real_predictions:
                results = evaluate(val, pred, train, scaler, groupings, log, output_f)
            else:
                results = get_predictions(pred, scaler, groupings, target_is_integer)
            if args.visualize:
                visualize_forecast(
                    val,
                    pred,
                    train,
                    scaler,
                    groupings,
                    args,
                    model,
                    storage,
                    log,
                    output_f,
                    target_is_integer,
                )
            all_results[model] = results
            _log(f"{model} training complete.", output_f, log)

        if on_progress:
            on_progress(i + 2, total_models, f"Training ({i + 2}/{total_models})")

    if not all_results and not check_cancel():
        raise ValueError(
            "No models were trained — requested: "
            + ", ".join(args.train_group)
            + ". Check the output log for skip/error details."
        )

    _log("==============FINAL RESULTS=================", output_f, log)
    _log(str(all_results), output_f, log)

    if all_results:
        write_results(all_results, args, storage, output_f, log)
