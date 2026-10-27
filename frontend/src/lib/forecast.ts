/**
 * Shared forecasting-mode constants and helpers (ported from the
 * Forecaster web app's model list, labels and chart palette).
 */

export const FORECASTING_MODELS = [
  "linear-regression",
  "random-forest",
  "lgbm",
  "xgboost",
  "nlinear",
  "tft",
  "exponential-smoothing",
  "arima",
];

export const FORECAST_MODEL_LABELS: Record<string, string> = {
  "linear-regression": "Linear Regression",
  "random-forest": "Random Forest",
  lgbm: "Light Gradient Boosting Machine",
  xgboost: "XGBoost",
  nlinear: "N-Linear Model",
  tft: "Temporal Fusion Transformer",
  "exponential-smoothing": "Exponential Smoothing",
  arima: "ARIMA",
  baseline: "Baseline Average Prediction",
  "Baseline Average Prediction": "Baseline Average Prediction",
};

export const FORECAST_PARAM_SECTIONS: Record<string, string[]> = {
  arima: ["arima"],
  "linear-regression": ["shared"],
  "random-forest": ["shared"],
  lgbm: ["shared"],
  xgboost: ["shared"],
  nlinear: ["shared", "nn"],
  tft: ["shared", "nn"],
  "exponential-smoothing": [],
};

export const FORECAST_METRIC_ORDER = [
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
];

export const FORECAST_TRACE_COLORS = {
  train: "rgb(54, 162, 235)",
  val: "rgb(75, 192, 192)",
  pred: "rgb(255, 99, 132)",
  forecast: "rgb(255, 165, 0)",
};

export const COMPARE_PALETTE = [
  "rgb(54, 162, 235)",
  "rgb(255, 99, 132)",
  "rgb(75, 192, 192)",
  "rgb(255, 159, 64)",
  "rgb(153, 102, 255)",
  "rgb(46, 204, 113)",
  "rgb(255, 206, 86)",
  "rgb(75, 75, 75)",
];

export interface ForecastChartData {
  [grouping: string]: {
    [component: string]: {
      train_x: string[];
      train_y: number[];
      val_x?: string[];
      val_y?: number[];
      pred_x: string[];
      pred_y: number[];
    };
  };
}

export interface ForecastsPayload {
  [model: string]: {
    groupings?: Record<string, { components: Record<string, Record<string, number>> }>;
    components?: Record<string, Record<string, number>>;
  };
}

export interface ForecastMetricRow {
  model: string;
  grouping: string;
  component: string;
  [metric: string]: string | number | null;
}

export function modelLabel(model: string): string {
  return FORECAST_MODEL_LABELS[model] ?? model;
}

export function sortDates(keys: string[]): string[] {
  return [...keys].sort((a, b) => new Date(a).getTime() - new Date(b).getTime());
}
