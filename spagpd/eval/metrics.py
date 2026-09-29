"""Evaluation metrics: RMSE / PCC / SSIM / JSD at spot and cell-type level."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from spagpd.io.utils import standardize_prediction

METRICS = ("RMSE", "PCC", "SSIM", "JSD")


def safe_pcc(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(pearsonr(x, y)[0])


def safe_jsd(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    x = np.clip(np.asarray(x, dtype=float), 0.0, None) + eps
    y = np.clip(np.asarray(y, dtype=float), 0.0, None) + eps
    return float(jensenshannon(x / x.sum(), y / y.sum()))


def ssim_1d(x: np.ndarray, y: np.ndarray) -> float:
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x), np.var(y)
    covariance = np.mean((x - mean_x) * (y - mean_y))
    c1 = c2 = 1e-4
    denominator = (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
    if denominator == 0:
        return np.nan
    return float(
        ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / denominator
    )


def metric_record(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "PCC": safe_pcc(y_true, y_pred),
        "SSIM": ssim_1d(y_true, y_pred),
        "JSD": safe_jsd(y_true, y_pred),
    }


def evaluate_prediction(
    prediction: pd.DataFrame,
    processed_dir,
    cell_types: list[str],
) -> pd.DataFrame:
    truth = pd.read_csv(processed_dir / "true_proportions.csv", index_col=0)
    truth.index = truth.index.astype(str)
    truth = truth.loc[:, cell_types]
    prediction = standardize_prediction(
        prediction, truth.index.astype(str).tolist(), cell_types
    )

    spot_records = [
        metric_record(truth.loc[spot].values, prediction.loc[spot].values)
        for spot in truth.index
    ]
    celltype_records = [
        metric_record(truth[cell_type].values, prediction[cell_type].values)
        for cell_type in cell_types
    ]
    spot_metrics = pd.DataFrame(spot_records).mean()
    celltype_metrics = pd.DataFrame(celltype_records).mean()
    summary = {
        **{f"Spot Avg {m}": spot_metrics[m] for m in METRICS},
        **{f"CellType Avg {m}": celltype_metrics[m] for m in METRICS},
    }
    return pd.DataFrame([summary])
