"""Shared I/O and runtime helpers used across spaGPD embedding pipelines."""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def set_runtime_cache(cache_subdir: str = "spaGPD_runtime_cache") -> None:
    """Redirect numba/matplotlib caches to a writable temp dir (macOS-safe)."""
    cache_root = Path(tempfile.gettempdir()) / cache_subdir
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_cell_types(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def to_dense_float32(x) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def matrix_from_layer(adata, layer: str):
    if layer in adata.layers:
        return adata.layers[layer]
    return adata.X


def use_lognorm_as_x(adata):
    adata = adata.copy()
    if "lognorm" not in adata.layers:
        raise ValueError("AnnData is missing layers['lognorm']")
    adata.X = adata.layers["lognorm"].copy()
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr()
    return adata


def normalize_rows(values, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    return values / np.maximum(values.sum(axis=1, keepdims=True), eps)


def standardize_prediction(
    prop: pd.DataFrame, spot_ids: list[str], cell_types: list[str]
) -> pd.DataFrame:
    prop = prop.copy()
    prop.index = prop.index.astype(str)
    missing = [spot for spot in spot_ids if spot not in prop.index]
    if missing:
        raise ValueError(f"Prediction is missing spots, examples: {missing[:5]}")
    prop = prop.loc[spot_ids]
    for cell_type in cell_types:
        if cell_type not in prop.columns:
            prop[cell_type] = 0.0
    prop = prop[cell_types].astype(float)
    prop = prop.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prop[prop < 0] = 0.0
    return pd.DataFrame(normalize_rows(prop.values), index=spot_ids, columns=cell_types)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
