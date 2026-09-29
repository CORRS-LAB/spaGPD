"""Expression PCA helpers (shared between GCN-CLPLS integration and any future encoders)."""
from __future__ import annotations

import numpy as np


def pca_scores(x: np.ndarray, n_pcs: int) -> np.ndarray:
    """Truncated SVD PCA. Returns float32 scores."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return (u[:, :n_pcs] * s[:n_pcs]).astype(np.float32)


def build_expression_pca_features(
    sc_adata,
    st_adata,
    n_pcs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint PCA over (scRNA + ST) expression; standardize by scRNA statistics."""
    from spagpd.io.utils import to_dense_float32

    x_sc = np.nan_to_num(to_dense_float32(sc_adata.X).astype(np.float64), copy=False)
    x_st = np.nan_to_num(to_dense_float32(st_adata.X).astype(np.float64), copy=False)
    x_all = np.vstack([x_sc, x_st])
    n_pcs = min(n_pcs, x_all.shape[0] - 1, x_all.shape[1] - 1)
    if n_pcs < 2:
        raise ValueError(f"PCA dimension is too small: n_pcs={n_pcs}")
    x_pca = pca_scores(x_all, n_pcs)
    pca_sc = x_pca[: x_sc.shape[0]]
    pca_st = x_pca[x_sc.shape[0] :]
    mean = pca_sc.mean(axis=0, keepdims=True)
    std = pca_sc.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return ((pca_sc - mean) / std).astype(np.float32), ((pca_st - mean) / std).astype(np.float32)


def normalize_feature(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (x - mean) / std
