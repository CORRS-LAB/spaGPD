"""Pseudo-proportion (soft-NN) supervision used to train the deconvolution head."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def build_pseudo_proportions(
    feature_st: np.ndarray,
    feature_sc: np.ndarray,
    sc_adata,
    celltype_col: str,
    cell_types: list[str],
    temperature: float,
) -> np.ndarray:
    """Cosine-similarity soft-kNN weighting from ST features onto scRNA labels.

    Returns an (n_spots, n_celltypes) float32 matrix in [0, 1] summing to 1 per row.
    """
    labels = sc_adata.obs[celltype_col].astype(str).to_numpy()
    feat_st = F.normalize(torch.tensor(feature_st, dtype=torch.float32), p=2, dim=1)
    feat_sc = F.normalize(torch.tensor(feature_sc, dtype=torch.float32), p=2, dim=1)
    weights = torch.softmax(torch.matmul(feat_st, feat_sc.T) / temperature, dim=1)

    cell_type_matrix = np.zeros((len(labels), len(cell_types)), dtype=np.float32)
    celltype_to_index = {cell_type: idx for idx, cell_type in enumerate(cell_types)}
    for idx, label in enumerate(labels):
        if label in celltype_to_index:
            cell_type_matrix[idx, celltype_to_index[label]] = 1.0

    pseudo = torch.matmul(weights, torch.tensor(cell_type_matrix, dtype=torch.float32))
    pseudo = pseudo / (pseudo.sum(dim=1, keepdim=True) + 1e-8)
    return pseudo.numpy().astype(np.float32)


def robust_l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize after per-row max-abs rescale."""
    scale = np.max(np.abs(x), axis=1, keepdims=True)
    scale[scale < 1e-12] = 1.0
    x_scaled = x / scale
    norm = np.linalg.norm(x_scaled, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    out = x_scaled / norm
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
