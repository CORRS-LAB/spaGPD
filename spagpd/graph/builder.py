"""Graph construction utilities (KNN, symmetrization, normalization)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree


def symmetrize_graph(adj: sp.spmatrix) -> sp.coo_matrix:
    adj = adj.tocoo().astype(np.float32)
    adj = adj + adj.T
    adj.data[:] = 1.0
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    return adj.tocoo()


def build_knn_adj(features: np.ndarray, n_neighbors: int) -> sp.coo_matrix:
    """Build a symmetric kNN adjacency using cKDTree; values are 1.0; diagonal is 0."""
    n_obs = features.shape[0]
    k = min(n_neighbors, max(n_obs - 1, 1))
    _, idx = cKDTree(features).query(features, k=k + 1)
    idx = np.asarray(idx)
    if idx.ndim == 1:
        idx = idx[:, None]
    idx = idx[:, 1:]
    src = np.repeat(np.arange(n_obs), k)
    dst = idx.reshape(-1)
    adj = sp.coo_matrix(
        (np.ones_like(src, dtype=np.float32), (src, dst)), shape=(n_obs, n_obs)
    )
    return symmetrize_graph(adj)


def normalize_adjacency(adj: sp.spmatrix) -> sp.coo_matrix:
    """Symmetric normalization D^{-1/2} (A + I) D^{-1/2} (used by GCN)."""
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0], dtype=np.float32, format="coo")
    rowsum = np.asarray(adj.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
    mask = rowsum > 0
    inv_sqrt[mask] = np.power(rowsum[mask], -0.5)
    degree = sp.diags(inv_sqrt)
    return degree.dot(adj).dot(degree).tocoo().astype(np.float32)


def scipy_to_torch_sparse(adj: sp.spmatrix) -> torch.Tensor:
    coo = adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


def knn_edge_index(features: np.ndarray, k: int, add_self_loop: bool = True) -> torch.Tensor:
    """Build a 2 x E edge_index (long) tensor; bidirected + de-duplicated.

    Adds a self-loop for every node by default.
    """
    n_obs = features.shape[0]
    k_eff = min(k, max(n_obs - 1, 1))
    _, idx = cKDTree(features).query(features, k=k_eff + 1)
    idx = np.asarray(idx)
    if idx.ndim == 1:
        idx = idx[:, None]
    idx = idx[:, 1:]
    src = np.repeat(np.arange(n_obs), k_eff)
    dst = idx.reshape(-1)
    parts_src = [src, dst]
    parts_dst = [dst, src]
    if add_self_loop:
        parts_src.append(np.arange(n_obs))
        parts_dst.append(np.arange(n_obs))
    edges = np.vstack(
        [np.concatenate(parts_src), np.concatenate(parts_dst)]
    )
    edges = np.unique(edges, axis=1)
    return torch.as_tensor(edges, dtype=torch.long)
