"""Marker-gene-guided NNLS target used to regularize the final prediction."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import nnls

from spagpd.io.utils import matrix_from_layer, normalize_rows


def library_normalize(matrix, scale_factor: float = 6000.0):
    if sp.issparse(matrix):
        library_size = np.asarray(matrix.sum(axis=1)).reshape(-1)
        library_size[library_size <= 0] = 1.0
        return matrix.multiply((scale_factor / library_size)[:, None]).tocsr()
    matrix = np.asarray(matrix, dtype=np.float32)
    library_size = matrix.sum(axis=1, keepdims=True)
    library_size[library_size <= 0] = 1.0
    return matrix / library_size * scale_factor


def mean_rows(matrix, mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    return np.asarray(matrix[mask].mean(axis=0)).reshape(-1).astype(np.float32)


def select_marker_genes(
    signature_all: np.ndarray,
    top_per_celltype: int,
) -> np.ndarray:
    """Score genes by specificity * abundance, drop zero/variance, return sorted indices."""
    selected: set[int] = set()
    global_mean = signature_all.mean(axis=0)
    n_celltypes = signature_all.shape[0]
    for celltype_index in range(n_celltypes):
        celltype_mean = signature_all[celltype_index]
        other_mean = (signature_all.sum(axis=0) - celltype_mean) / max(n_celltypes - 1, 1)
        specificity = np.log1p(celltype_mean) - np.log1p(other_mean)
        abundance = np.log1p(celltype_mean)
        score = specificity * np.maximum(abundance, 0.0)
        score[celltype_mean <= 0] = -np.inf
        score[global_mean <= 0] = -np.inf
        n_genes = min(int(top_per_celltype), int(np.isfinite(score).sum()))
        if n_genes > 0:
            selected.update(np.argsort(score)[-n_genes:].tolist())
    return np.array(sorted(selected), dtype=int)


def build_marker_guided_target(
    processed_dir: Path,
    cell_types: list[str],
    celltype_col: str,
    top_per_celltype: int,
    sum_weight: float,
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    sc_adata = ad.read_h5ad(processed_dir / "sc_reference.h5ad")
    st_adata = ad.read_h5ad(processed_dir / "simulated_ST_with_spatial.h5ad")

    st_gene_set = set(st_adata.var_names.astype(str))
    common_genes = [str(g) for g in sc_adata.var_names.astype(str) if str(g) in st_gene_set]
    if len(common_genes) < 10:
        raise ValueError(f"Too few common genes for marker guidance: {len(common_genes)}")
    sc_adata = sc_adata[:, common_genes].copy()
    st_adata = st_adata[:, common_genes].copy()

    sc_x = library_normalize(matrix_from_layer(sc_adata, "counts"))
    st_x = library_normalize(matrix_from_layer(st_adata, "counts"))
    labels = sc_adata.obs[celltype_col].astype(str).to_numpy()
    signature_all = np.vstack(
        [mean_rows(sc_x, labels == cell_type) for cell_type in cell_types]
    ).astype(np.float32)
    signature_all[signature_all < 0] = 0.0

    selected = select_marker_genes(signature_all, top_per_celltype)
    if selected.size < 10:
        raise ValueError(f"Too few selected marker genes: {selected.size}")

    signature = signature_all[:, selected].astype(np.float32)
    st_selected = st_x[:, selected]
    if sp.issparse(st_selected):
        st_selected = st_selected.toarray()
    st_selected = np.asarray(st_selected, dtype=np.float32)
    st_selected[st_selected < 0] = 0.0

    platform_scale = np.clip(
        st_selected.mean(axis=0) / (signature.mean(axis=0) + 1e-6), 0.1, 10.0
    )
    signature = signature * platform_scale.reshape(1, -1)
    gene_scale = np.sqrt((signature * signature).sum(axis=0))
    gene_scale[gene_scale < 1e-6] = 1.0
    signature = signature / gene_scale.reshape(1, -1)
    st_selected = st_selected / gene_scale.reshape(1, -1)

    n_celltypes = len(cell_types)
    design = np.vstack(
        [signature.T, np.sqrt(sum_weight) * np.ones((1, n_celltypes))]
    )
    estimates: list[np.ndarray] = []
    residuals: list[float] = []
    for expression in st_selected:
        target = np.concatenate(
            [expression, np.array([np.sqrt(sum_weight)], dtype=np.float32)]
        )
        solution, residual = nnls(design, target, maxiter=n_celltypes * 50)
        if solution.sum() <= 1e-12:
            solution[:] = 1.0 / n_celltypes
        else:
            solution = solution / solution.sum()
        estimates.append(solution)
        residuals.append(residual)

    marker_genes = [common_genes[i] for i in selected]
    target = pd.DataFrame(
        normalize_rows(np.vstack(estimates)),
        index=st_adata.obs_names.astype(str),
        columns=cell_types,
    )
    diagnostics = {
        "n_selected_marker_genes": int(selected.size),
        "marker_nnls_residual_mean": float(np.mean(residuals)),
    }
    return target, marker_genes, diagnostics
