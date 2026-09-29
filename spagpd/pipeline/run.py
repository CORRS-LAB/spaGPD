"""End-to-end pipeline: load data -> embedder -> integrate -> pseudo -> deconv -> marker fusion -> evaluate."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch

from spagpd.embeddings import EmbedderContext, EmbedderOutput, get_embedder
from spagpd.eval.metrics import evaluate_prediction
from spagpd.features.pca import build_expression_pca_features, normalize_feature
from spagpd.io.utils import (
    normalize_rows,
    read_cell_types,
    set_runtime_cache,
    standardize_prediction,
    to_dense_float32,
    use_lognorm_as_x,
)
from spagpd.marker.guided import build_marker_guided_target
from spagpd.pseudo.pseudo import build_pseudo_proportions


@dataclass
class PipelineConfig:
    processed_dir: Path
    output_dir: Path
    embedder: str = "gcn_clpls"
    celltype_col: str = "celltype"
    method_name: str = "spaGPD"
    keep_intermediate: bool = False

    # Integration / deconvolution / marker-fusion
    expr_pca_dim: int = 30
    pca_weight: float = 1.0
    pseudo_temperature: float = 0.1
    deconv_hidden_dim: int = 128
    deconv_epochs: int = 300
    lr: float = 1e-3
    deconv_lr: float | None = None  # if set, overrides lr for deconv head
    deconv_weight_decay: float = 1e-5
    marker_top_per_celltype: int = 80
    marker_sum_weight: float = 10.0
    marker_weight: float = 0.30
    seed: int = 11

    # Embedder-specific kwargs forwarded to the embedder constructor
    embedder_kwargs: dict[str, Any] = field(default_factory=dict)


def _load_inputs(
    processed_dir: Path,
    cell_types: list[str],
    celltype_col: str,
) -> tuple[ad.AnnData, ad.AnnData, np.ndarray, np.ndarray, np.ndarray]:
    sc_adata = ad.read_h5ad(processed_dir / "sc_reference.h5ad")
    st_adata = ad.read_h5ad(processed_dir / "simulated_ST_with_spatial.h5ad")
    sc_adata.obs_names = sc_adata.obs_names.astype(str)
    st_adata.obs_names = st_adata.obs_names.astype(str)

    if celltype_col not in sc_adata.obs.columns:
        raise ValueError(f"Missing sc_reference.obs['{celltype_col}']")
    if "spatial" not in st_adata.obsm:
        raise ValueError("simulated_ST_with_spatial.h5ad is missing obsm['spatial']")

    sc_adata = use_lognorm_as_x(sc_adata)
    st_adata = use_lognorm_as_x(st_adata)
    sc_adata.obs[celltype_col] = sc_adata.obs[celltype_col].astype(str)
    sc_adata = sc_adata[sc_adata.obs[celltype_col].isin(cell_types)].copy()

    common_genes = sc_adata.var_names.intersection(st_adata.var_names)
    if len(common_genes) == 0:
        raise ValueError("No common genes between scRNA and ST data")
    sc_adata = sc_adata[:, common_genes].copy()
    st_adata = st_adata[:, common_genes].copy()

    st_expr = to_dense_float32(st_adata.X)
    sc_expr = to_dense_float32(sc_adata.X)
    st_coords = np.asarray(st_adata.obsm["spatial"], dtype=np.float32)
    labels_raw = sc_adata.obs[celltype_col].astype(str).to_numpy()
    sc_labels = np.array(
        [cell_types.index(label) for label in labels_raw], dtype=np.int64
    )
    return sc_adata, st_adata, st_expr, sc_expr, st_coords, sc_labels, common_genes


def _integrate_features(
    embedder_out: EmbedderOutput,
    sc_adata,
    st_adata,
    pca_dim: int,
    pca_weight: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the embedder's aux features (if any) to the embedding.

    For encoders that don't return aux (e.g. ``gcn_clpls``), we concatenate a
    PCA block to mimic the original spaGPD integration step.
    """
    if embedder_out.aux_st is not None and embedder_out.aux_sc is not None:
        feature_sc = np.hstack(
            [normalize_feature(embedder_out.emb_sc), embedder_out.aux_sc]
        ).astype(np.float32)
        feature_st = np.hstack(
            [normalize_feature(embedder_out.emb_st), embedder_out.aux_st]
        ).astype(np.float32)
        return feature_sc, feature_st

    pca_sc, pca_st = build_expression_pca_features(
        sc_adata, st_adata, pca_dim, seed
    )
    feature_sc = np.concatenate(
        [
            normalize_feature(embedder_out.emb_sc),
            normalize_feature(pca_sc) * pca_weight,
        ],
        axis=1,
    ).astype(np.float32)
    feature_st = np.concatenate(
        [
            normalize_feature(embedder_out.emb_st),
            normalize_feature(pca_st) * pca_weight,
        ],
        axis=1,
    ).astype(np.float32)
    return feature_sc, feature_st


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    from spagpd.deconv.head import train_deconv_head

    set_runtime_cache()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cell_types = read_cell_types(cfg.processed_dir / "cell_types.txt")
    sc_adata, st_adata, st_expr, sc_expr, st_coords, sc_labels, common_genes = (
        _load_inputs(cfg.processed_dir, cell_types, cfg.celltype_col)
    )

    embedder = get_embedder(cfg.embedder, **cfg.embedder_kwargs)
    out = embedder.fit_transform(
        EmbedderContext(
            st_expr=st_expr,
            sc_expr=sc_expr,
            st_coords=st_coords,
            sc_labels=sc_labels,
        )
    )

    feature_sc, feature_st = _integrate_features(
        out, sc_adata, st_adata, cfg.expr_pca_dim, cfg.pca_weight, cfg.seed
    )
    pseudo_label = build_pseudo_proportions(
        feature_st=feature_st,
        feature_sc=feature_sc,
        sc_adata=sc_adata,
        celltype_col=cfg.celltype_col,
        cell_types=cell_types,
        temperature=cfg.pseudo_temperature,
    )
    neural_values = train_deconv_head(
        feature_st=feature_st,
        pseudo_label=pseudo_label,
        hidden_dim=cfg.deconv_hidden_dim,
        epochs=cfg.deconv_epochs,
        lr=cfg.deconv_lr if cfg.deconv_lr is not None else cfg.lr,
        weight_decay=cfg.deconv_weight_decay,
        log_prefix=f"[{cfg.embedder}]",
    )
    neural_prediction = pd.DataFrame(
        neural_values,
        index=st_adata.obs_names.astype(str),
        columns=cell_types,
    )
    neural_prediction = standardize_prediction(
        neural_prediction, st_adata.obs_names.astype(str).tolist(), cell_types
    )

    marker_target, marker_genes, marker_diag = build_marker_guided_target(
        processed_dir=cfg.processed_dir,
        cell_types=cell_types,
        celltype_col=cfg.celltype_col,
        top_per_celltype=cfg.marker_top_per_celltype,
        sum_weight=cfg.marker_sum_weight,
    )
    marker_target = marker_target.loc[neural_prediction.index, cell_types]
    final_values = normalize_rows(
        (1.0 - cfg.marker_weight) * neural_prediction.values
        + cfg.marker_weight * marker_target.values
    )
    final_prediction = pd.DataFrame(
        final_values, index=neural_prediction.index, columns=cell_types
    )

    prop_path = cfg.output_dir / f"{cfg.method_name}_prop.csv"
    metrics_path = cfg.output_dir / f"{cfg.method_name}_metrics.csv"
    diagnostics_path = cfg.output_dir / f"{cfg.method_name}_run_diagnostics.csv"
    module_path = cfg.output_dir / f"{cfg.method_name}_module_record.json"
    final_prediction.to_csv(prop_path)

    metrics = evaluate_prediction(final_prediction, cfg.processed_dir, cell_types)
    metrics.to_csv(metrics_path, index=False)
    print(f"[{cfg.embedder}] Evaluation metrics:", flush=True)
    print(metrics.to_string(index=False), flush=True)

    embedder_diag = out.diagnostics or {}
    diagnostics = {
        "n_spots": int(st_adata.n_obs),
        "n_sc_cells": int(sc_adata.n_obs),
        "n_common_genes": int(len(common_genes)),
        "embedder": cfg.embedder,
        "integrated_feature_dim": int(feature_st.shape[1]),
        "marker_weight": float(cfg.marker_weight),
        "method_name": cfg.method_name,
        **marker_diag,
        **embedder_diag,
    }
    pd.DataFrame([diagnostics]).to_csv(diagnostics_path, index=False)

    module_record = {
        "modules": (out.module_record or [])
        + [
            "expression PCA / wide-feature integration",
            "pseudo-proportion supervision",
            "neural-network deconvolution",
            "marker-gene-guided NNLS fusion",
            "RMSE/PCC/SSIM/JSD evaluation",
        ],
        "diagnostics": diagnostics,
        "marker_genes": marker_genes,
    }
    module_path.write_text(json.dumps(module_record, indent=2, ensure_ascii=False))

    if cfg.keep_intermediate:
        inter_dir = cfg.output_dir / "intermediate"
        inter_dir.mkdir(parents=True, exist_ok=True)
        neural_prediction.to_csv(
            inter_dir / f"{cfg.method_name}_neural_prediction.csv"
        )
        marker_target.to_csv(
            inter_dir / f"{cfg.method_name}_marker_guided_target.csv"
        )
        pd.DataFrame(
            pseudo_label,
            index=st_adata.obs_names.astype(str),
            columns=cell_types,
        ).to_csv(inter_dir / f"{cfg.method_name}_pseudo_label.csv")
        pd.DataFrame(
            out.emb_st, index=st_adata.obs_names.astype(str)
        ).to_csv(inter_dir / f"{cfg.method_name}_embedding_st.csv")
        pd.DataFrame(
            out.emb_sc, index=sc_adata.obs_names.astype(str)
        ).to_csv(inter_dir / f"{cfg.method_name}_embedding_sc.csv")
        pd.DataFrame(
            feature_st, index=st_adata.obs_names.astype(str)
        ).to_csv(inter_dir / f"{cfg.method_name}_feature_st.csv")
        pd.DataFrame(
            feature_sc, index=sc_adata.obs_names.astype(str)
        ).to_csv(inter_dir / f"{cfg.method_name}_feature_sc.csv")

    print(f"[{cfg.embedder}] Final prediction: {prop_path}", flush=True)
    print(f"[{cfg.embedder}] Metrics: {metrics_path}", flush=True)
    return {
        "metrics": metrics,
        "diagnostics": diagnostics,
        "module_record": module_record,
    }
