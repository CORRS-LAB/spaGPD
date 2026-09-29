"""Base interface for graph-embedding encoders and the registry that wires them.

Each embedder takes aligned (st_expr, sc_expr) plus pre-built graphs and returns
two float32 numpy arrays (emb_st, emb_sc) suitable for downstream integration.
The class is free to perform its own graph construction if it needs to.

To add a new encoder:
  1. Subclass ``BaseEmbedder`` and implement ``fit_transform``.
  2. Decorate it with ``@register_embedder("my_name")`` or append to ``REGISTRY``.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class EmbedderContext:
    """All the inputs an embedder may consume.

    ``st_graph`` / ``sc_graph`` are scipy COO matrices. ``st_edge_index`` /
    ``sc_edge_index`` are torch.long 2xE tensors (bidirected, self-looped).
    Either representation is fine, depending on the encoder's preference.
    """

    st_expr: np.ndarray  # (n_spots, n_genes)
    sc_expr: np.ndarray  # (n_sc, n_genes)
    st_graph: sp.spmatrix | None = None
    sc_graph: sp.spmatrix | None = None
    st_edge_index: torch.Tensor | None = None
    sc_edge_index: torch.Tensor | None = None
    st_coords: np.ndarray | None = None  # raw 2D spatial coordinates
    sc_labels: np.ndarray | None = None  # int-encoded cell-type labels


@dataclass
class EmbedderOutput:
    """Output of an embedder.

    ``emb_st`` / ``emb_sc``: float32, shape (n, d) — the cell/spot embeddings.
    ``aux_st`` / ``aux_sc``: optional float32 features to concatenate
        before pseudo-labeling.
    ``module_record``: human-readable description of what the embedder does.
    ``diagnostics``: scalar diagnostics to surface in the run_diagnostics CSV.
    """

    emb_st: np.ndarray
    emb_sc: np.ndarray
    aux_st: np.ndarray | None = None
    aux_sc: np.ndarray | None = None
    module_record: list[str] | None = None
    diagnostics: dict[str, Any] | None = None


class BaseEmbedder(abc.ABC):
    """Abstract base class for all graph-embedding encoders."""

    #: short identifier used on the CLI (``--embedder <name>``)
    name: str = ""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @abc.abstractmethod
    def fit_transform(self, ctx: EmbedderContext) -> EmbedderOutput:
        """Train (if needed) and return embeddings."""


REGISTRY: dict[str, type[BaseEmbedder]] = {}


def register_embedder(name: str):
    """Class decorator to register an embedder under a CLI-friendly name."""

    def _wrap(cls: type[BaseEmbedder]) -> type[BaseEmbedder]:
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return _wrap


def get_embedder(name: str, **kwargs: Any) -> BaseEmbedder:
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown embedder '{name}'. Registered: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](**kwargs)
