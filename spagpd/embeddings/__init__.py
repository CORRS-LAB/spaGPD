"""Embedder registry. Importing the package triggers registration."""
from __future__ import annotations

from spagpd.embeddings.base import (
    REGISTRY,
    BaseEmbedder,
    EmbedderContext,
    EmbedderOutput,
    get_embedder,
    register_embedder,
)
from spagpd.embeddings.gcn_clpls import GCNCLPLSEembedder

__all__ = [
    "REGISTRY",
    "BaseEmbedder",
    "EmbedderContext",
    "EmbedderOutput",
    "GCNCLPLSEembedder",
    "get_embedder",
    "register_embedder",
]
