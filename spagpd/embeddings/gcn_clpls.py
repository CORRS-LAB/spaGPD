"""GCN encoder with the CLPLS (dual-view contrastive + classifier) auxiliary loss.

Faithful re-implementation of the encoder from the original
``run_spaGPD_demo.py``. The encoder's behaviour and exact defaults are preserved.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from spagpd.embeddings.base import (
    BaseEmbedder,
    EmbedderContext,
    EmbedderOutput,
    register_embedder,
)
from spagpd.graph.builder import (
    build_knn_adj,
    normalize_adjacency,
    scipy_to_torch_sparse,
)
from spagpd.io.utils import to_dense_float32, get_device


class DenseGCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adj_norm, x @ self.weight) + self.bias


class GraphEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_channels: int, out_features: int):
        super().__init__()
        self.conv1 = DenseGCNLayer(in_features, hidden_channels)
        self.prelu1 = nn.PReLU(hidden_channels)
        self.conv2 = DenseGCNLayer(hidden_channels, out_features)
        self.prelu2 = nn.PReLU(out_features)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        x = self.prelu1(self.conv1(x, adj_norm))
        x = self.prelu2(self.conv2(x, adj_norm))
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Bilinear(hidden_dim, hidden_dim, 1)
        nn.init.xavier_uniform_(self.score.weight.data)
        if self.score.bias is not None:
            self.score.bias.data.fill_(0.0)

    def forward(
        self, summary: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        summary = summary.expand_as(positive)
        pos_score = self.score(positive, summary)
        neg_score = self.score(negative, summary)
        return torch.cat((pos_score, neg_score), dim=1)


class AvgReadout(nn.Module):
    def forward(self, embedding: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value_sum = torch.mm(mask, embedding)
        row_sum = torch.sum(mask, dim=1).clamp_min(1e-8)
        return F.normalize(value_sum / row_sum[:, None], p=2, dim=1)


class CLPLSGraphEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_channels: int,
        out_features: int,
        graph_neigh: torch.Tensor,
    ):
        super().__init__()
        self.graph_neigh = graph_neigh
        self.graph_encoder = GraphEncoder(in_features, hidden_channels, out_features)
        self.discriminator = Discriminator(out_features)
        self.readout = AvgReadout()

    def forward(
        self,
        expr: torch.Tensor,
        expr_aug: torch.Tensor | None,
        adj_norm: torch.Tensor,
    ):
        z = self.graph_encoder(expr, adj_norm)
        if expr_aug is None:
            return z
        z_aug = self.graph_encoder(expr_aug, adj_norm)
        summary = torch.sigmoid(self.readout(z, self.graph_neigh.to(z.device)))
        summary_aug = torch.sigmoid(self.readout(z_aug, self.graph_neigh.to(z.device)))
        ret = self.discriminator(summary, z, z_aug)
        ret_aug = self.discriminator(summary_aug, z_aug, z)
        return z, ret, ret_aug


class CellTypeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


def _corruption(x: torch.Tensor) -> torch.Tensor:
    return x[torch.randperm(x.size(0))]


def _add_contrastive_label(n_spot: int) -> torch.Tensor:
    return torch.cat(
        [torch.ones((n_spot, 1)), torch.zeros((n_spot, 1))], dim=1
    ).float()


def _process_st_graph(
    st_expr: np.ndarray, st_coords: np.ndarray, n_neighbors: int
) -> tuple[sp.coo_matrix, torch.Tensor, torch.Tensor]:
    spatial_adj = build_knn_adj(st_coords, n_neighbors=n_neighbors)
    graph_neigh = torch.FloatTensor(
        spatial_adj.copy().toarray() + np.eye(spatial_adj.shape[0])
    )
    label_csl = _add_contrastive_label(st_expr.shape[0])
    return spatial_adj, graph_neigh, label_csl


def _process_sc_graph(
    sc_expr: np.ndarray,
    sc_labels: np.ndarray,
    n_neighbors: int,
) -> sp.coo_matrix:
    """Build a KNN graph and keep only same-cell-type edges (matches original)."""
    from spagpd.features.pca import pca_scores

    n_pcs = min(30, sc_expr.shape[0] - 1, sc_expr.shape[1] - 1)
    if n_pcs < 2:
        raise ValueError(
            f"scRNA data are too small for graph construction: n_pcs={n_pcs}"
        )
    x_pca = pca_scores(sc_expr, n_pcs)
    adj = build_knn_adj(x_pca, n_neighbors=n_neighbors).tocoo()
    keep = np.array(
        [sc_labels[int(row)] == sc_labels[int(col)] for row, col in zip(adj.row, adj.col)],
        dtype=bool,
    )
    return sp.coo_matrix(
        (adj.data[keep], (adj.row[keep], adj.col[keep])),
        shape=adj.shape,
        dtype=np.float32,
    )


@register_embedder("gcn_clpls")
class GCNCLPLSEembedder(BaseEmbedder):
    """Two-layer GCN + CLPLS dual-view contrastive + cell-type MLP classifier.

    Hyper-parameters
    ----------------
    hidden_channels : int (default 128)
    embedding_dim   : int (default 50)
    beta            : float (default 0.1) - weight of contrastive loss
    graph_epochs    : int  (default 400; ``--quick`` uses 20)
    lr              : float (default 1e-3)
    st_neighbors    : int  (default 10)
    sc_neighbors    : int  (default 15)
    seed            : int  (default 11)
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        embedding_dim: int = 50,
        beta: float = 0.1,
        graph_epochs: int = 400,
        lr: float = 1e-3,
        st_neighbors: int = 10,
        sc_neighbors: int = 15,
        seed: int = 11,
        **kwargs,
    ) -> None:
        super().__init__(
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            beta=beta,
            graph_epochs=graph_epochs,
            lr=lr,
            st_neighbors=st_neighbors,
            sc_neighbors=sc_neighbors,
            seed=seed,
            **kwargs,
        )
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.graph_epochs = graph_epochs
        self.lr = lr
        self.st_neighbors = st_neighbors
        self.sc_neighbors = sc_neighbors
        self.seed = seed

    def fit_transform(self, ctx: EmbedderContext) -> EmbedderOutput:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = get_device()

        st_expr = ctx.st_expr
        sc_expr = ctx.sc_expr
        st_coords = ctx.st_coords
        sc_labels = ctx.sc_labels

        spatial_adj, graph_neigh, label_csl = _process_st_graph(
            st_expr, st_coords, self.st_neighbors
        )
        sc_adj = _process_sc_graph(sc_expr, sc_labels, self.sc_neighbors)

        expr_st = torch.tensor(
            to_dense_float32(st_expr), dtype=torch.float32, device=device
        )
        expr_sc = torch.tensor(
            to_dense_float32(sc_expr), dtype=torch.float32, device=device
        )
        adj_st = scipy_to_torch_sparse(normalize_adjacency(spatial_adj)).to(device)
        adj_sc = scipy_to_torch_sparse(normalize_adjacency(sc_adj)).to(device)
        graph_neigh = graph_neigh.to(device)
        ctype_lab = torch.as_tensor(sc_labels, dtype=torch.long, device=device)
        label_csl = label_csl.to(device)

        encoder = CLPLSGraphEncoder(
            st_expr.shape[1], self.hidden_channels, self.embedding_dim, graph_neigh
        ).to(device)
        classifier = CellTypeMLP(
            self.embedding_dim, 25, int(ctype_lab.max().item()) + 1
        ).to(device)
        graph_loss = nn.BCEWithLogitsLoss()
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=self.lr)
        classifier_optimizer = torch.optim.Adam(
            classifier.parameters(), lr=5e-4, weight_decay=5e-4
        )

        print(
            f"[gcn_clpls] Train graph encoder: epochs={self.graph_epochs}, device={device}",
            flush=True,
        )
        for epoch in range(self.graph_epochs):
            encoder.train()
            classifier.train()
            expr_aug = _corruption(expr_st)
            _, ret, _ = encoder(expr_st, expr_aug, adj_st)
            z_sc = encoder(expr_sc, None, adj_sc)
            output = classifier(z_sc)
            loss_class = F.nll_loss(output, ctype_lab)
            loss_graph = graph_loss(ret, label_csl)
            loss = loss_class + self.beta * loss_graph

            encoder_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            classifier_optimizer.step()

            if epoch % 100 == 0 or epoch == self.graph_epochs - 1:
                print(
                    f"[gcn_clpls] Graph epoch={epoch:04d} loss={loss.item():.6f} "
                    f"class={loss_class.item():.6f} graph={loss_graph.item():.6f}",
                    flush=True,
                )

        encoder.eval()
        with torch.no_grad():
            embedding_st = encoder(expr_st, None, adj_st).cpu().numpy()
            embedding_sc = encoder(expr_sc, None, adj_sc).cpu().numpy()

        return EmbedderOutput(
            emb_st=embedding_st.astype(np.float32),
            emb_sc=embedding_sc.astype(np.float32),
            module_record=[
                "GCN graph encoder (2 DenseGCN layers + PReLU)",
                "CLPLS contrastive dual-view + cell-type classifier auxiliary loss",
            ],
            diagnostics={
                "gcn_embedding_dim": int(self.embedding_dim),
                "st_graph_edges": int(spatial_adj.nnz),
                "sc_graph_edges_after_label_filter": int(sc_adj.nnz),
            },
        )
