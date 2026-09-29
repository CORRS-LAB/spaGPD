"""Command-line interface for the final spaGPD pipeline.

Examples
--------
Run the final spaGPD model on the bundled Dataset1_simulated example::

    python -m spagpd --processed-dir data/Dataset1_simulated/processed \
        --output-dir results/Dataset1_simulated --method-name spaGPD
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Union

_PARSER_LIKE = Union[argparse.ArgumentParser, argparse._ArgumentGroup]

from spagpd.embeddings import REGISTRY
from spagpd.io.utils import set_runtime_cache, set_seed
from spagpd.pipeline.run import PipelineConfig, run_pipeline


def _add_shared_args(parser: _PARSER_LIKE) -> None:
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=root / "data" / "Dataset1_simulated" / "processed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "Dataset1_simulated",
    )
    parser.add_argument("--celltype-col", default="celltype")
    parser.add_argument(
        "--method-name",
        default=None,
        help="Prefix for output files (defaults to spaGPD).",
    )
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer epochs for a smoke test.")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--seed", type=int, default=11)


def _add_integration_args(parser: _PARSER_LIKE) -> None:
    parser.add_argument("--expr-pca-dim", type=int, default=30)
    parser.add_argument("--pca-weight", type=float, default=1.0)
    parser.add_argument("--pseudo-temperature", type=float, default=0.1)
    parser.add_argument("--deconv-hidden-dim", type=int, default=128)
    parser.add_argument("--deconv-epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--deconv-lr", type=float, default=None)
    parser.add_argument("--deconv-weight-decay", type=float, default=1e-5)
    parser.add_argument("--marker-top-per-celltype", type=int, default=80)
    parser.add_argument("--marker-sum-weight", type=float, default=10.0)
    parser.add_argument("--marker-weight", type=float, default=0.30)
    # KNN-graph size for the final GCN-based spaGPD encoder.
    parser.add_argument("--st-neighbors", type=int, default=10)
    parser.add_argument("--sc-neighbors", type=int, default=15)


def _add_gcn_clpls_args(parser: _PARSER_LIKE) -> None:
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=50)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--graph-epochs", type=int, default=400)


EMBEDDER_ARG_BUILDERS = {
    "gcn_clpls": _add_gcn_clpls_args,
}


def _gcn_clpls_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "hidden_channels": args.hidden_channels,
        "embedding_dim": args.embedding_dim,
        "beta": args.beta,
        "graph_epochs": args.graph_epochs,
        "lr": args.lr,
        "st_neighbors": args.st_neighbors,
        "sc_neighbors": args.sc_neighbors,
        "seed": args.seed,
    }


EMBEDDER_KWARG_BUILDERS = {
    "gcn_clpls": _gcn_clpls_kwargs,
}


def _apply_quick_defaults(embedder: str, args: argparse.Namespace) -> None:
    if embedder == "gcn_clpls":
        if args.graph_epochs is None or args.graph_epochs >= 400:
            args.graph_epochs = 20
    if args.deconv_epochs is None or args.deconv_epochs >= 250:
        args.deconv_epochs = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Final spaGPD pipeline for spatial cell-type proportion deconvolution."
        )
    )
    parser.add_argument(
        "--embedder",
        choices=sorted(REGISTRY),
        default="gcn_clpls",
        help="Graph-embedding encoder to use (default: gcn_clpls).",
    )
    _add_shared_args(parser)
    _add_integration_args(parser)
    # Each embedder's args are added under their own argparse group so the
    # --help output is organized without conflict on shared flag names.
    for name, fn in EMBEDDER_ARG_BUILDERS.items():
        group = parser.add_argument_group(
            f"{name} embedder options"
        )
        fn(group)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.quick:
        _apply_quick_defaults(args.embedder, args)
    set_runtime_cache()
    set_seed(args.seed)
    cfg = PipelineConfig(
        processed_dir=args.processed_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        embedder=args.embedder,
        celltype_col=args.celltype_col,
        method_name=args.method_name or "spaGPD",
        keep_intermediate=args.keep_intermediate,
        expr_pca_dim=args.expr_pca_dim,
        pca_weight=args.pca_weight,
        pseudo_temperature=args.pseudo_temperature,
        deconv_hidden_dim=args.deconv_hidden_dim,
        deconv_epochs=args.deconv_epochs,
        lr=args.lr,
        deconv_lr=args.deconv_lr,
        deconv_weight_decay=args.deconv_weight_decay,
        marker_top_per_celltype=args.marker_top_per_celltype,
        marker_sum_weight=args.marker_sum_weight,
        marker_weight=args.marker_weight,
        seed=args.seed,
        embedder_kwargs=EMBEDDER_KWARG_BUILDERS[args.embedder](args),
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
