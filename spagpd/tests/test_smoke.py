"""Smoke tests for the final spaGPD package.

Run with::

    .venv/bin/python -m pytest spagpd/tests/test_smoke.py -v

These tests exercise the final GCN-based spaGPD model end-to-end in
``--quick`` mode on the bundled ``Dataset1_simulated`` example.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "Dataset1_simulated" / "processed"

def _run_spagpd(tmp_out: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "spagpd",
        "--quick",
        "--processed-dir",
        str(PROCESSED),
        "--output-dir",
        str(tmp_out),
        "--method-name",
        "spaGPD",
        "--seed",
        "11",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"spaGPD smoke run failed (rc={result.returncode}).\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def test_spagpd_runs_and_writes_outputs(tmp_path: Path) -> None:
    _run_spagpd(tmp_path)
    prop = tmp_path / "spaGPD_prop.csv"
    metrics = tmp_path / "spaGPD_metrics.csv"
    diag = tmp_path / "spaGPD_run_diagnostics.csv"
    module = tmp_path / "spaGPD_module_record.json"
    for f in (prop, metrics, diag, module):
        assert f.exists(), f"Missing output {f}"
        assert f.stat().st_size > 0, f"Empty output {f}"


def test_metrics_have_expected_columns(tmp_path: Path) -> None:
    _run_spagpd(tmp_path)
    metrics_path = tmp_path / "spaGPD_metrics.csv"
    text = metrics_path.read_text().strip().splitlines()
    assert len(text) == 2, f"metrics should have a header + 1 row; got {text!r}"
    cols = text[0].split(",")
    expected = {
        "Spot Avg RMSE",
        "Spot Avg PCC",
        "Spot Avg SSIM",
        "Spot Avg JSD",
        "CellType Avg RMSE",
        "CellType Avg PCC",
        "CellType Avg SSIM",
        "CellType Avg JSD",
    }
    assert expected.issubset(set(cols)), (
        f"Missing metric columns: {expected - set(cols)}"
    )


def test_registry_lists_expected_embedders() -> None:
    from spagpd.embeddings import REGISTRY

    assert set(REGISTRY) == {"gcn_clpls"}


def test_gcn_clpls_has_reasonable_rmse(tmp_path: Path) -> None:
    """Loose sanity bound: --quick training on Dataset1_simulated should
    still produce a non-degenerate Spot RMSE (RMSE < 0.30)."""
    _run_spagpd(tmp_path)
    text = (tmp_path / "spaGPD_metrics.csv").read_text().strip().splitlines()
    header = text[0].split(",")
    row = text[1].split(",")
    idx = header.index("Spot Avg RMSE")
    rmse = float(row[idx])
    assert 0.0 < rmse < 0.30, f"Unexpected Spot RMSE: {rmse}"


@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_data() -> None:
    if not (PROCESSED / "cell_types.txt").exists():
        pytest.skip(f"Dataset not found at {PROCESSED}")
