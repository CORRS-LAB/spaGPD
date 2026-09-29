# spaGPD

spaGPD (Spatial Graph-Guided Proportion Deconvolution) is a Python package for cell-type proportion deconvolution in spatial transcriptomics. The model takes scRNA-seq reference data and spatial transcriptomics data as input, and outputs predicted proportions of different cell types in each spatial spot.

This repository contains the final spaGPD version used in the paper, including model code, example data, reproduction scripts, and running instructions.

## Method Overview

The main pipeline of spaGPD includes:

1. Perform shared gene matching and unified preprocessing for the scRNA-seq reference and ST data;
2. Construct a reference cell graph based on scRNA-seq expression similarity, and a spatial spot graph based on ST spatial coordinates;
3. Use a shared GCN to learn 50-dimensional graph-structure representations;
4. Concatenate the graph representations with 30-dimensional PCA expression features into 80-dimensional joint features;
5. Construct pseudo-proportion supervision based on reference cell labels;
6. Train a neural network deconvolution head to predict a spot-by-cell-type proportion matrix;
7. Use marker gene-guided calibration to calibrate the predictions.

The default parameters are the final model settings from the paper:

| Parameter | Default |
|---|---:|
| GCN embedding dimension | 50 |
| PCA dimension | 30 |
| Joint feature dimension | 80 |
| Graph training epochs | 400 |
| Deconvolution epochs | 300 |
| scRNA-seq kNN neighbors | 15 |
| ST spatial kNN neighbors | 10 |
| Pseudo-label temperature | 0.1 |
| Graph loss weight beta | 0.1 |
| Marker fusion weight | 0.3 |
| Random seed | 11 |

## File Structure

```text
spaGPD/
  spagpd/                  spaGPD Python package
    graph/                 graph construction and adjacency matrix normalization
    embeddings/            GCN graph representation learning module
    features/              PCA expression features
    pseudo/                pseudo-proportion supervision
    deconv/                neural network deconvolution module
    marker/                marker gene-guided calibration
    eval/                  RMSE, PCC, SSIM, JSD evaluation metrics
    pipeline/              complete running pipeline
    cli/                   command-line entry point
    tests/                 smoke test
  data/
    Dataset1_simulated/    directly runnable simulated example data; unzip `Dataset1_simulated.zip` first
  examples/                example running scripts
  results/                 output directory after running
  WALKTHROUGH.md           step-by-step reproduction instructions
  reproduce_example.sh     one-click example reproduction script
  pyproject.toml
  requirements.txt
```

## Installation

Python 3.10 or later is recommended.

```bash
cd spaGPD
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If you need to run tests, you can install the test dependencies:

```bash
python -m pip install -e ".[test]"
```

If you do not use editable install, you can also install dependencies according to `requirements.txt` and run directly from the repository root.

## Example Data

The repository provides a directly reproducible simulated dataset:

```text
# unzip `Dataset1_simulated.zip` first
cd data
unzip Dataset1_simulated.zip
data/Dataset1_simulated/processed/
```

This dataset is a mouse cortex-like simulated spatial transcriptomics dataset with known ground-truth spot-level cell-type proportions. It can be used to check spaGPD's complete running pipeline and simulated-data evaluation metrics. The data scale is as follows:

| Item | Count |
|---|---:|
| Spatial spots | 490 |
| Reference cells | 10,266 |
| Shared genes | 320 |
| Cell types | 7 |

This directory contains:

| File | Description |
|---|---|
| `sc_reference.h5ad` | scRNA-seq reference count matrix and cell type labels |
| `simulated_ST_with_spatial.h5ad` | ST expression matrix and spatial coordinates |
| `cell_types.txt` | Cell type order |
| `spatial_coordinates.csv` | Spot spatial coordinates |
| `true_proportions.csv` | Ground-truth spot-by-cell-type proportion matrix for simulated data |

For new simulated datasets, they also need to be organized into the same `processed/` format. If only real-data prediction is performed and simulated-data metrics are not calculated, `true_proportions.csv` is not required; however, to reproduce RMSE, PCC, SSIM, and JSD, the ground-truth proportion matrix must be provided.

## Quick Test

The quick test reduces training epochs and is used to check whether the environment, data, and code can run through:

```bash
python -m spagpd \
  --quick \
  --processed-dir data/Dataset1_simulated/processed \
  --output-dir results/Dataset1_simulated_quick \
  --method-name spaGPD
```

## Reproduction Example

Run with the final parameter settings from the paper:

```bash
python -m spagpd \
  --processed-dir data/Dataset1_simulated/processed \
  --output-dir results/Dataset1_simulated \
  --method-name spaGPD
```

You can also run directly:

```bash
bash reproduce_example.sh
```

## Output Files

After running, the following files will be generated in the output directory:

| File | Description |
|---|---|
| `spaGPD_prop.csv` | Final spot-by-cell-type predicted proportion matrix |
| `spaGPD_metrics.csv` | RMSE, PCC, SSIM, JSD metrics on simulated data |
| `spaGPD_run_diagnostics.csv` | Running parameters and diagnostic information |
| `spaGPD_module_record.json` | Model module records and marker gene information |

## Running Tests

```bash
python -m pytest spagpd/tests/test_smoke.py -v
```

The test runs quick mode on the example data and checks whether the prediction matrix and evaluation metric files are generated.

## Results Summary

spaGPD combines expression features, spatial graph structure, pseudo-proportion supervision constructed from reference cell labels, and marker gene-guided calibration. In simulated data, the model is used to evaluate proportion prediction accuracy at the spot level and cell type level; in real data, model predictions can be further combined with expert annotations or pathological region annotations for spatial enrichment region validation.

For more running steps, see `WALKTHROUGH.md`.
