#!/usr/bin/env bash
set -euo pipefail

python -m spagpd \
  --processed-dir data/Dataset1_simulated/processed \
  --output-dir results/Dataset1_simulated \
  --method-name spaGPD
