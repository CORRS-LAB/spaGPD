#!/usr/bin/env bash
set -euo pipefail

python -m spagpd \
  --quick \
  --processed-dir data/Dataset1_simulated/processed \
  --output-dir results/Dataset1_simulated_quick \
  --method-name spaGPD
