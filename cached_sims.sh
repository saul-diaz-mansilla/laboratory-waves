#!/bin/bash

set -e

echo "WARNING: This might take over 8 hours."
echo "Running lengthy simulations..."
python3 scripts/03_run_simulations.py --config configs/experiment/gaussian_matched.yaml
python3 scripts/03_run_simulations.py --config configs/experiment/sine_matched.yaml
