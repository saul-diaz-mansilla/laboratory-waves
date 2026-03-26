#!/bin/bash

set -e

echo "Phase 1: Processing experimental data"
python3 scripts/01_filter_experimental.py --config configs/experiment/gaussian_matched.yaml
python3 scripts/02_extract_features.py --config configs/experiment/gaussian_matched.yaml

python3 scripts/01_filter_experimental.py --config configs/experiment/sine_matched.yaml
python3 scripts/02_extract_features.py --config configs/experiment/sine_matched.yaml

python3 scripts/01_filter_experimental.py --config configs/experiment/gaussian_unmatched.yaml
python3 scripts/02_extract_features.py --config configs/experiment/gaussian_unmatched.yaml

echo "Phase 2: Simulating basic data"
python3 scripts/03_run_simulations.py --config configs/experiment/gaussian_matched_testing.yaml
python3 scripts/03_run_simulations.py --config configs/experiment/gaussian_matched_noscaling.yaml
python3 scripts/03_run_simulations.py --config configs/experiment/gaussian_unmatched_testing.yaml
python3 scripts/03_run_simulations.py --config configs/experiment/sine_matched_base.yaml

echo "Phase 3: Comparing data and generating plots"
python3 scripts/04_base_model_comparison.py --config configs/experiment/sine_matched_base.yaml
python3 scripts/05_dispersion_relation.py --config configs/experiment/gaussian_unmatched_testing.yaml
python3 scripts/06_scaling_comparison.py --config1 configs/experiment/gaussian_matched_noscaling.yaml --config2 configs/experiment/gaussian_matched_testing.yaml
