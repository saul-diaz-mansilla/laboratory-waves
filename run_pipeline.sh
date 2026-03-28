#!/bin/bash

set -e

echo "Note: heavy simulations need to be downloaded or simulated."
echo "Phase 1: Processing experimental data"
python scripts/01_filter_experimental.py --config configs/experiment/gaussian_matched.yaml
python scripts/02_extract_features.py --config configs/experiment/gaussian_matched.yaml

python scripts/01_filter_experimental.py --config configs/experiment/sine_matched.yaml
python scripts/02_extract_features.py --config configs/experiment/sine_matched.yaml

python scripts/01_filter_experimental.py --config configs/experiment/gaussian_unmatched.yaml
python scripts/02_extract_features.py --config configs/experiment/gaussian_unmatched.yaml

echo "Phase 2: Simulating basic data"
python scripts/03_run_simulations.py --config configs/experiment/gaussian_matched_testing.yaml
python scripts/03_run_simulations.py --config configs/experiment/gaussian_matched_noscaling.yaml
python scripts/03_run_simulations.py --config configs/experiment/gaussian_unmatched_testing.yaml
python scripts/03_run_simulations.py --config configs/experiment/sine_matched_testing.yaml
python scripts/03_run_simulations.py --config configs/experiment/sine_matched_base.yaml

echo "Phase 3: Comparing data and generating plots"
python scripts/04_base_model_comparison.py --config configs/experiment/sine_matched_base.yaml
python scripts/05_dispersion_relation.py --config configs/experiment/sine_matched_testing.yaml
python scripts/06_scaling_comparison.py --config1 configs/experiment/gaussian_matched_noscaling.yaml --config2 configs/experiment/gaussian_matched_testing.yaml
python scripts/07_trend_comparison.py --config configs/experiment/gaussian_matched.yaml

echo "Phase 4: Train nn and infer data"

python scripts/08_train_model.py --config configs/experiment/gaussian_matched.yaml
python scripts/09_infer_parameters.py --config configs/experiment/gaussian_matched.yaml
