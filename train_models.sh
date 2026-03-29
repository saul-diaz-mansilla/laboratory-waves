#!/bin/bash

set -e

echo "Training models..."
python scripts/08_train_model.py --config configs/experiment/gaussian_matched.yaml
python scripts/08_train_model.py --config configs/experiment/gaussian_unmatched.yaml
python scripts/08_train_model.py --config configs/experiment/gaussian_matched_simple.yaml
