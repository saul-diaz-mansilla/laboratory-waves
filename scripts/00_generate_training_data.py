import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.electrical.simulation import simulate
import src.utils.io as io


def main():
    # Set up argparse
    parser = argparse.ArgumentParser(description="Simulate transmission line")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the master simulation configuration file in YAML format",
    )

    args = parser.parse_args()
    config_path = args.config
    config_parameters = io.load_config(config_path)
    output_dir = config_parameters["paths"]["simulation_dir"]

    # Run simulation
    all_targets, all_results, all_freqs = simulate(config_path)

    # Save data to PARQUET files
    io.save_parquet(all_targets, output_dir, name="targets_")
    io.save_parquet(all_results, output_dir, name="results_")
    io.save_parquet(all_freqs, output_dir, name="freqs_")


if __name__ == "__main__":
    main()
