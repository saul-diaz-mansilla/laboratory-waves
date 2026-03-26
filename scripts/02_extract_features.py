import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.electrical.signals as signals
import src.utils.io as io


def main():
    # Set up argparse
    parser = argparse.ArgumentParser(
        description="Extract and save experimental transfer functions."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the master experimental configuration file. Located in configs/experiment",
    )

    args = parser.parse_args()
    config_path = args.config

    exp_parameters = io.load_config(config_path)
    # Path management
    exp_dir = exp_parameters["paths"]["processed_dir"]
    metadata_path = exp_parameters["paths"]["metadata"]
    waveform = exp_parameters["input"]["waveform"]

    # Input signal parameters from experimental metadata
    df_metadata = pd.read_csv(metadata_path)

    all_results = {}
    all_freqs_data = {}

    # Group by node to process data for each node separately
    for node_value, node_group in df_metadata.groupby("node"):
        all_v0_node = []
        all_vnode_node = []

        # Extract filenames relevant to the current node
        files_node = io.get_processed_filename(node_group["filename"].to_numpy())

        t_node = None

        # Loop over frequencies
        for file_path_processed in files_node:
            full_file_path = Path(exp_dir) / file_path_processed
            data_exp = pd.read_parquet(full_file_path)

            if t_node is None:
                t_node = data_exp.iloc[:, 0].to_numpy()

            v0_raw = data_exp.iloc[:, 1].to_numpy()
            vnode_raw = data_exp.iloc[:, 2].to_numpy()

            all_v0_node.append(v0_raw)
            all_vnode_node.append(vnode_raw)

        # Calculate transfer function for the current node
        if waveform == "gaussian":
            # Awg parameters and sigma should be constant for same experiment
            awg_frequency_hz = df_metadata["awg_frequency_hz"].iloc[0]
            std_percent = df_metadata["std_percent"].iloc[0]
            frequencies_node = node_group["frequency_hz"].to_numpy()

            duration = 1 / awg_frequency_hz
            sigma = std_percent / 100 * duration

            H_node, freqs_global_node = signals.H_gaussian(
                all_v0_node,
                all_vnode_node,
                t_node,
                frequencies_node,
                sigma,
            )
        elif waveform == "sine":
            frequencies_node = node_group["frequency_hz"].to_numpy()

            H_node, freqs_global_node = signals.H_sine(
                all_v0_node,
                all_vnode_node,
                t_node,
                frequencies_node,
            )
        elif waveform == "pulse":
            # Awg parameters and duty_cycle should be constant for same experiment
            awg_frequency_hz = df_metadata["awg_frequency_hz"].iloc[0]
            duty_cycle = df_metadata["duty_cycle_percent"].iloc[0]

            duration = 1 / awg_frequency_hz
            pulse_width = duty_cycle / 100 * duration

            H_node, freqs_global_node = signals.H_sine(
                all_v0_node, all_vnode_node, t_node, pulse_width
            )

        # Store magnitude and phase for the current node
        all_results[f"H_Mag_{node_value}"] = [np.abs(H_node).tolist()]
        all_results[f"H_Phase_{node_value}"] = [np.angle(H_node).tolist()]

        # Store frequencies. Assuming f_exp is consistent across nodes, store it once.
        if not all_freqs_data:
            all_freqs_data["freqs_global"] = [freqs_global_node.tolist()]

    # Save all collected results and frequencies after processing all nodes
    io.save_parquet(all_results, exp_dir, prefix="results_")
    io.save_parquet(all_freqs_data, exp_dir, prefix="freqs_")


if __name__ == "__main__":
    main()
