import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.electrical.signals as signals
import src.utils.io as io
import src.utils.visualization as vis


def main():
    # Set up argparse
    parser = argparse.ArgumentParser(
        description="Plot experimental vs simulated transfer function"
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
    sim_dir = exp_parameters["paths"]["simulation_dir"]
    circuit_parameters = io.load_config(exp_parameters["paths"]["circuit_config"])
    f_cutoff = (
        1
        / np.sqrt(circuit_parameters["L"]["value"] * circuit_parameters["C"]["value"])
        / np.pi
    )

    # Input signal
    f_start = exp_parameters["input"]["f_start"]
    f_end = exp_parameters["input"]["f_end"]
    num_inputs = exp_parameters["input"]["num_inputs"]
    sigma = exp_parameters["input"]["sigma"]
    frequencies = np.linspace(f_start, f_end, num_inputs)

    exp_v0_all = []
    exp_v40_all = []

    all_exp_files = list(Path(exp_dir).glob("*.parquet"))

    for file_path in all_exp_files:
        data_exp = pd.read_parquet(file_path)
        t_raw = data_exp.iloc[:, 0].to_numpy()
        v0_raw = data_exp.iloc[:, 1].to_numpy()
        v40_raw = data_exp.iloc[:, 2].to_numpy()

        exp_v0_all.append(v0_raw)
        exp_v40_all.append(v40_raw)

    H_40_exp, f_exp = signals.H_gaussian(
        exp_v0_all, exp_v40_all, t_raw, frequencies, sigma
    )
    H_40_exp = np.abs(H_40_exp)

    df_results = io.load_parquet_data(sim_dir, prefix="results_")
    df_axes = io.load_parquet_data(sim_dir, prefix="freqs_")

    f_sim = df_axes["freqs_global"].iloc[0]

    sim_index = 0
    H_40_sim = df_results["H_Mag_40"].iloc[sim_index]

    fig, ax1 = plt.subplots(**vis.apply_standard_style(1, 1))
    vis.plot_style(
        ax1,
        f_sim,
        H_40_sim,
        f_exp,
        H_40_exp,
        vline_x=f_cutoff,
        vline_label=r"Cut-off frequency $f_c$",
    )
    vis.axes_transfer_function(ax1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
