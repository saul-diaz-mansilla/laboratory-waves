import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.utils.io as io
import src.utils.visualization as vis


def main():
    # Set up argparse
    parser = argparse.ArgumentParser(
        description="Plot experimental vs simulated transfer function"
    )

    parser.add_argument(
        "--config1",
        type=str,
        required=True,
        help="Path to the first master experimental configuration file. Located in configs/experiment",
    )
    parser.add_argument(
        "--config2",
        type=str,
        required=True,
        help="Path to the second master experimental configuration file. Located in configs/experiment",
    )

    args = parser.parse_args()
    config_path = args.config1
    compare_path = args.config2

    exp_parameters = io.load_config(config_path)
    compare_parameters = io.load_config(compare_path)

    # Path management
    exp_dir = exp_parameters["paths"]["processed_dir"]
    sim_dir = exp_parameters["paths"]["simulation_dir"]
    comp_dir = compare_parameters["paths"]["simulation_dir"]

    # Obtain cutoff frequency for plotting
    circuit_parameters = io.load_config(exp_parameters["paths"]["circuit_config"])
    f_cutoff = (
        1
        / np.sqrt(circuit_parameters["L"]["value"] * circuit_parameters["C"]["value"])
        / np.pi
    )

    # Find experimental data
    df_results = io.load_parquet_data(exp_dir, prefix="results_")
    df_axes = io.load_parquet_data(exp_dir, prefix="freqs_")

    f_exp = df_axes["freqs_global"].iloc[0]

    sim_index = 0
    H_40_exp = df_results["H_Mag_40"].iloc[sim_index]

    # Find simulated data
    df_results = io.load_parquet_data(sim_dir, prefix="results_")
    df_axes = io.load_parquet_data(sim_dir, prefix="freqs_")

    f_sim = df_axes["freqs_global"].iloc[0]

    H_40_sim = df_results["H_Mag_40"].iloc[sim_index]

    # Find simulated data to compare
    df_results_compare = io.load_parquet_data(comp_dir, prefix="results_")
    df_axes_compare = io.load_parquet_data(comp_dir, prefix="freqs_")

    f_compare = df_axes_compare["freqs_global"].iloc[0]

    H_40_compare = df_results_compare["H_Mag_40"].iloc[sim_index]

    # Plotting logic
    _, (ax1, ax2) = plt.subplots(**vis.apply_standard_style(1, 2))
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
    vis.plot_style(
        ax2,
        f_compare,
        H_40_compare,
        f_exp,
        H_40_exp,
        vline_x=f_cutoff,
        vline_label=r"Cut-off frequency $f_c$",
    )
    vis.axes_transfer_function(ax2)

    plt.tight_layout()
    plt.savefig("figures/06_scaling_comparison.png")


if __name__ == "__main__":
    main()
