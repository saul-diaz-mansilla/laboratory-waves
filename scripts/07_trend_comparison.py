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
    coeff_exp = np.polyfit(f_exp, H_40_exp, 1)

    # Find simulated data
    df_results = io.load_parquet_data(sim_dir, prefix="results_")
    df_axes = io.load_parquet_data(sim_dir, prefix="freqs_")

    f_sim = df_axes["freqs_global"].iloc[0]

    H_40_sim = df_results["H_Mag_40"].iloc[sim_index]
    coeff_sim = np.polyfit(f_sim, H_40_sim, 1)

    # Plotting logic
    _, ax1 = plt.subplots(**vis.apply_standard_style(1, 1))
    vis.plot_style(
        ax1,
        f_sim,
        H_40_sim,
        f_exp,
        H_40_exp,
        x_exp_trend=f_exp,
        x_sim_trend=f_sim,
        y_exp_trend=np.polyval(coeff_exp, f_exp),
        y_sim_trend=np.polyval(coeff_sim, f_sim),
        vline_x=f_cutoff,
        vline_label=r"Cut-off frequency $f_c$",
    )
    vis.axes_transfer_function(ax1)

    plt.tight_layout()
    plt.savefig("figures/07_trend_comparison.png")


if __name__ == "__main__":
    main()
