import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.utils.data_io as io
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
    node = 40
    H_node_exp = df_results[f"H_Phase_{node}"].iloc[sim_index]
    k_exp = -np.unwrap(H_node_exp) / node

    # Avoid unwrapping errors as the dispersion relation is strictly increasing
    for i in range(1, len(k_exp)):
        if k_exp[i] < k_exp[i - 1]:
            k_exp[i:] += (2 * np.pi) / node

    # Find simulated data
    df_results = io.load_parquet_data(sim_dir, prefix="results_")
    df_axes = io.load_parquet_data(sim_dir, prefix="freqs_")

    f_sim = df_axes["freqs_global"].iloc[0]

    H_node_sim = df_results[f"H_Phase_{node}"].iloc[sim_index]
    k_sim = -np.unwrap(H_node_sim) / node

    # Avoid unwrapping errors as the dispersion relation is strictly increasing
    for i in range(1, len(k_sim)):
        if k_sim[i] < k_sim[i - 1]:
            k_sim[i:] += (2 * np.pi) / node

    # Plotting logic
    _, ax1 = plt.subplots(**vis.apply_standard_style(1, 1))
    vis.plot_style(
        ax1,
        k_sim,
        f_sim,
        k_exp,
        f_exp,
        hline_y=f_cutoff,
        hline_label=r"Cut-off frequency $f_c$",
    )
    vis.axes_dispersion_relation(ax1)

    plt.tight_layout()
    plt.savefig("figures/05_dispersion_relation.png")


if __name__ == "__main__":
    main()
