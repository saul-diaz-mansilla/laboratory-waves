import os
import sys
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.utils.data_io as io
from src.simulation.monte_carlo import simulate
from src.inference.models import build_model_from_config
from src.utils import visualization as vis


def _load_nn_and_paths(config_path):
    master = io.load_config(config_path)
    sim_path = master["paths"]["simulation_config"]
    sim_full = io.load_config(sim_path)
    if "neural_network" not in sim_full:
        raise KeyError(f"Missing neural_network in {sim_path}")
    nn_cfg = sim_full["neural_network"]
    circuit = io.load_config(master["paths"]["circuit_config"])
    sim_exec = sim_full["execution"]
    target_nodes = sim_exec["target_nodes"]
    return master, nn_cfg, circuit, target_nodes


def main():
    parser = argparse.ArgumentParser(description="Inverse-problem inference (gaussian)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the master experimental configuration file. Located in configs/experiment",
    )
    args = parser.parse_args()

    master_cfg, nn_cfg, circuit_parameters, target_nodes = _load_nn_and_paths(
        args.config
    )
    if master_cfg["input"]["waveform"] != "gaussian":
        raise ValueError(
            f"This script expects gaussian waveform, got: {master_cfg['input']['waveform']}"
        )

    arch_cfg = nn_cfg["architecture"]
    num_nodes = len(target_nodes)
    num_freq_bins = arch_cfg["num_freq_bins"]
    n_circ = circuit_parameters["N"]["value"]
    num_outputs = 2 * n_circ
    C_0 = circuit_parameters["C"]["value"]
    L_0 = circuit_parameters["L"]["value"]
    f_cutoff = 1 / np.sqrt(L_0 * C_0) / np.pi

    paths = master_cfg["paths"]
    processed_dir = paths["processed_dir"]
    model_type = nn_cfg.get("model_type", "2d")
    ckpt = paths["model"]

    df_results = io.load_parquet_data(processed_dir, prefix="results_")
    df_freqs = io.load_parquet_data(processed_dir, prefix="freqs_")
    freqs_exp = np.asarray(df_freqs.iloc[0]["freqs_global"])
    sort_idx_exp = np.argsort(freqs_exp)
    freqs_exp = freqs_exp[sort_idx_exp]
    freq_sim_target = np.linspace(freqs_exp.min(), freqs_exp.max(), num_freq_bins)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_1d = target_nodes[-1]

    runs = []

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if model_type == "2d":
        # --- 2D mag + phase ---
        model = build_model_from_config(
            "2d", num_nodes, num_freq_bins, num_outputs, arch_cfg
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        inp = np.zeros((1, 2 * num_nodes, num_freq_bins), dtype=np.float32)
        for i, node in enumerate(target_nodes):
            key_mag = f"H_Mag_{node}"
            key_phase = f"H_Phase_{node}"
            
            if key_mag in df_results.columns and key_phase in df_results.columns:
                H_mag = np.asarray(df_results.iloc[0][key_mag])[sort_idx_exp]
                H_phase = np.asarray(df_results.iloc[0][key_phase])[sort_idx_exp]
            else:
                print(f"Warning: Node {node} data missing from experimental results. Padding with 0.")
                H_mag = np.zeros_like(freqs_exp)
                H_phase = np.zeros_like(freqs_exp)
                
            fm = interp1d(
                freqs_exp,
                H_mag,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            fp = interp1d(
                freqs_exp,
                H_phase,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            inp[0, 2 * i] = fm(freq_sim_target)
            inp[0, 2 * i + 1] = fp(freq_sim_target)
        with torch.no_grad():
            pred = model(torch.tensor(inp, device=device)).cpu().numpy().flatten()
        C_c = pred[:n_circ] * C_0
        L_c = (pred[n_circ : 2 * n_circ] * L_0)[: n_circ - 1]
        runs.append(("2d", C_c, L_c))

    elif model_type == "2d_magnitude":
        # --- 2D magnitude-only ---
        model = build_model_from_config(
            "2d_magnitude", num_nodes, num_freq_bins, num_outputs, arch_cfg
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        inp = np.zeros((1, num_nodes, num_freq_bins), dtype=np.float32)
        for i, node in enumerate(target_nodes):
            key_mag = f"H_Mag_{node}"
            
            if key_mag in df_results.columns:
                H_mag = np.asarray(df_results.iloc[0][key_mag])[sort_idx_exp]
            else:
                print(f"Warning: Node {node} mag data missing. Padding with 0.")
                H_mag = np.zeros_like(freqs_exp)
                
            fm = interp1d(
                freqs_exp,
                H_mag,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            inp[0, i] = fm(freq_sim_target)
        with torch.no_grad():
            pred = model(torch.tensor(inp, device=device)).cpu().numpy().flatten()
        C_c = pred[:n_circ] * C_0
        L_c = (pred[n_circ : 2 * n_circ] * L_0)[: n_circ - 1]
        runs.append(("2d_magnitude", C_c, L_c))

    elif model_type == "1d":
        # --- 1D ---
        model = build_model_from_config(
            "1d", num_nodes, num_freq_bins, num_outputs, arch_cfg
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        H_mag = np.asarray(df_results.iloc[0][f"H_Mag_{node_1d}"])[sort_idx_exp]
        H_phase = np.asarray(df_results.iloc[0][f"H_Phase_{node_1d}"])[sort_idx_exp]
        fm = interp1d(
            freqs_exp, H_mag, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )
        fp = interp1d(
            freqs_exp,
            H_phase,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        inp = np.zeros((1, 2, num_freq_bins), dtype=np.float32)
        inp[0, 0] = fm(freq_sim_target)
        inp[0, 1] = fp(freq_sim_target)
        with torch.no_grad():
            pred = model(torch.tensor(inp, device=device)).cpu().numpy().flatten()
        C_c = pred[:n_circ] * C_0
        L_c = (pred[n_circ : 2 * n_circ] * L_0)[: n_circ - 1]
        runs.append(("1d", C_c, L_c))

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if not runs:
        raise FileNotFoundError(
            f"No checkpoints found. Expected checkpoint at: {ckpt}"
        )

    key_mag = f"H_Mag_{node_1d}"
    H_exp_mag = np.asarray(df_results.iloc[0][key_mag])[sort_idx_exp]

    for tag, C_custom, L_custom in runs:
        print(f"\n--- Inference ({tag}) ---")
        print(f"Mean Predicted L: {np.mean(L_custom) * 1e6:.2f} uH")
        print(f"Mean Predicted C: {np.mean(C_custom) * 1e9:.2f} nF")

        _, sim_results, sim_freqs = simulate(
            args.config, L_custom=L_custom.tolist(), C_custom=C_custom.tolist()
        )
        H_sim_mag = np.asarray(sim_results[key_mag][0])
        freqs_sim = np.asarray(sim_freqs["freqs_global"][0])
        idx_sort = np.argsort(freqs_sim)
        freqs_sim = freqs_sim[idx_sort]
        H_sim_mag = H_sim_mag[idx_sort]

        safe = tag.replace("/", "_")
        out_plot = f"figures/09_infer_parameters_{safe}.png"
        os.makedirs(os.path.dirname(out_plot), exist_ok=True)

        _, ax1 = plt.subplots(**vis.apply_standard_style(1, 1))

        vis.plot_style(
            ax1, 
            freqs_sim, 
            H_sim_mag, 
            freqs_exp, 
            H_exp_mag,
            vline_x=f_cutoff,
            vline_label=r"Cut-off frequency $f_c$",
        )

        vis.axes_transfer_function(ax1)
        plt.savefig(out_plot)
        print(f"Validation plot saved to: {out_plot}")


if __name__ == "__main__":
    main()
