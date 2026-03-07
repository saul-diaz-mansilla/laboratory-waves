import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import os
import scipy.fft as fft
import matplotlib.pyplot as plt
import time

simulation_on = False

time_start = time.time()


def H_func_global(V_in_data, V_out_data, t_array):
    """
    Computes a global transfer function using Cross-Spectral Density.
    Accepts lists of arrays to aggregate multiple independent runs.
    """
    dt = t_array[1] - t_array[0]

    if not isinstance(V_in_data, list):
        V_in_data = [V_in_data]
        V_out_data = [V_out_data]

    y0 = fft.fftfreq(len(V_in_data[0]), d=dt)
    mask_pos = y0 > 0
    y0 = y0[mask_pos]

    S_xx = np.zeros(len(y0))
    S_xy = np.zeros(len(y0), dtype=complex)

    for v_in, v_out in zip(V_in_data, V_out_data):
        z_in = fft.fft(v_in)[mask_pos]
        z_out = fft.fft(v_out)[mask_pos]

        S_xx += np.abs(z_in) ** 2
        S_xy += z_out * np.conj(z_in)

    epsilon = np.max(S_xx) * 1e-10
    H_global = S_xy / (S_xx + epsilon)

    return H_global, y0


if simulation_on:
    simulation_number = 10000
else:
    simulation_number = 1

# --- 1. Oscilloscope AWG Parameters ---
awg_frequency = 500.0
vpp = 5.0
num_points = 10000

duration = 1.0 / awg_frequency
t_eval_points = np.linspace(0, duration, num_points, endpoint=False)
dt = t_eval_points[1] - t_eval_points[0]

frequencies = np.linspace(20000, 140000, 30)
t0 = duration / 2
sigma = 0.00002
amplitude = vpp / 2.0


for _ in range(simulation_number):
    # 1. Circuit Parameters
    L_0 = 330e-6
    dL = 0.2 * L_0
    C_0 = 15e-9
    dC = 0.1 * C_0
    C_end = 7.5e-9
    dC_end = dC / 2
    R_in = 150
    # R_out is now calculated dynamically inside the loop

    N = 41
    N_ind = N - 1

    R_L_0 = 0.730
    dR_L = 0.1 * R_L_0
    noise_std = 1e-3

    # 2. Build the Matrices (Randomized)
    C = np.concatenate(
        (
            [np.random.normal(C_end, dC_end)],
            np.random.normal(C_0, dC, N - 2),
            [np.random.normal(C_end, dC_end)],
        )
    )
    L = np.random.normal(L_0, dL, N - 1)
    R_L = np.random.normal(R_L_0, dR_L, N - 1)

    # Calculate theoretical cutoff angular frequency based on nominal values
    omega_c = 2.0 / np.sqrt(L_0 * C_0)

    # 3. Build First-Order V-I State-Space Matrices
    total_states = 2 * N - 1
    A = np.zeros((total_states, total_states))
    B = np.zeros(total_states)

    B[0] = 1 / (R_in * C[0])
    A[0, 0] = -1 / (R_in * C[0])
    # A[N - 1, N - 1] is reserved for R_out and will be updated dynamically
    A[N - 1, total_states - 1] = 1 / C[-1]

    for i in range(N - 1):
        A[i, N + i] = -1 / C[i]
        A[i + 1, N + i] = 1 / C[i + 1]
        A[N + i, i] = 1 / L[i]
        A[N + i, i + 1] = -1 / L[i]
        A[N + i, N + i] = -R_L[i] / L[i]

    V_in_all_runs = []
    V_out_nodes_all_runs = {node: [] for node in np.arange(1, 11) * 4}

    # 4. Integrate 30 times (Dynamic Impedance Matching)
    for f_c in frequencies:
        omega = 2.0 * np.pi * f_c

        # Calculate frequency-dependent matched impedance
        if omega >= omega_c:
            # Prevent complex numbers/division by zero if approaching or exceeding cutoff
            R_out = 1e6
        else:
            R_out = np.sqrt(L_0 / C_0) / np.sqrt(1 - (omega / omega_c) ** 2)

        # Update the load boundary condition in the A matrix for this specific frequency
        A[N - 1, N - 1] = -1 / (R_out * C[-1])

        def odefunc(t, Y):
            V_in_val = (
                amplitude
                * np.exp(-((t - t0) ** 2) / (2 * sigma**2))
                * np.cos(2 * np.pi * f_c * t)
            )
            return A @ Y + B * V_in_val

        Y0 = np.zeros(total_states)
        sol = solve_ivp(
            odefunc,
            (0, duration),
            Y0,
            method="RK45",
            max_step=1e-7,
            t_eval=t_eval_points,
        )

        V_clean = sol.y[:N, :]
        V_noisy = V_clean + np.random.normal(0, noise_std, V_clean.shape)

        V_in_all_runs.append(V_noisy[0, :])
        for node in V_out_nodes_all_runs.keys():
            V_out_nodes_all_runs[node].append(V_noisy[node, :])

    # 6. Calculate Global Transfer Functions for Target Nodes
    target_nodes = np.arange(1, 11) * 4
    freq_data = None
    freqs_global = None

    for node in target_nodes:
        H, freqs_global = H_func_global(
            V_in_all_runs, V_out_nodes_all_runs[node], t_eval_points
        )

        if freq_data is None:
            freq_data = {"Frequency (Hz)": freqs_global}

        freq_data[f"H_Mag_{node}"] = np.abs(H)
        freq_data[f"H_Phase_{node}"] = np.angle(H)

    # 7. Exporting Data via Parquet
    output_dir = "data/temp"
    os.makedirs(output_dir, exist_ok=True)

    existing_indices = []
    for f in os.listdir(output_dir):
        if f.startswith("freq_data_") and f.endswith(".parquet"):
            existing_indices.append(int(f[10:-8]))

    next_idx = max(existing_indices) + 1 if existing_indices else 1

    df_freq = pd.DataFrame(freq_data)
    df_freq.to_parquet(
        os.path.join(output_dir, f"freq_data_{next_idx}.parquet"), engine="pyarrow"
    )

    target_data = {
        "C_norm": C / C_0,
        "L_norm": np.pad(L / L_0, (0, 1), constant_values=0),
    }
    df_targets = pd.DataFrame(target_data)
    df_targets.to_parquet(
        os.path.join(output_dir, f"targets_{next_idx}.parquet"), engine="pyarrow"
    )

    print(
        f"Simulation {next_idx} completed. Time elapsed: {time.time() - time_start:.2f} s"
    )

    # 8. Experimental Plot Logic (30 Files)
    if not simulation_on:
        exp_dir = "data/electrical/gaussian_2/"
        exp_v0_all = []
        exp_v40_all = []
        t_exp_common = None

        for i in range(1, 31):
            # Attempt to read both non-padded and zero-padded filenames to be safe
            file_path = os.path.join(exp_dir, f"FILTERED{i}.CSV")
            if not os.path.exists(file_path):
                file_path = os.path.join(exp_dir, f"FILTERED{i:02d}.CSV")

            if os.path.exists(file_path):
                data_exp = pd.read_csv(file_path)
                t_raw = data_exp.iloc[:, 0].to_numpy()
                v0_raw = data_exp.iloc[:, 1].to_numpy()
                v40_raw = data_exp.iloc[:, 2].to_numpy()

                mask_time = t_raw < (1e-3 - 1e-4)

                if t_exp_common is None:
                    t_exp_common = t_raw[mask_time]

                exp_v0_all.append(v0_raw[mask_time])
                exp_v40_all.append(v40_raw[mask_time])
            else:
                print(f"Warning: Experimental file {file_path} not found.")

        if len(exp_v0_all) > 0:
            # Extract global H from the full list of 30 experimental arrays
            H_exp, y0_exp = H_func_global(exp_v0_all, exp_v40_all, t_exp_common)

            plt.figure()
            plt.plot(
                freqs_global / 1e3, np.abs(H), label="Simulation $|H_{global}(f)|$"
            )
            plt.plot(
                y0_exp / 1e3, np.abs(H_exp), label="Experimental $|H_{global}(f)|$"
            )
            plt.xlabel("Frequency (kHz)")
            plt.ylabel("Magnitude $|V_{40}(f) / V_0(f)|$")
            plt.title("System Resonances (Global Transfer Function)")
            plt.xlim(0, 160)
            plt.ylim(0, 1.2)
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("No experimental files were successfully loaded. Skipping plot.")
