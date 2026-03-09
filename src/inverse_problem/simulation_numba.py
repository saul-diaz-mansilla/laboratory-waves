import numpy as np
import pandas as pd
import os
import scipy.fft as fft
import matplotlib.pyplot as plt
import time
from numba import njit

simulation_on = True

if simulation_on:
    simulation_number = 10000
else:
    simulation_number = 1

# Circuit Parameters
L_0 = 330e-6
dL = 0.2 * L_0
C_0 = 15e-9
dC = 0.1 * C_0
C_end = 7.5e-9
dC_end = dC / 2
R_in_0 = 150

N = 41
N_ind = N - 1

# Deviations from ideal behavior
# Parasitic resistances in L
R_L_0 = 0.730
dR_L = 0.1 * R_L_0

# Resistance ratio at 796 kHz from quality factor
f_test = 796e3
Q_test = 65
R_ratio_test = 2 * np.pi * f_test * L_0 / Q_test / R_L_0

# Model R_AC = R_DC * (1 + k_power * f**power_rule)
power_rule_0 = 1.25

# Oscilloscope AWG Parameters for input function
awg_frequency = 500.0
vpp = 5.0
num_points = 10000

# Gaussian wavepacket parameters
duration = 1.0 / awg_frequency
frequencies = np.linspace(20000, 140000, 30)
t0 = duration / 2
sigma = 0.00002
amplitude = vpp / 2.0

# RK4 parameters
t_eval_points = np.linspace(0, duration, num_points, endpoint=False)
dt = t_eval_points[1] - t_eval_points[0]

time_start = time.time()


@njit
def compute_deriv(t, Y, A, B, amplitude, t0, sigma, f_c):
    V_in_val = (
        amplitude
        * np.exp(-((t - t0) ** 2) / (2 * sigma**2))
        * np.cos(2 * np.pi * f_c * t)
    )
    return A @ Y + B * V_in_val


@njit
def rk4_solve(t_eval, Y0, A, B, amplitude, t0, sigma, f_c):
    n_points = len(t_eval)
    n_states = len(Y0)
    Y_out = np.zeros((n_states, n_points))

    Y_curr = Y0.copy()
    Y_out[:, 0] = Y_curr

    dt = t_eval[1] - t_eval[0]

    for i in range(1, n_points):
        t = t_eval[i - 1]

        k1 = compute_deriv(t, Y_curr, A, B, amplitude, t0, sigma, f_c)
        k2 = compute_deriv(
            t + dt / 2, Y_curr + dt / 2 * k1, A, B, amplitude, t0, sigma, f_c
        )
        k3 = compute_deriv(
            t + dt / 2, Y_curr + dt / 2 * k2, A, B, amplitude, t0, sigma, f_c
        )
        k4 = compute_deriv(t + dt, Y_curr + dt * k3, A, B, amplitude, t0, sigma, f_c)

        Y_curr = Y_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        Y_out[:, i] = Y_curr

    return Y_out


def H_func_global(V_in_data, V_out_data, t_array, freqs, t_std):
    """
    Computes a global transfer function using Cross-Spectral Density,
    masking out the noisy tails of the Gaussian wavepackets.
    """
    dt = t_array[1] - t_array[0]
    # Calculate the frequency standard deviation theoretically
    f_std_2 = 1.0 / (2 * np.pi * t_std) / 2

    if not isinstance(V_in_data, list):
        V_in_data = [V_in_data]
        V_out_data = [V_out_data]

    y0 = fft.fftfreq(len(V_in_data[0]), d=dt)
    mask_pos = (y0 > (freqs[0] - f_std_2)) & (y0 < 150e3)
    y0 = y0[mask_pos]

    S_xx = np.zeros(len(y0))
    S_xy = np.zeros(len(y0), dtype=complex)

    # Track which frequency bins received valid data to avoid zero-division later
    coverage_mask = np.zeros(len(y0), dtype=bool)

    for v_in, v_out, f_mean in zip(V_in_data, V_out_data, freqs):
        z_in = fft.fft(v_in)[mask_pos]
        z_out = fft.fft(v_out)[mask_pos]

        # Calculate the Power Spectral Density (PSD) of the input
        P_in = np.abs(z_in) ** 2

        # Identify the high-energy region of this specific wavepacket
        valid_bins = (y0 >= f_mean - f_std_2) & (y0 <= f_mean + f_std_2)

        # Accumulate only the significant bins (cutting off the noisy tails)
        S_xx[valid_bins] += P_in[valid_bins]
        S_xy[valid_bins] += (z_out * np.conj(z_in))[valid_bins]

        # Mark these frequencies as covered
        coverage_mask[valid_bins] = True

    # Calculate global transfer function safely
    H_global = np.zeros(len(y0), dtype=complex)

    H_global[coverage_mask] = S_xy[coverage_mask] / (S_xx[coverage_mask])

    return H_global, y0


for _ in range(simulation_number):
    # Domain Randomization Parameters
    power_rule = np.random.normal(power_rule_0, 0.02 * power_rule_0)
    R_in = np.random.normal(R_in_0, R_in_0 * 0.05)  # 5% tolerance on input resistor
    R_out_mult = np.random.normal(1.0, 0.05)  # 5% tolerance on output resistors
    noise_std = np.random.uniform(0.5e-3, 2.0e-3)  # Varying noise floor
    global_temp_drift = np.random.uniform(0.98, 1.02)  # +/- 2% global capacitance drift
    C_batch_factor = np.random.uniform(1.0, 5.0)
    L_batch_factor = np.random.uniform(1.0, 5.0)

    k_power = (R_ratio_test - 1) / f_test**power_rule_0

    # Build the Matrices (randomized)
    C = np.concatenate(
        (
            [np.random.normal(C_end, dC_end / C_batch_factor)],
            np.random.normal(C_0, dC / C_batch_factor, N - 2),
            [np.random.normal(C_end, dC_end / C_batch_factor)],
        )
    )
    C *= global_temp_drift

    L = np.random.normal(L_0, dL / L_batch_factor, N - 1)
    R_L = np.random.normal(R_L_0, dR_L, N - 1)

    # Calculate theoretical cutoff angular frequency based on nominal values
    omega_c = 2.0 / np.sqrt(L_0 * C_0)

    # Build First-Order V-I State-Space Matrices
    total_states = 2 * N - 1
    A = np.zeros((total_states, total_states))
    B = np.zeros(total_states)

    B[0] = 1 / (R_in * C[0])
    A[0, 0] = -1 / (R_in * C[0])
    A[N - 1, total_states - 1] = 1 / C[-1]

    for i in range(N - 1):
        A[i, N + i] = -1 / C[i]
        A[i + 1, N + i] = 1 / C[i + 1]
        A[N + i, i] = 1 / L[i]
        A[N + i, i + 1] = -1 / L[i]
        A[N + i, N + i] = -R_L[i] / L[i]

    V_in_all_runs = []
    V_out_nodes_all_runs = {node: [] for node in np.arange(1, 11) * 4}

    # Integrate 30 times (Dynamic Impedance Matching)
    for f_c in frequencies:
        omega = 2.0 * np.pi * f_c

        # Calculate frequency-dependent matched impedance
        if omega >= omega_c:
            # Prevent complex numbers/division by zero if approaching or exceeding cutoff
            R_out = 1e6
        else:
            R_out = np.sqrt(L_0 / C_0) / np.sqrt(1 - (omega / omega_c) ** 2)

        R_out *= R_out_mult

        # Update the load boundary condition in the A matrix for this specific frequency
        A[N - 1, N - 1] = -1 / (R_out * C[-1])

        # --- Update Inductor AC Losses ---
        R_L_AC = R_L * (1 + k_power * f_c**power_rule)
        for i in range(N - 1):
            A[N + i, N + i] = -R_L_AC[i] / L[i]

        Y0 = np.zeros(total_states)

        Y_all = rk4_solve(t_eval_points, Y0, A, B, amplitude, t0, sigma, f_c)
        V_clean = Y_all[:N, :]

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
            V_in_all_runs, V_out_nodes_all_runs[node], t_eval_points, frequencies, sigma
        )

        if freq_data is None:
            freq_data = {"Frequency (Hz)": freqs_global}

        freq_data[f"H_Mag_{node}"] = np.abs(H)
        freq_data[f"H_Phase_{node}"] = np.angle(H)

    # 7. Exporting Data via Parquet
    output_dir = "data/inverse_problem/simulations_gaussians"
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

    # Save targets (L, C, and new ML parameters padded/broadcasted to length N)
    target_data = {
        "C_norm": C / C_0,
        "L_norm": np.pad(L / L_0, (0, 1), constant_values=0),
        "R_L_norm": np.pad(R_L / R_L_0, (0, 1), constant_values=0),
        "power_rule": np.full(N, power_rule),
        "R_in_norm": np.full(N, R_in / R_in_0),
        "R_out_mult": np.full(N, R_out_mult),
        "noise_std_mV": np.full(N, noise_std / 1e-3),
        "global_temp_drift": np.full(N, global_temp_drift),
        "C_batch_factor": np.full(N, C_batch_factor),
        "L_batch_factor": np.full(N, L_batch_factor),
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
        frequencies = np.linspace(20000, 140000, 30)
        sigma = 0.00002

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
            H_exp, y0_exp = H_func_global(
                exp_v0_all, exp_v40_all, t_exp_common, frequencies, sigma
            )

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
            plt.xlim(0, 150)
            # plt.xlim((frequencies[0] - (0.5 / (2 * np.pi * sigma))) / 1e3, 150)
            plt.ylim(0, 1.2)
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("No experimental files were successfully loaded. Skipping plot.")
