import numpy as np
import pandas as pd
import os

# import scipy.fft as fft
import matplotlib.pyplot as plt
import time
from numba import njit

simulation_on = False

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

# Shunt conductance in capacitors
G_C_0 = 1e-4  # 10kOhm equivalent parallel resistance
dG_C = 0.1 * G_C_0

# Skin effect coefficient
k_skin_0 = 0.008

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


@njit
def compute_H_global_numba(Z_in_matrix, Z_out_matrix, freqs, y0_full, t_std):
    n_runs, n_points = Z_in_matrix.shape

    # Mask for positive frequencies
    mask_pos = y0_full > 0
    y0 = y0_full[mask_pos]
    n_freqs = len(y0)

    S_xx = np.zeros(n_freqs, dtype=np.float64)
    S_xy = np.zeros(n_freqs, dtype=np.complex128)
    coverage_mask = np.zeros(n_freqs, dtype=np.bool_)

    f_std = 1.0 / (2 * np.pi * t_std) / 2

    for i in range(n_runs):
        # FFT
        z_in_full = Z_in_matrix[i]
        z_out_full = Z_out_matrix[i]

        z_in = z_in_full[mask_pos]
        z_out = z_out_full[mask_pos]

        f_mean = freqs[i]

        # Power
        P_in = np.abs(z_in) ** 2

        # Valid bins logic
        lower = f_mean - f_std
        upper = f_mean + f_std

        for j in range(n_freqs):
            if y0[j] >= lower and y0[j] <= upper:
                S_xx[j] += P_in[j]
                S_xy[j] += z_out[j] * np.conj(z_in[j])
                coverage_mask[j] = True

    H_global = np.zeros(n_freqs, dtype=np.complex128)
    for j in range(n_freqs):
        if coverage_mask[j]:
            H_global[j] = S_xy[j] / S_xx[j]

    return H_global, y0


def H_func_global(V_in_data, V_out_data, t_array, freqs, t_std, threshold=0.3):
    """
    Computes a global transfer function using Cross-Spectral Density,
    masking out the noisy tails of the Gaussian wavepackets.

    threshold: The fraction of the peak power below which the Gaussian's
               tail is ignored (default 1e-3, or 0.1%).
    """
    dt = t_array[1] - t_array[0]

    # Convert inputs to numpy arrays for Numba
    if isinstance(V_in_data, list):
        V_in_matrix = np.stack(V_in_data)
    else:
        V_in_matrix = np.atleast_2d(V_in_data)

    if isinstance(V_out_data, list):
        V_out_matrix = np.stack(V_out_data)
    else:
        V_out_matrix = np.atleast_2d(V_out_data)

    freqs_arr = np.asarray(freqs, dtype=np.float64)
    n_points = V_in_matrix.shape[1]
    y0_full = np.fft.fftfreq(n_points, d=dt)

    Z_in_matrix = np.fft.fft(V_in_matrix, axis=1)
    Z_out_matrix = np.fft.fft(V_out_matrix, axis=1)

    return compute_H_global_numba(Z_in_matrix, Z_out_matrix, freqs_arr, y0_full, t_std)


@njit
def run_frequency_sweep_numba(
    frequencies,
    t_eval_points,
    A_init,
    B,
    L_0,
    C_0,
    omega_c,
    R_out_mult,
    C_last,
    G_C_last,
    R_L,
    L,
    k_skin,
    amplitude,
    t0,
    sigma,
    noise_std,
    N,
):
    n_freqs = len(frequencies)
    n_points = len(t_eval_points)
    total_states = A_init.shape[0]

    V_results = np.zeros((n_freqs, N, n_points), dtype=np.float64)
    A = A_init.copy()

    for i in range(n_freqs):
        f_c = frequencies[i]
        omega = 2.0 * np.pi * f_c

        if omega >= omega_c:
            R_out = 1e6
        else:
            R_out = np.sqrt(L_0 / C_0) / np.sqrt(1 - (omega / omega_c) ** 2)

        R_out *= R_out_mult

        # Update A matrix
        A[N - 1, N - 1] = -G_C_last / C_last - 1 / (R_out * C_last)

        # Update Inductor AC Losses
        sqrt_f = np.sqrt(f_c)
        for j in range(N - 1):
            R_L_AC_val = R_L[j] * (1 + k_skin * sqrt_f)
            A[N + j, N + j] = -R_L_AC_val / L[j]

        Y0 = np.zeros(total_states)

        Y_out = rk4_solve(t_eval_points, Y0, A, B, amplitude, t0, sigma, f_c)

        for node_idx in range(N):
            for t_idx in range(n_points):
                V_results[i, node_idx, t_idx] = Y_out[
                    node_idx, t_idx
                ] + np.random.normal(0, noise_std)

    return V_results


for _ in range(simulation_number):
    # Domain Randomization Parameters
    k_skin = np.random.normal(k_skin_0, 0.1 * k_skin_0)  # ~10% variation
    R_in = np.random.normal(R_in_0, R_in_0 * 0.05)  # 5% tolerance on input resistor
    R_out_mult = np.random.normal(1.0, 0.05)  # 5% tolerance on output resistors
    noise_std = np.random.uniform(0.5e-3, 2.0e-3)  # Varying noise floor
    global_temp_drift = np.random.uniform(0.98, 1.02)  # +/- 2% global capacitance drift

    # Build the Matrices (randomized)
    C = np.concatenate(
        (
            [np.random.normal(C_end, dC_end)],
            np.random.normal(C_0, dC, N - 2),
            [np.random.normal(C_end, dC_end)],
        )
    )
    C *= global_temp_drift
    G_C = np.random.normal(G_C_0, dG_C, N)

    L = np.random.normal(L_0, dL, N - 1)
    R_L = np.random.normal(R_L_0, dR_L, N - 1)

    # Calculate theoretical cutoff angular frequency based on nominal values
    omega_c = 2.0 / np.sqrt(L_0 * C_0)

    # Build First-Order V-I State-Space Matrices
    total_states = 2 * N - 1
    A = np.zeros((total_states, total_states))
    B = np.zeros(total_states)

    B[0] = 1 / (R_in * C[0])
    A[0, 0] = -1 / (R_in * C[0]) - G_C[0] / C[0]
    A[N - 1, total_states - 1] = 1 / C[-1]

    for i in range(N - 1):
        A[i, N + i] = -1 / C[i]
        A[i + 1, N + i] = 1 / C[i + 1]
        A[N + i, i] = 1 / L[i]
        A[N + i, i + 1] = -1 / L[i]
        A[N + i, N + i] = -R_L[i] / L[i]
        A[i + 1, i + 1] = -G_C[i + 1] / C[i + 1]

    V_results = run_frequency_sweep_numba(
        frequencies,
        t_eval_points,
        A,
        B,
        L_0,
        C_0,
        omega_c,
        R_out_mult,
        C[-1],
        G_C[-1],
        R_L,
        L,
        k_skin,
        amplitude,
        t0,
        sigma,
        noise_std,
        N,
    )

    # 6. Calculate Global Transfer Functions for Target Nodes
    target_nodes = np.arange(1, 11) * 4
    freq_data = None
    freqs_global = None

    for node in target_nodes:
        H, freqs_global = H_func_global(
            V_results[:, 0, :], V_results[:, node, :], t_eval_points, frequencies, sigma
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

    # Save targets (L, C, and new ML parameters padded/broadcasted to length N)
    target_data = {
        "C_norm": C / C_0,
        "L_norm": np.pad(L / L_0, (0, 1), constant_values=0),
        "R_L_norm": np.pad(R_L / R_L_0, (0, 1), constant_values=0),
        "k_skin_norm": np.full(N, k_skin / k_skin_0),
        "R_in_norm": np.full(N, R_in / R_in_0),
        "R_out_mult": np.full(N, R_out_mult),
        "noise_std_mV": np.full(N, noise_std / 1e-3),
        "global_temp_drift": np.full(N, global_temp_drift),
        "G_C_norm": np.pad(G_C / G_C_0, (0, 0), constant_values=0),
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
            plt.xlim(0, 160)
            plt.ylim(0, 1.2)
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("No experimental files were successfully loaded. Skipping plot.")
