import numpy as np
import pandas as pd
import os
import scipy.fft as fft
import matplotlib.pyplot as plt
import time
from numba import njit

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

dV_exp = 1e-2

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


def H_func_global(V_in_data, V_out_data, t_array, freqs, t_std, dz_exp=0.0):
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
    dH_global = np.zeros(len(y0))

    H_global[coverage_mask] = S_xy[coverage_mask] / S_xx[coverage_mask]

    # Apply error propagation formula
    if dz_exp > 0.0:
        dH_global[coverage_mask] = dz_exp * np.sqrt(
            (1 + np.abs(H_global[coverage_mask]) ** 2) / S_xx[coverage_mask]
        )

    return H_global, y0, dH_global


# Domain Randomization Parameters
power_rule = np.random.normal(power_rule_0, 0.02 * power_rule_0)
R_in = np.random.normal(R_in_0, R_in_0 * 0.05)  # 5% tolerance on input resistor
R_out_mult = np.random.normal(1.0, 0.05)  # 5% tolerance on output resistors
noise_std = np.random.uniform(0.5e-3, 2.0e-3)  # Varying noise floor
global_temp_drift = np.random.uniform(0.98, 1.02)  # +/- 2% global capacitance drift
C_batch_factor = np.random.uniform(1.0, 5.0)
L_batch_factor = np.random.uniform(1.0, 5.0)

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

k_power = 0.0

for sim_num in range(2):
    if sim_num == 1:
        k_power = (R_ratio_test - 1) / f_test**power_rule_0

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

    for node in target_nodes:
        H, freqs_global, _ = H_func_global(
            V_in_all_runs, V_out_nodes_all_runs[node], t_eval_points, frequencies, sigma
        )
    if sim_num == 0:
        H_1 = H
        freqs_1 = freqs_global
    else:
        H_2 = H
        freqs_2 = freqs_global

# 8. Experimental Plot Logic (30 Files)
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

    data_exp = pd.read_csv(file_path)
    t_raw = data_exp.iloc[:, 0].to_numpy()
    v0_raw = data_exp.iloc[:, 1].to_numpy()
    v40_raw = data_exp.iloc[:, 2].to_numpy()

    mask_time = t_raw < (1e-3 - 1e-4)

    if t_exp_common is None:
        t_exp_common = t_raw[mask_time]

    exp_v0_all.append(v0_raw[mask_time])
    exp_v40_all.append(v40_raw[mask_time])

dz_exp = np.sqrt(len(t_raw[mask_time])) * dV_exp

# Extract global H from the full list of 30 experimental arrays
H_exp, y0_exp, dH_exp = H_func_global(
    exp_v0_all, exp_v40_all, t_exp_common, frequencies, sigma, dz_exp
)

plt.rc("axes", labelsize=14)
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 9.6))  # 6.4, 4.8

f_cutoff = np.sqrt(1 / L_0 / C_0) / np.pi

# Top subplot: H_1 (k_power=0) vs H_exp
(sim_1,) = ax1.plot(freqs_1 / 1e3, np.abs(H_1), "bo-", label="Simulation")
exp_1 = ax1.errorbar(
    y0_exp / 1e3, np.abs(H_exp), yerr=dH_exp, fmt="ro-", label="Experiment"
)
vline_1 = ax1.axvline(
    x=f_cutoff / 1000,
    color="k",
    linestyle="--",
    label=r"Cut-off frequency $f_c$",
)

handles_1 = [sim_1, exp_1, vline_1]
labels_1 = [h.get_label() for h in handles_1]
ax1.set_xlabel("Frequency [kHz]", fontsize=16)
ax1.set_ylabel("$V_{40}(f) / V_0(f)$", fontsize=16)
ax1.legend(handles_1, labels_1, loc="lower left", fontsize=12)
ax1.set_xlim((frequencies[0] - (0.5 / (2 * np.pi * sigma))) / 1e3, 150)
ax1.set_ylim(0, 1.2)

# Bottom subplot: H_2 (randomized k_power) vs H_exp
(sim_2,) = ax2.plot(freqs_2 / 1e3, np.abs(H_2), "bo-", label="Simulation")
exp_2 = ax2.errorbar(
    y0_exp / 1e3, np.abs(H_exp), yerr=dH_exp, fmt="ro-", label="Experiment"
)
vline_2 = ax2.axvline(
    x=f_cutoff / 1000,
    color="k",
    linestyle="--",
    label=r"Cut-off frequency $f_c$",
)
handles_2 = [sim_2, exp_2, vline_2]
labels_2 = [h.get_label() for h in handles_2]
ax2.set_xlabel("Frequency [kHz]", fontsize=16)
ax2.set_ylabel("$V_{40}(f) / V_0(f)$", fontsize=16)
ax2.legend(handles_2, labels_2, loc="lower left", fontsize=12)
ax2.set_xlim((frequencies[0] - (0.5 / (2 * np.pi * sigma))) / 1e3, 150)
ax2.set_ylim(0, 1.2)

plt.tight_layout()
plt.savefig("figures/transfer_comparison.png")
plt.show()


# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))

f_cutoff = np.sqrt(1 / L_0 / C_0) / np.pi

# Top subplot: H_1 (k_power=0) vs H_exp
(sim_1,) = ax1.plot(freqs_1 / 1e3, np.abs(H_1), "bo-", label="Simulation")
exp_1 = ax1.errorbar(
    y0_exp / 1e3, np.abs(H_exp), yerr=dH_exp, fmt="ro-", label="Experimental data"
)
vline_1 = ax1.axvline(
    x=f_cutoff / 1000,
    color="k",
    linestyle="--",
    label=r"Cut-off frequency $f_c$",
)

handles_1 = [sim_1, exp_1, vline_1]
labels_1 = [h.get_label() for h in handles_1]
ax1.set_xlabel("Frequency [kHz]", fontsize=16)
ax1.set_ylabel("$V_{40}(f) / V_0(f)$", fontsize=16)
ax1.legend(handles_1, labels_1, loc="lower left", fontsize=12)
ax1.set_xlim((frequencies[0] - (0.5 / (2 * np.pi * sigma))) / 1e3, 150)
ax1.set_ylim(0, 1.2)

# Bottom subplot: H_2 (randomized k_power) vs H_exp
(sim_2,) = ax2.plot(freqs_2 / 1e3, np.abs(H_2), "bo-", label="Simulation")
exp_2 = ax2.errorbar(
    y0_exp / 1e3, np.abs(H_exp), yerr=dH_exp, fmt="ro-", label="Experimental data"
)
vline_2 = ax2.axvline(
    x=f_cutoff / 1000,
    color="k",
    linestyle="--",
    label=r"Cut-off frequency $f_c$",
)
handles_2 = [sim_2, exp_2, vline_2]
labels_2 = [h.get_label() for h in handles_2]
ax2.set_xlabel("Frequency [kHz]", fontsize=16)
ax2.set_ylabel("$V_{40}(f) / V_0(f)$", fontsize=16)
ax2.legend(handles_2, labels_2, loc="lower left", fontsize=12)
ax2.set_xlim((frequencies[0] - (0.5 / (2 * np.pi * sigma))) / 1e3, 150)
ax2.set_ylim(0, 1.2)

plt.tight_layout()
plt.savefig("figures/transfer_comparison_horizontal.png")
plt.show()
