import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
from scipy.signal import find_peaks
from numba import njit
import time


time_start = time.time()

# Circuit Parameters
L_0 = 330e-6
dL = 0.2 * L_0 / 2.5
C_0 = 15e-9
dC = 0.1 * C_0 / 2.5
C_end = 7.5e-9
dC_end = dC / 2 / 2.5
R_in_0 = 150
f_cutoff = np.sqrt(1 / L_0 / C_0) / np.pi

N = 41
N_ind = N - 1

# Deviations from ideal behavior
R_L_0 = 0.730
dR_L = 0.1 * R_L_0
G_C_0 = 1e-12
dG_C = 0.1 * G_C_0

# Resistance ratio at 796 kHz from quality factor
f_test = 796e3
Q_test = 65
R_ratio_test = 2 * np.pi * f_test * L_0 / Q_test / R_L_0

# Model R_AC = R_DC * (1 + k_power * f**power_rule)
power_rule = 1.2
k_power_0 = (R_ratio_test - 1) / f_test**power_rule

# Oscilloscope AWG Parameters for input function
awg_frequency = 500.0
vpp = 5.0
num_points = 10000

# Sine wave parameters
duration = 1.0 / awg_frequency
frequencies = np.linspace(2000, 140000, 50)
amplitude = vpp / 2.0

# RK4 parameters
t_eval_points = np.linspace(0, duration, num_points, endpoint=False)
dt = t_eval_points[1] - t_eval_points[0]


@njit
def compute_deriv(t, Y, A, B, amplitude, f_c):
    # Updated to a pure sine wave
    V_in_val = amplitude * np.sin(2 * np.pi * f_c * t)
    return A @ Y + B * V_in_val


@njit
def rk4_solve(t_eval, Y0, A, B, amplitude, f_c):
    n_points = len(t_eval)
    n_states = len(Y0)
    Y_out = np.zeros((n_states, n_points))

    Y_curr = Y0.copy()
    Y_out[:, 0] = Y_curr
    dt = t_eval[1] - t_eval[0]

    for i in range(1, n_points):
        t = t_eval[i - 1]

        k1 = compute_deriv(t, Y_curr, A, B, amplitude, f_c)
        k2 = compute_deriv(t + dt / 2, Y_curr + dt / 2 * k1, A, B, amplitude, f_c)
        k3 = compute_deriv(t + dt / 2, Y_curr + dt / 2 * k2, A, B, amplitude, f_c)
        k4 = compute_deriv(t + dt, Y_curr + dt * k3, A, B, amplitude, f_c)

        Y_curr = Y_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        Y_out[:, i] = Y_curr

    return Y_out


# Domain Randomization Parameters
k_power = np.random.normal(k_power_0, 0.1 * k_power_0)
R_in = np.random.normal(R_in_0, R_in_0 * 0.05)
R_out_mult = np.random.normal(1.0, 0.05)
noise_std = np.random.uniform(0.5e-3, 2.0e-3)
global_temp_drift = np.random.uniform(0.98, 1.02)

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

omega_c = 2.0 / np.sqrt(L_0 * C_0)

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

# Storage for extracted amplitudes
ratio_V40_V0 = []
ratio_error = []

for f_c in frequencies:
    omega = 2.0 * np.pi * f_c

    if omega >= omega_c:
        R_out = 1e6
    else:
        R_out = np.sqrt(L_0 / C_0) / np.sqrt(1 - (omega / omega_c) ** 2)

    R_out *= R_out_mult
    A[N - 1, N - 1] = -G_C[-1] / C[-1] - 1 / (R_out * C[-1])

    R_L_AC = R_L * (1 + k_power * f_c**power_rule)
    for i in range(N - 1):
        A[N + i, N + i] = -R_L_AC[i] / L[i]

    Y0 = np.zeros(total_states)

    # Solve with pure sine wave
    Y_all = rk4_solve(t_eval_points, Y0, A, B, amplitude, f_c)

    # Extract Node 0 and Node 40, injecting noise
    V_0 = Y_all[0, :] + np.random.normal(0, noise_std, num_points)
    V_40 = Y_all[40, :] + np.random.normal(0, noise_std, num_points)

    # if f_c == frequencies[0]:
    #     plt.figure()
    #     plt.plot(t_eval_points, V_0, label=r"$V_0$")
    #     plt.plot(t_eval_points, V_40, label=r"$V_{40}$")
    #     plt.xlabel("Time (s)", fontsize=12)
    #     plt.ylabel("Voltage (V)", fontsize=12)
    #     plt.legend(loc="upper right")
    #     plt.grid(True)
    #     plt.show()

    # Extract steady-state amplitude (using the last 20% of the wave)
    steady_idx = int(0.8 * num_points)

    f_sampling = num_points / duration
    samples_per_period = f_sampling / f_c
    min_dist = int(samples_per_period * 0.6)

    idx_0 = find_peaks(
        V_0[steady_idx:], distance=min_dist / 2, prominence=amplitude * 0.1
    )[0]
    idx_40 = find_peaks(
        V_40[steady_idx:], distance=min_dist, prominence=amplitude * 0.1
    )[0]
    peaks_0 = V_0[steady_idx:][idx_0]
    peaks_40 = V_40[steady_idx:][idx_40]

    if peaks_0.size > 0:
        amp_0 = np.mean(peaks_0)
        damp_0 = np.std(peaks_0)
    else:
        amp_0 = (np.max(V_0[steady_idx:]) - np.min(V_0[steady_idx:])) / 2.0
        damp_0 = 0.0

    if peaks_40.size > 0:
        amp_40 = np.mean(peaks_40)
        damp_40 = np.std(peaks_40)
    else:
        amp_40 = (np.max(V_40[steady_idx:]) - np.min(V_40[steady_idx:])) / 2.0
        damp_40 = 0.0

    # amp_40 = (np.max(V_40[steady_idx:]) - np.min(V_40[steady_idx:])) / 2.0

    ratio_V40_V0.append(amp_40 / amp_0 if amp_0 != 0 else 0)
    if amp_0 != 0:
        ratio_error.append(
            np.sqrt((damp_40 / amp_0) ** 2 + (damp_0 * amp_40 / amp_0**2) ** 2)
        )
    else:
        ratio_error.append(0.0)

print(f"Elapsed time: {time.time() - time_start} seconds")

# Convert lists to arrays for plotting
ratio_V40_V0 = np.array(ratio_V40_V0)

# Manual data
path = "data/electrical/task_2_6_res.xlsx"
df = pd.read_excel(path, header=None, engine="openpyxl")

n = df.iloc[2:33, 0].to_numpy()
in_out = df.iloc[2:33, 1].to_numpy()
f_m = df.iloc[2:33, 4].to_numpy()
v0_m = df.iloc[2:33, 5].to_numpy()
dv0_m = df.iloc[2:33, 6].to_numpy()
v38_m = df.iloc[2:33, 7].to_numpy()
dv38_m = df.iloc[2:33, 8].to_numpy()

w_m = 2 * np.pi * f_m
v0_m = v0_m / 2
dv0_m = dv0_m / 2
v38_m = v38_m / 2
dv38_m = dv38_m / 2
k = np.pi * n / 40


# Fitting function
def sine(t, A, w, phi, c):
    return A * np.sin(w * t + phi) + c


# Loop over oscilloscope data to find more precise values
w = []
dw = []
vmax_0 = []
vmax_38 = []
vmax_in = []
dvmax_0 = []
dvmax_38 = []
dvmax_in = []
phase_diffs = []

for i in range(len(n)):
    file_path = "data/electrical/task_2_6_res/" + f"AMPPUL{i:02d}.CSV"
    data = pd.read_csv(file_path)
    t = data.iloc[:, 0].to_numpy()
    v_0 = data.iloc[:, 1].to_numpy()
    v_38 = data.iloc[:, 2].to_numpy()
    v_in = data.iloc[:, 3].to_numpy()

    popt0, pcov0 = opt.curve_fit(sine, t, v_0, p0=[v0_m[i], w_m[i], 0, 0])
    popt38, pcov38 = opt.curve_fit(sine, t, v_38, p0=[v38_m[i], w_m[i], 0, 0])
    popt_in, pcov_in = opt.curve_fit(sine, t, v_in, p0=[np.max(v_in), w_m[i], 0, 0])

    w.append(popt0[1])
    dw.append(np.sqrt(pcov0[1, 1]))
    vmax_0.append(np.abs(popt0[0]))
    vmax_38.append(np.abs(popt38[0]))
    vmax_in.append(np.abs(popt_in[0]))
    dvmax_0.append(np.sqrt(pcov0[0, 0]))
    dvmax_38.append(np.sqrt(pcov38[0, 0]))
    dvmax_in.append(np.sqrt(pcov_in[0, 0]))

    phi0 = popt0[2] if popt0[0] > 0 else popt0[2] + np.pi
    phi38 = popt38[2] if popt38[0] > 0 else popt38[2] + np.pi
    phase_diffs.append((phi38 - phi0) % (2 * np.pi))

w = np.array(w)
dw = np.array(dw)
vmax_0 = np.array(vmax_0)
vmax_38 = np.array(vmax_38)
vmax_in = np.array(vmax_in)
dvmax_0 = np.array(dvmax_0)
dvmax_38 = np.array(dvmax_38)
dvmax_in = np.array(dvmax_in)
phase_diffs = np.array(phase_diffs)

ratio = vmax_38 / vmax_0
d_ratio = ratio * np.sqrt((dvmax_38 / vmax_38) ** 2 + (dvmax_0 / vmax_0) ** 2)


# Linear fit
def linear_func(x, m, c):
    return m * x + c


popt_sim, _ = opt.curve_fit(linear_func, frequencies, ratio_V40_V0)

freq_exp = w / (2 * np.pi)
popt_exp, _ = opt.curve_fit(linear_func, freq_exp, ratio)

# 4. Plotting
# Amplitude Ratio
plt.figure()
plt.rc("axes", labelsize=14)
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
sim = plt.errorbar(
    frequencies / 1000,
    ratio_V40_V0,
    yerr=ratio_error,
    fmt="bo",
    label="Simulation",
)
exp = plt.errorbar(
    w * 1e-3 / 2 / np.pi, ratio, yerr=d_ratio, fmt="ro", label="Experimental data"
)
(sim_fit,) = plt.plot(
    frequencies / 1000,
    linear_func(frequencies, *popt_sim),
    "b--",
    label="Simulation Fit",
)
(exp_fit,) = plt.plot(
    frequencies / 1000,
    linear_func(frequencies, *popt_exp),
    "r--",
    label="Experimental Fit",
)
vline = plt.axvline(
    x=f_cutoff / 1000,
    color="k",
    linestyle="--",
    label=r"Cut-off frequency $f_c$",
)
handles = [sim, exp, sim_fit, exp_fit, vline]
labels = [h.get_label() for h in handles]

plt.xlabel("Frequency [kHz]", fontsize=16)
plt.ylabel("$V_{40} / V_0$", fontsize=16)
plt.legend(handles, labels, loc="lower left", fontsize=14)
plt.tight_layout()
plt.savefig("figures/ratio_comparison.png")
plt.show()
