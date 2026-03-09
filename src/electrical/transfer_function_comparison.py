import numpy as np
import pandas as pd
import os
import scipy.fft as fft
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Circuit Parameters
L_0 = 330e-6
dL = 0.2 * L_0 / 2.5
C_0 = 15e-9
dC = 0.1 * C_0 / 2.5
C_end = 7.5e-9
dC_end = dC / 2 / 2.5
R_in_0 = 150

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
    mask_pos = y0 > (freqs[0] - f_std_2)
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

# Manual data
path = "data/electrical/task_2_6.xlsx"
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

    popt0, pcov0 = curve_fit(sine, t, v_0, p0=[v0_m[i], w_m[i], 0, 0])
    popt38, pcov38 = curve_fit(sine, t, v_38, p0=[v38_m[i], w_m[i], 0, 0])
    popt_in, pcov_in = curve_fit(sine, t, v_in, p0=[np.max(v_in), w_m[i], 0, 0])

    # if i == 0:
    #     plt.figure()
    #     plt.plot(t, v_0, label="v0 data")
    #     plt.plot(t, sine(t, *popt0), label="v0 fit")
    #     plt.plot(t, v_38, label="v38 data")
    #     plt.plot(t, sine(t, *popt38), label="v38 fit")
    #     plt.legend()
    #     plt.show()

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

if len(exp_v0_all) > 0:
    # Extract global H from the full list of 30 experimental arrays
    H_exp, y0_exp = H_func_global(
        exp_v0_all, exp_v40_all, t_exp_common, frequencies, sigma
    )

    plt.figure()
    plt.plot(y0_exp / 1e3, np.abs(H_exp), label="Experimental $|H_{global}(f)|$")
    plt.errorbar(
        w * 1e-3 / 2 / np.pi, ratio, yerr=d_ratio, fmt=".", label="Experimental"
    )
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude $|V_{40}(f) / V_0(f)|$")
    plt.title("System Resonances (Global Transfer Function)")
    plt.xlim(0, 160)
    plt.ylim(0, 1.2)
    plt.legend()
    plt.grid(True)
    plt.show()
