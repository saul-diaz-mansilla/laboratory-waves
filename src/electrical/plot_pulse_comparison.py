import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.fft as fft
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import src.utils.landaubeta as lb

lb.use_latex_fonts()

# 1. Load Simulation Data
output_dir = "data/electrical/pulses"
existing_indices = []
if os.path.exists(output_dir):
    for f in os.listdir(output_dir):
        if f.startswith("output_") and f.endswith(".csv"):
            try:
                existing_indices.append(int(f[7:-4]))
            except ValueError:
                continue

if not existing_indices:
    print("No simulation data found.")
    sys.exit(1)

latest_idx = max(existing_indices)
sim_file = os.path.join(output_dir, f"output_{latest_idx}.csv")
print(f"Loading simulation data from {sim_file}")

data_sim = pd.read_csv(sim_file)
t_sim = data_sim["time"].to_numpy()
v0_sim = data_sim["v_0"].to_numpy()
v38_sim = data_sim["v_38"].to_numpy()

# 2. Plot Time Domain Simulation
plt.figure(figsize=(10, 6))
plt.plot(t_sim * 1000, v0_sim, label="$V_0$ (Input)")
plt.plot(t_sim * 1000, v38_sim, label="$V_{38}$ (Output)")
plt.title("Single Pulse Propagation (Simulation)")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.xlim(0, 1)
plt.tight_layout()


# 3. FFT Simulation
dt = t_sim[1] - t_sim[0]
z_0 = fft.fft(v0_sim)
y_0 = fft.fftfreq(len(v0_sim), d=dt)
mask = y_0 > 0
y_0 = y_0[mask]
z_0 = z_0[mask]

z_38 = fft.fft(v38_sim)
y_38 = fft.fftfreq(len(v38_sim), d=dt)
mask_38 = y_38 > 0
y_38 = y_38[mask_38]
z_38 = z_38[mask_38]

plt.figure()
plt.plot(y_0 / 1e3, (np.abs(z_0)), label=r"$V_0$")
plt.plot(y_38 / 1e3, (np.abs(z_38)), label=r"$V_{38}$")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Amplitude")
plt.xlim(0, 145)
plt.legend()


# 4. Load Experimental Data
exp_file = "data/electrical/pulses/FILTERED01.CSV"
if os.path.exists(exp_file):
    data_exp = pd.read_csv(exp_file)
    t_exp = data_exp.iloc[:, 0].to_numpy()
    v0_exp = data_exp.iloc[:, 1].to_numpy()
    v38_exp = data_exp.iloc[:, 2].to_numpy()

    dt_exp = t_exp[1] - t_exp[0]
    z0_exp = fft.fft(v0_exp)
    y0_exp = fft.fftfreq(len(v0_exp), d=dt_exp)
    mask_exp = y0_exp > 0

    plt.figure()
    plt.plot(t_exp, v0_exp, label="Experimental $V_0$")
    plt.plot(t_exp, v38_exp, label="Experimental $V_{38}$")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend()

    plt.figure()
    plt.plot(y_0 / 1e3, np.abs(z_0) / np.max(np.abs(z_0)), label=r"Theoretical $V_0$")
    plt.plot(
        y0_exp[mask_exp] / 1e3,
        np.abs(z0_exp[mask_exp]) / np.max(np.abs(z0_exp[mask_exp])),
        label=r"Experimental $V_0$",
    )
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 140)
    plt.legend()
    plt.show()
else:
    print(f"Experimental data file not found: {exp_file}")
