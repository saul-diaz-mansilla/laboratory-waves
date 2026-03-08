import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
# from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import src.utils.landaubeta as lb

lb.use_latex_fonts()

# 1. Load Data
i = 3
input_path = f"data/electrical/theory_system_random/matched/output_{i}.csv"
try:
    df = pd.read_csv(input_path)
except FileNotFoundError:
    print(f"Error: Data file not found at '{input_path}'")
    print("Please run 'theory_system_random.py' first to generate the data.")
    sys.exit(1)

# 2. Process Data
frequencies = df["frequency"].to_numpy()
V_amps = df.iloc[:, 1:].to_numpy()

# Calculate ratios
ratios = [V_amps[:, i] / V_amps[:, 0] for i in range(1, V_amps.shape[1])]

# 3. Plotting
plot_ratios = [38]
plt.figure(figsize=(10, 6))
for j in plot_ratios:
    plt.plot(
        frequencies / 1000,
        ratios[j - 1],
        "-o",
        linewidth=2,
        label="$|V_{" + str(j) + "} / V_0|$",
    )
plt.title("Steady-State Amplitude Ratio (Random Components)", fontsize=14)
plt.xlabel("Frequency (kHz)", fontsize=12)
plt.ylabel("Amplitude Ratio", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("figures/theory_system_random.pdf")

plot_amps = [0, 1]
plt.figure(figsize=(10, 6))
for j in plot_amps:
    plt.plot(
        frequencies / 1000,
        V_amps[:, j],
        "-o",
        linewidth=2,
        label="$|V_{" + str(j) + "}|$",
    )
plt.title("Steady-State Amplitude (Random Components)", fontsize=14)
plt.xlabel("Frequency (kHz)", fontsize=12)
plt.ylabel("Amplitude (V)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Prepare data for 3D plot
nodes = np.arange(1, V_amps.shape[1])  # Node indices 1 to 40
X, Y = np.meshgrid(frequencies / 1000, nodes)  # X: Freq (kHz), Y: Node

# Calculate ratios matrix (Nodes x Frequencies)
# V_amps[:, 1:] is (Freqs x Nodes), we need transpose for meshgrid (Nodes x Freqs)
Z = (V_amps[:, 1:] / V_amps[:, 0][:, None]).T

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)

ax.set_title("Amplitude Ratio Distribution vs Frequency and Node")
ax.set_xlabel("Frequency (kHz)")
ax.set_ylabel("Node Index")
ax.set_zlabel("Amplitude Ratio $|V_n / V_0|$")

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Ratio")
plt.savefig("figures/theory_system_random_3d.pdf")
plt.show()
