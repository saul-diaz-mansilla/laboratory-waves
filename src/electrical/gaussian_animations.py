import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import sys
import scipy.fft as fft

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import src.utils.landaubeta as lb

lb.use_latex_fonts()

# 1. Circuit Parameters
L = 330e-6
C = 15e-9
C_end = 7.5e-9
R_in = 150
V_in = 2.5
N = 41  # Nodes 0 to 40
f_c = (2 / np.sqrt(L * C)) / (2 * np.pi)

NUM_FILES = 30

# --- Animation 1: Time Domain Signals ---

fig1, ax1 = plt.subplots()
(line1,) = ax1.plot([], [], lw=2, label=r"$V_0$")
(line2,) = ax1.plot([], [], lw=2, label=r"$V_{38}$")


def init_time():
    """Initializes the time-domain plot."""
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Voltage (V)")
    ax1.legend()
    ax1.grid(True)
    # Read the first file to set axis limits
    data = pd.read_csv("data/electrical/gaussian_2/AMPPUL01.CSV")
    t = data.iloc[:, 0].to_numpy()
    ax1.set_xlim(t.min(), t.max())
    ax1.set_ylim(-1.5, 1.5)
    return (line1, line2)


def update_time(i):
    """Updates the time-domain plot for frame i."""
    file_path = f"data/electrical/gaussian_2/FILTERED{i + 1:02d}.CSV"
    data = pd.read_csv(file_path)
    t = data.iloc[:, 0].to_numpy()
    v_0 = data.iloc[:, 1].to_numpy()
    v_38 = data.iloc[:, 2].to_numpy()

    line1.set_data(t, v_0)
    line2.set_data(t, v_38)
    ax1.set_title(f"Time Domain Signal for File {i + 1:02d}")
    return (line1, line2)


ani_time = animation.FuncAnimation(
    fig1, update_time, frames=NUM_FILES, init_func=init_time, blit=True, interval=200
)

# --- Animation 2: Fourier Transforms ---

fig2, ax2 = plt.subplots()
(fft_line1,) = ax2.plot([], [], lw=2, label=r"FFT($V_0$)")
(fft_line2,) = ax2.plot([], [], lw=2, label=r"FFT($V_{38}$)")


def init_fft():
    """Initializes the frequency-domain plot."""
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Amplitude (V)")
    ax2.set_xlim(0, 200000)
    ax2.set_ylim(0, 0.2)
    # ax2.axvline(f_c, color="r", linestyle="--", label=r"$f_c$")
    ax2.legend()
    ax2.grid(True)
    return (fft_line1, fft_line2)


def update_fft(i):
    """Updates the frequency-domain plot for frame i."""
    file_path = f"data/electrical/gaussian_2/FILTERED{i + 1:02d}.CSV"
    # This part is duplicated from update_time but is necessary for the animation
    data = pd.read_csv(file_path)
    t = data.iloc[:, 0].to_numpy()
    v_0 = data.iloc[:, 1].to_numpy()
    v_38 = data.iloc[:, 2].to_numpy()
    dt = t[1] - t[0]

    # FFT for v_0
    N0 = len(v_0)
    yf0 = fft.fft(v_0)
    xf0 = fft.fftfreq(N0, dt)
    mask0 = (xf0 > 0) & (xf0 < 200000)
    fft_line1.set_data(xf0[mask0], 2.0 / N0 * np.abs(yf0[mask0]))

    # FFT for v_38
    N38 = len(v_38)
    yf38 = fft.fft(v_38)
    xf38 = fft.fftfreq(N38, dt)
    mask38 = (xf38 > 0) & (xf38 < 200000)
    fft_line2.set_data(xf38[mask38], 2.0 / N38 * np.abs(yf38[mask38]))

    ax2.set_title(f"FFT for File {i + 1:02d}")
    return (fft_line1, fft_line2)


ani_fft = animation.FuncAnimation(
    fig2, update_fft, frames=NUM_FILES, init_func=init_fft, blit=True, interval=200
)

plt.show()
