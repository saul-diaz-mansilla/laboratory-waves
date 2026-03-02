import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.fft as fft
import os
import sys
import scipy.signal as signal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import src.utils.landaubeta as lb

lb.use_latex_fonts()

# Peak height thershold
percent_threshold = 0.2

# Pulse data
exp_file = "data/electrical/pulses/FILTERED01.CSV"
data_exp = pd.read_csv(exp_file)
t_exp = data_exp.iloc[:, 0].to_numpy()
v0_exp = data_exp.iloc[:, 1].to_numpy()
v38_exp = data_exp.iloc[:, 2].to_numpy()

dt_exp = t_exp[1] - t_exp[0]
z0_exp = fft.fft(v0_exp)
y0_exp = fft.fftfreq(len(v0_exp), d=dt_exp)
mask_exp = y0_exp > 0

z_0_exp = np.abs(z0_exp[mask_exp]) / len(v0_exp)
sinc_exp = np.abs(np.sinc(y0_exp[mask_exp] * 2e-5)) * np.max(z_0_exp)
# sinc_exp = np.abs(np.sinc(y0_exp[mask_exp] * 2e-5)) * 2e-5 * np.max(v0_exp) / 1e-3
deviation = z_0_exp - sinc_exp


file_indices = []
peak_freqs = []

# Gaussian wavepacket data
for i in range(30):
    file_path = f"data/electrical/gaussian_2/FILTERED{i + 1:02d}.CSV"
    # This part is duplicated from update_time but is necessary for the animation
    data = pd.read_csv(file_path)
    t = data.iloc[:, 0].to_numpy()
    v_0 = data.iloc[:, 1].to_numpy()
    dt = t[1] - t[0]

    # FFT for v_0
    N0 = len(v_0)
    yf0 = fft.fft(v_0)
    xf0 = fft.fftfreq(N0, dt)
    mask0 = (xf0 > 0) & (xf0 < 200000)

    # Find peaks for V0
    magnitudes = np.abs(yf0[mask0])
    peaks, _ = signal.find_peaks(
        magnitudes, height=percent_threshold * np.max(magnitudes)
    )

    for p in peaks:
        if yf0[mask0][p] != np.max(yf0[mask0]):
            file_indices.append(i + 1)
            peak_freqs.append(xf0[mask0][p])

mask_freq = (y0_exp > 0) & (y0_exp < 200000)
mask_limit = y0_exp[mask_exp] < 200000
deviation_limited = deviation[mask_limit]
magnitudes_exp = np.abs(deviation_limited - np.min(deviation_limited))
peaks_exp, _ = signal.find_peaks(
    magnitudes_exp, height=percent_threshold * np.max(magnitudes_exp)
)

exp_freqs = y0_exp[mask_freq][peaks_exp]

plt.figure()
plt.plot(np.array(peak_freqs) / 1000, file_indices, "o", label=r"$V_0$ Peaks")

for i, freq in enumerate(exp_freqs):
    plt.axvline(
        x=freq / 1000,
        color="tab:orange",
        linestyle="--",
        label="Pulse Peaks" if i == 0 else None,
    )

plt.ylabel("File Index")
plt.xlabel("Frequency (kHz)")
plt.legend()
plt.grid(True)
plt.show()
