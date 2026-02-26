import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import scipy.fft as fft

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import src.utils.landaubeta as lb

lb.use_latex_fonts()

freq = np.linspace(20000, 140000, 10)

i = 29
file_path = "data/electrical/gaussian_2/" + f"FILTERED{i + 1:02d}.CSV"
data = pd.read_csv(file_path)
t = data.iloc[:, 0].to_numpy()
v_0 = data.iloc[:, 1].to_numpy()
v_38 = data.iloc[:, 2].to_numpy()
v_in = data.iloc[:, 3].to_numpy()

dt = t[1] - t[0]

plt.figure()
plt.plot(t, v_0, label=r"$V_0$")
plt.plot(t, v_38, label=r"$V_{38}$")  # Changed from v38 data to V_38
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.show()

z_0 = fft.fft(v_0)
y_0 = fft.fftfreq(len(v_0), d=dt)
mask = y_0 > 0
y_0 = y_0[mask]
z_0 = z_0[mask]
z_38 = fft.fft(v_38)
y_38 = fft.fftfreq(len(v_38), d=dt)
mask = y_38 > 0
y_38 = y_38[mask]
z_38 = z_38[mask]

plt.figure()
plt.plot(y_0, np.abs(z_0) / len(v_0) * 2, label=r"$V_0$")
plt.plot(y_38, np.abs(z_38) / len(v_38) * 2, label=r"$V_{38}$")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 200000)
plt.legend()
plt.show()
