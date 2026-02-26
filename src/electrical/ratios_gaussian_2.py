import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import src.utils.landaubeta as lb

lb.use_latex_fonts()

sections = 38

freq = np.linspace(20000, 140000, 30)

ratio = []
v = []
for i in range(30):
    file_path = "data/electrical/gaussian_2/" + f"FILTERED{i + 1:02d}.CSV"
    data = pd.read_csv(file_path)
    t = data.iloc[:, 0].to_numpy()
    v_0 = data.iloc[:, 1].to_numpy()
    v_38 = data.iloc[:, 2].to_numpy()
    v_in = data.iloc[:, 3].to_numpy()
    ratio.append(np.max(v_38) / np.max(v_0))
    idx_1 = np.argmax(v_0)
    idx_2 = np.argmax(v_38)
    v.append(sections / np.abs(t[idx_1] - t[idx_2]))

plt.figure()
plt.plot(freq / 1e3, ratio)
plt.xlabel("Frequency (kHz)")
plt.ylabel("Ratio")
# plt.savefig("figures/ratios_gaussian.pdf")

plt.figure()
plt.plot(freq / 1e3, v)
plt.xlabel("Frequency (kHz)")
plt.ylabel("Velocity (m/s)")
# plt.savefig("figures/velocity_gaussian.pdf")
plt.show()
