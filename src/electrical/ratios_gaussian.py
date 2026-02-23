import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import src.utils.landaubeta as lb

lb.use_latex_fonts()

freq = np.linspace(20000, 140000, 10)

ratio = []
for i in range(10):
    file_path = "data/electrical/gaussian/" + f"AMPPUL{i + 1:02d}.CSV"
    data = pd.read_csv(file_path)
    t = data.iloc[:, 0].to_numpy()
    v_0 = data.iloc[:, 1].to_numpy()
    v_38 = data.iloc[:, 2].to_numpy()
    v_in = data.iloc[:, 3].to_numpy()
    ratio.append(np.max(v_38) / np.max(v_0))

ratio_res = []

for i in range(10):
    if i < 4:
        ratio_res.append(np.nan)
    else:
        file_path = "data/electrical/gaussian_res/" + f"AMPPUL{i + 1:02d}.CSV"
        data = pd.read_csv(file_path)
        t = data.iloc[:, 0].to_numpy()
        v_0 = data.iloc[:, 1].to_numpy()
        v_38 = data.iloc[:, 2].to_numpy()
        v_in = data.iloc[:, 3].to_numpy()
        ratio_res.append(np.max(v_38) / np.max(v_0))

plt.figure()
plt.plot(freq / 1e3, ratio, label=r"Constant resistance $\sqrt{L/C}$")
plt.plot(freq / 1e3, ratio_res, label="Matched impedance")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Ratio")
plt.legend()
plt.savefig("figures/ratios_gaussian.pdf")
plt.show()
