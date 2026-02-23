import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Length of RL circuits & transmission line
l_t = 38  # sections
l_0 = 1  # sections

# Inductance and capacitance
L = 330e-6  # H / section
dL = 0.2 * L
C = 0.015e-6  # F / section
dC = 0.1 * C

# Cutoff frequency
w_c = 2 / np.sqrt(L * C)


i = 1
file_path = "data/electrical/gaussian/" + f"AMPPUL{i:02d}.CSV"
data = pd.read_csv(file_path)
t = data.iloc[:, 0].to_numpy()
v_0 = data.iloc[:, 1].to_numpy()
v_38 = data.iloc[:, 2].to_numpy()
v_in = data.iloc[:, 3].to_numpy()

plt.figure()
plt.plot(t, v_0, label="v0 data")
plt.plot(t, v_38, label="v38 data")
plt.legend()
plt.show()
