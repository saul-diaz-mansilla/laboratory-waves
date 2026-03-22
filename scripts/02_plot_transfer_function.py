import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.electrical.signals as signals
import src.utils.io as io

# TODO: Swap hardcoded paths for YAML configs
input_parameters = io.load_config("config/input/gaussian_1.yaml")
f_start = input_parameters["f_start"]
f_end = input_parameters["f_end"]
num_inputs = input_parameters["num_inputs"]
sigma = input_parameters["sigma"]
frequencies = np.linspace(f_start, f_end, num_inputs)

exp_dir = "data/electrical/gaussian_2/"
exp_v0_all = []
exp_v40_all = []

for i in range(num_inputs):
    # Attempt to read both non-padded and zero-padded filenames to be safe
    file_path = os.path.join(exp_dir, f"FILTERED{i + 1}.CSV")
    if not os.path.exists(file_path):
        file_path = os.path.join(exp_dir, f"FILTERED{i + 1:02d}.CSV")

    if os.path.exists(file_path):
        data_exp = pd.read_csv(file_path)
        t_raw = data_exp.iloc[:, 0].to_numpy()
        v0_raw = data_exp.iloc[:, 1].to_numpy()
        v40_raw = data_exp.iloc[:, 2].to_numpy()

        exp_v0_all.append(v0_raw)
        exp_v40_all.append(v40_raw)
    else:
        print(f"Warning: Experimental file {file_path} not found.")

H_exp, y0_exp = signals.H_gaussian(exp_v0_all, exp_v40_all, t_raw, frequencies, sigma)

sim_dir = "data/temp/"
df_results = io.load_parquet_data(sim_dir, prefix="results_")
df_axes = io.load_parquet_data(sim_dir, prefix="freqs_")

freqs = df_axes["freqs_global"].iloc[0]

sim_index = 15
H_mag_node_40 = df_results["H_Mag_40"].iloc[sim_index]

# TODO: Automate plots using new io function
plt.figure()
plt.plot(freqs / 1e3, np.abs(H_mag_node_40), label="Simulation")
plt.plot(y0_exp / 1e3, np.abs(H_exp), label="Experimental data")
plt.xlabel("Frequency (kHz)")
plt.ylabel("$V_{40}(f) / V_0(f)$")
plt.legend()
plt.show()
