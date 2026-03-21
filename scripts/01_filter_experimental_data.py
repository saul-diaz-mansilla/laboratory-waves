import numpy as np
import pandas as pd
import scipy.signal as signal
import os
import argparse

# Set up argparse
parser = argparse.ArgumentParser(description="Filter LC transmission line data.")

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to the raw data files",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path to the output directory",
)
parser.add_argument(
    "--pulse",
    type=str,
    required=True,
    help="True to subtract the mean correctly for a pulse, False otherwise",
)
args = parser.parse_args()
input_path = args.input
output_path = args.output
pulse_subtract = args.pulse

if not os.path.exists(output_path):
    os.makedirs(output_path)

pulse_subtract = bool(pulse_subtract)

if pulse_subtract is not True and pulse_subtract is not False:
    raise ValueError(
        f"Invalid value for pulse: {pulse_subtract}. Must be True or False."
    )


for i in range(10):
    file_path = input_path + f"AMPPUL{i + 1:02d}.CSV"
    data = pd.read_csv(file_path)
    t = data.iloc[:, 0].to_numpy()
    v_0 = data.iloc[:, 1].to_numpy()
    v_38 = data.iloc[:, 2].to_numpy()
    v_in = data.iloc[:, 3].to_numpy()

    dt = t[1] - t[0]
    fs = 1 / dt
    sos = signal.butter(4, 500000, "lowpass", fs=fs, output="sos")
    v_0 = signal.sosfiltfilt(sos, v_0)
    v_38 = signal.sosfiltfilt(sos, v_38)

    if pulse_subtract:
        v_0 = v_0 - np.mean(v_0[len(v_0) // 2 :])
        v_38 = v_38 - np.mean(v_38[len(v_38) // 2 :])

    else:
        v_0 = v_0 - np.mean(v_0)
        v_38 = v_38 - np.mean(v_38)

    output_file_path = output_path + f"FILTERED{i + 1:02d}.CSV"
    df_out = pd.DataFrame(
        np.column_stack([t, v_0, v_38, v_in]), columns=data.columns[:4]
    )
    df_out.to_csv(output_file_path, index=False)
    print(f"Saved {output_file_path}")
