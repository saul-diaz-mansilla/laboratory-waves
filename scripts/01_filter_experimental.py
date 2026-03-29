import numpy as np
import pandas as pd
import scipy.signal as signal
import os
import sys
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.utils.data_io as io
import src.simulation.transfer_functions as signals


def main():
    # Set up argparse
    parser = argparse.ArgumentParser(description="Filter LC transmission line data.")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the master experimental configuration file. Located in configs/experiment",
    )
    args = parser.parse_args()
    config_path = args.config
    exp_parameters = io.load_config(config_path)
    input_path = exp_parameters["paths"]["raw_dir"]
    output_path = exp_parameters["paths"]["processed_dir"]
    metadata_path = exp_parameters["paths"]["metadata"]
    circuit_path = exp_parameters["paths"]["circuit_config"]

    circuit_parameters = io.load_config(circuit_path)
    df_metadata = pd.read_csv(metadata_path)

    waveform = exp_parameters["input"]["waveform"]

    f_cutoff = (
        1
        / np.sqrt(circuit_parameters["L"]["value"] * circuit_parameters["C"]["value"])
        / np.pi
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_exp_files = df_metadata[
        "filename"
    ].to_numpy()  # Contains raw filenames like 'AMPPUL00.CSV'

    for file_path in all_exp_files:
        file_name = Path(file_path).stem
        data = pd.read_csv(Path(input_path) / file_path)

        t = data.iloc[:, 0].to_numpy()
        v_0 = data.iloc[:, 1].to_numpy()
        v_node = data.iloc[:, 2].to_numpy()
        v_in = data.iloc[:, 3].to_numpy()

        dt = t[1] - t[0]
        fs = 1 / dt
        sos = signal.butter(4, 4 * f_cutoff, "lowpass", fs=fs, output="sos")
        v_0 = signal.sosfiltfilt(sos, v_0)
        v_node = signal.sosfiltfilt(sos, v_node)

        if waveform == "pulse" or waveform == "gaussian":
            v_0 = signals.remove_pulse_offset(v_0)
            v_node = signals.remove_pulse_offset(v_node)
            v_in = signals.remove_pulse_offset(v_in)
        else:
            v_0 = signals.remove_sine_offset(v_0)
            v_node = signals.remove_sine_offset(v_node)
            v_in = signals.remove_sine_offset(v_in)

        output_file_path = os.path.join(output_path, f"{file_name}.parquet")
        df_out = pd.DataFrame(
            np.column_stack([t, v_0, v_node, v_in]), columns=data.columns[:4]
        )
        df_out.to_parquet(output_file_path, engine="pyarrow")


if __name__ == "__main__":
    main()
