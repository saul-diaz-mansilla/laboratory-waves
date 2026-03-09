import numpy as np
import pandas as pd
import scipy.signal as signal

for i in range(10):
    file_path = "data/electrical/gaussian/" + f"AMPPUL{i + 1:02d}.CSV"
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

    # v_0 = v_0 - np.mean(v_0)
    # v_38 = v_38 - np.mean(v_38)

    v_0 = v_0 - np.mean(v_0[len(v_0) // 2 :])
    v_38 = v_38 - np.mean(v_38[len(v_38) // 2 :])

    output_path = "data/electrical/gaussian/" + f"FILTERED{i + 1:02d}.CSV"
    df_out = pd.DataFrame(
        np.column_stack([t, v_0, v_38, v_in]), columns=data.columns[:4]
    )
    df_out.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
