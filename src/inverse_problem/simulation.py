import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import os
import scipy.fft as fft
import matplotlib.pyplot as plt

simulation_on = False

if simulation_on:
    simulation_number = 10000
else:
    simulation_number = 1

for _ in range(simulation_number):
    # 1. Circuit Parameters
    L_0 = 330e-6
    dL = 0.2 * L_0
    C_0 = 15e-9
    dC = 0.1 * C_0
    C_end = 7.5e-9
    dC_end = dC / 2
    R_in = 150
    R_out = 130
    N = 41  # Nodes 0 to 40
    N_ind = N - 1  # 40 inductors

    # # Parasitic Guesses
    # R_L_0 = 0.0  # Nominal Inductor DCR (Ohms)
    # dR_L = 0.0 * R_L_0
    # G_C_0 = 0e-4  # Nominal Capacitor Shunt Conductance (1/Ohms)
    # dG_C = 0.0 * G_C_0
    noise_std = 1e-3  # Gaussian noise standard deviation (Volts)

    # Pulse Parameters
    duration = 1e-3  # Total duration of the waveform
    pulse_width = 2e-5
    pulse_height = 5.0
    tr = 1e-7
    t_eval_points = np.linspace(0, duration, int(duration / tr))
    dt = t_eval_points[1] - t_eval_points[0]

    # 2. Build the Matrices (Randomized)
    # Capacitance Vector
    C = np.concatenate(
        (
            [np.random.normal(C_end, dC_end)],
            np.random.normal(C_0, dC, N - 2),
            [np.random.normal(C_end, dC_end)],
        )
    )
    # Inductance Vector (N-1 links)
    L = np.random.normal(L_0, dL, N_ind)

    # # Random Parasitics
    # R_L = np.random.normal(R_L_0, dR_L, N_ind)
    # G_C = np.random.normal(G_C_0, dG_C, N)

    # 3. Build V-I State-Space Matrices
    # 2. Build the Matrices (Randomized)
    # Capacitance Vector
    C = np.concatenate(
        (
            [np.random.normal(C_end, dC_end)],
            np.random.normal(C_0, dC, N - 2),
            [np.random.normal(C_end, dC_end)],
        )
    )
    Mc = np.diag(C)
    Mc_inv = np.linalg.inv(Mc)

    # Resistance Matrix (Mr)
    Mr = np.zeros((N, N))
    Mr[0, 0] = 1 / R_in
    Mr[-1, -1] = 1 / R_out

    # Inductance Vector (N-1 links)
    L = np.random.normal(L_0, dL, N - 1)
    Ml = np.zeros((N, N))
    Ml[0, 0] = 1 / L[0]
    Ml[0, 1] = -1 / L[0]
    for i in range(1, N - 1):
        Ml[i, i - 1] = -1 / L[i - 1]
        Ml[i, i] = 1 / L[i - 1] + 1 / L[i]
        Ml[i, i + 1] = -1 / L[i]
    Ml[-1, -2] = -1 / L[-1]
    Ml[-1, -1] = 1 / L[-1]

    # Isolate the second derivative (create A_R and A_L)
    Mc_inv = np.linalg.inv(Mc)
    A_R = Mc_inv @ Mr
    A_L = Mc_inv @ Ml

    # Define the ODE function for solve_ivp
    def odefunc(t, Y):
        V = Y[:N]
        V_dot = Y[N:]

        # Calculate the forcing vector F(t)
        F = np.zeros(N)

        # Literal signal approximation
        if t < tr:
            d_Vin = pulse_height / tr
        elif t >= pulse_width and t < pulse_width + tr:
            d_Vin = -pulse_height / tr
        else:
            d_Vin = 0.0

        F[0] = (1 / R_in) * d_Vin

        # State-space equations
        V_ddot = -A_R @ V_dot - A_L @ V + Mc_inv @ F

        return np.concatenate((V_dot, V_ddot))

    # Initial conditions (zero voltage, zero current)
    Y0 = np.zeros(2 * N)

    # 4. Integrate
    sol = solve_ivp(
        odefunc,
        (0, duration),
        Y0,
        method="RK45",
        max_step=tr / 10,
        t_eval=t_eval_points,
    )

    # 5. Extract Voltages and Add Noise
    # We only care about voltages (first N elements) for the transfer function
    V_clean = sol.y[:N, :]
    V_noisy = V_clean + np.random.normal(0, noise_std, V_clean.shape)

    # 6. Calculate Transfer Functions for Target Nodes
    target_nodes = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]

    z_0 = fft.fft(V_noisy[0, :])
    freqs = fft.fftfreq(len(t_eval_points), d=dt)
    mask = (freqs > 0) & (
        freqs < 160e3
    )  # Only keep physical positive frequencies up to 200kHz

    freq_data = {"Frequency (Hz)": freqs[mask]}

    for node in target_nodes:
        z_node = fft.fft(V_noisy[node, :])
        # Compute complex transfer function
        H = z_node[mask] / (z_0[mask] + 1e-10)

        # Store Magnitude and Phase (Standard practice for ML inputs)
        freq_data[f"H_Mag_{node}"] = np.abs(H)
        freq_data[f"H_Phase_{node}"] = np.angle(H)

    # 7. Exporting Data via Parquet
    output_dir = "data/temp"
    os.makedirs(output_dir, exist_ok=True)

    existing_indices = []
    for f in os.listdir(output_dir):
        if f.startswith("freq_data_") and f.endswith(".parquet"):
            existing_indices.append(int(f[10:-8]))

    next_idx = max(existing_indices) + 1 if existing_indices else 1

    # Save frequency features
    df_freq = pd.DataFrame(freq_data)
    df_freq.to_parquet(
        os.path.join(output_dir, f"freq_data_{next_idx}.parquet"), engine="pyarrow"
    )

    # Save targets (L, C, and parasitics)
    target_data = {
        "C_norm": C / C_0,
        "L_norm": np.pad(
            L / L_0, (0, 1), constant_values=0
        ),  # Pad to length 41 for easy column matching
    }
    df_targets = pd.DataFrame(target_data)
    df_targets.to_parquet(
        os.path.join(output_dir, f"targets_{next_idx}.parquet"), engine="pyarrow"
    )

    print(f"Simulation {next_idx} completed and saved using Parquet.")

    if not simulation_on:
        exp_file = "data/electrical/pulses/FILTERED01.CSV"
        data_exp = pd.read_csv(exp_file)
        t_exp = data_exp.iloc[:, 0].to_numpy()
        v0_exp = data_exp.iloc[:, 1].to_numpy()
        v40_exp = data_exp.iloc[:, 2].to_numpy()

        dt_exp = t_exp[1] - t_exp[0]
        z0_exp = fft.fft(v0_exp)
        z40_exp = fft.fft(v40_exp)
        y0_exp = fft.fftfreq(len(v0_exp), d=dt_exp)
        mask_exp = y0_exp > 0

        H_exp = np.abs(z40_exp[mask_exp]) / (np.abs(z0_exp[mask_exp]) + 1e-10)

        plt.figure()
        plt.plot(t_exp, v40_exp, label="Experimental $V_0$")
        plt.plot(t_eval_points, V_clean[-1, :], label="Simulation $V_0$")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.legend()

        plt.figure()
        plt.plot(
            freqs[mask] / 1e3, np.abs(H), label="Simulation Transfer Function $|H(f)|$"
        )
        plt.xlabel("Frequency (kHz)")
        plt.plot(
            y0_exp[mask_exp] / 1e3,
            H_exp,
            label="Experimental Transfer Function $|H(f)|$",
        )
        plt.ylabel("Magnitude $|V_{40}(f) / V_0(f)|$")
        plt.title("System Resonances (Transfer Function)")
        plt.xlim(0, 160)
        plt.ylim(0, 10)
        plt.legend()
        plt.show()
