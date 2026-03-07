import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import os
import scipy.fft as fft
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d

simulation_on = True

time_start = time.time()


def H_func(V_in, V_out, t, z_in=None):
    dt = t[1] - t[0]
    if z_in is not None:
        z0 = z_in
    else:
        z0 = fft.fft(V_in)
    z40 = fft.fft(V_out)
    y0 = fft.fftfreq(len(V_in), d=dt)
    mask_pos = y0 > 0

    y0 = y0[mask_pos]
    z0 = z0[mask_pos]
    z40 = z40[mask_pos]

    # Calculate the Power of the input signal
    P_in = np.abs(z0) ** 2

    # Define the "Water Level" (epsilon)
    # A good starting point is 1% to 0.1% of the maximum input power
    # If the peak remains, increase this slightly. If true resonances vanish, decrease it.
    epsilon = np.max(P_in) * 1e-3

    # Apply Tikhonov Regularized Division
    H_raw = (z40 * np.conj(z0)) / (P_in + epsilon)
    # Define the frequencies to drop (in Hz)
    to_drop = np.arange(1, 51) * 50000

    # mask_bad represents the spectral nulls where the data is garbage
    mask_bad = np.any([np.isclose(y0, f, atol=600) for f in to_drop], axis=0)

    # mask_good represents the clean, reliable data
    mask_good = ~mask_bad

    # We interpolate the Real and Imaginary parts separately
    f_interp_real = interp1d(
        y0[mask_good], np.real(H_raw)[mask_good], kind="cubic", fill_value="extrapolate"
    )

    f_interp_imag = interp1d(
        y0[mask_good], np.imag(H_raw)[mask_good], kind="cubic", fill_value="extrapolate"
    )

    # 4. Construct the Final, Equally Spaced Array
    H_clean = np.empty_like(H_raw)

    # Keep the good data exactly as it is
    H_clean[mask_good] = H_raw[mask_good]

    # Overwrite the bad data with the smoothly interpolated values
    H_clean[mask_bad] = f_interp_real(y0[mask_bad]) + 1j * f_interp_imag(y0[mask_bad])

    # Return the clean, equally spaced complex transfer function and frequency axis
    return H_clean, y0


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
    R_L_0 = 0.730  # Nominal Inductor DCR (Ohms)
    dR_L = 0.1 * R_L_0
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

    # Capacitance Vector (N nodes)
    C = np.concatenate(
        (
            [np.random.normal(C_end, dC_end)],
            np.random.normal(C_0, dC, N - 2),
            [np.random.normal(C_end, dC_end)],
        )
    )

    # Inductance Vector (N-1 links)
    L = np.random.normal(L_0, dL, N - 1)

    # Random Parasitic Resistance Vector (N-1 links)
    R_L = np.random.normal(R_L_0, dR_L, N - 1)

    # 3. Build First-Order V-I State-Space Matrices
    # Total state variables: N voltages + (N-1) currents
    total_states = 2 * N - 1

    A = np.zeros((total_states, total_states))
    B = np.zeros(total_states)

    # Boundary condition: Input generator mapping
    B[0] = 1 / (R_in * C[0])
    A[0, 0] = -1 / (R_in * C[0])
    # A[0, N] = -1 / C[0]

    # Boundary condition: Output load mapping
    A[N - 1, N - 1] = -1 / (R_out * C[-1])
    A[N - 1, total_states - 1] = 1 / C[-1]

    # Bulk connections mapping
    for i in range(N - 1):
        # KCL (Voltages): Current I_i leaves node V_i and enters node V_{i+1}
        A[i, N + i] = -1 / C[i]
        A[i + 1, N + i] = 1 / C[i + 1]

        # KVL (Currents): Voltage drop V_i - V_{i+1} across L_i, minus R_L parasitic drop
        A[N + i, i] = 1 / L[i]
        A[N + i, i + 1] = -1 / L[i]
        A[N + i, N + i] = -R_L[i] / L[i]

    # Define the ODE function for solve_ivp
    def odefunc(t, Y):
        # Literal signal approximation (Trapezoidal Pulse)
        # The first-order system maps absolute V_in(t), not the derivative d_Vin(t)
        if t < tr:
            V_in_val = (pulse_height / tr) * t
        elif t < pulse_width:
            V_in_val = pulse_height
        elif t < pulse_width + tr:
            V_in_val = pulse_height - (pulse_height / tr) * (t - pulse_width)
        else:
            V_in_val = 0.0

        # State-space equation: dY/dt = A*Y + B*V_in
        return A @ Y + B * V_in_val

    # Initial conditions (zero voltage, zero current)
    Y0 = np.zeros(total_states)

    # 4. Integrate
    sol = solve_ivp(
        odefunc, (0, duration), Y0, method="RK45", max_step=tr, t_eval=t_eval_points
    )
    # 5. Extract Voltages and Add Noise
    # We only care about voltages (first N elements) for the transfer function
    V_clean = sol.y[:N, :]
    V_noisy = V_clean + np.random.normal(0, noise_std, V_clean.shape)

    # 6. Calculate Transfer Functions for Target Nodes
    target_nodes = np.arange(1, 11) * 4

    z_0_full = fft.fft(V_noisy[0, :])
    freq_data = None

    for node in target_nodes:
        H, freqs = H_func(V_noisy[0, :], V_noisy[node, :], t_eval_points, z_in=z_0_full)

        if freq_data is None:
            freq_data = {"Frequency (Hz)": freqs}

        # Store Magnitude and Phase (Standard practice for ML inputs)
        freq_data[f"H_Mag_{node}"] = np.abs(H)
        freq_data[f"H_Phase_{node}"] = np.angle(H)

    # 7. Exporting Data via Parquet
    output_dir = "data/inverse_problem/simulations_pulse"
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

    print(
        f"Simulation {next_idx} completed. Time elapsed: {time.time() - time_start:.2f} s"
    )

    if not simulation_on:
        exp_file = "data/electrical/pulses/FILTERED01.CSV"
        data_exp = pd.read_csv(exp_file)
        t_exp = data_exp.iloc[:, 0].to_numpy()
        v0_exp = data_exp.iloc[:, 1].to_numpy()
        v40_exp = data_exp.iloc[:, 2].to_numpy()
        mask_time = t_exp < (1e-3 - 1e-4)
        t_exp = t_exp[mask_time]
        v0_exp = v0_exp[mask_time]
        v40_exp = v40_exp[mask_time]

        H_exp, y0_exp = H_func(v0_exp, v40_exp, t_exp)

        plt.figure()
        plt.plot(t_exp, v40_exp, label="Experimental $V_0$")
        plt.plot(t_eval_points, V_clean[-1, :], label="Simulation $V_0$")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.legend()

        plt.figure()
        plt.plot(freqs / 1e3, np.abs(H), label="Simulation Transfer Function $|H(f)|$")
        plt.xlabel("Frequency (kHz)")
        plt.plot(
            y0_exp / 1e3,
            np.abs(H_exp),
            label="Experimental Transfer Function $|H(f)|$",
        )
        plt.ylabel("Magnitude $|V_{40}(f) / V_0(f)|$")
        plt.title("System Resonances (Transfer Function)")
        plt.xlim(0, 160)
        plt.ylim(0, 1)
        plt.legend()
        plt.show()
