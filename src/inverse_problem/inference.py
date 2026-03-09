import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy.fft as fft
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import njit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.inverse_problem.model import TransferFunction2DCNN


# --- 1. Simulation Helper Functions (Copied to avoid global execution of simulation_numba.py) ---
@njit
def compute_deriv(t, Y, A, B, amplitude, t0, sigma, f_c):
    V_in_val = (
        amplitude
        * np.exp(-((t - t0) ** 2) / (2 * sigma**2))
        * np.cos(2 * np.pi * f_c * t)
    )
    return A @ Y + B * V_in_val


@njit
def rk4_solve(t_eval, Y0, A, B, amplitude, t0, sigma, f_c):
    n_points = len(t_eval)
    n_states = len(Y0)
    Y_out = np.zeros((n_states, n_points))
    Y_curr = Y0.copy()
    Y_out[:, 0] = Y_curr
    dt = t_eval[1] - t_eval[0]

    for i in range(1, n_points):
        t = t_eval[i - 1]
        k1 = compute_deriv(t, Y_curr, A, B, amplitude, t0, sigma, f_c)
        k2 = compute_deriv(
            t + dt / 2, Y_curr + dt / 2 * k1, A, B, amplitude, t0, sigma, f_c
        )
        k3 = compute_deriv(
            t + dt / 2, Y_curr + dt / 2 * k2, A, B, amplitude, t0, sigma, f_c
        )
        k4 = compute_deriv(t + dt, Y_curr + dt * k3, A, B, amplitude, t0, sigma, f_c)
        Y_curr = Y_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        Y_out[:, i] = Y_curr
    return Y_out


def H_func_global(V_in_data, V_out_data, t_array, freqs, t_std):
    dt = t_array[1] - t_array[0]
    f_std_2 = 1.0 / (2 * np.pi * t_std) / 2
    if not isinstance(V_in_data, list):
        V_in_data = [V_in_data]
        V_out_data = [V_out_data]

    y0 = fft.fftfreq(len(V_in_data[0]), d=dt)
    mask_pos = (y0 > (freqs[0] - f_std_2)) & (y0 < 150e3)
    y0 = y0[mask_pos]

    S_xx = np.zeros(len(y0))
    S_xy = np.zeros(len(y0), dtype=complex)
    coverage_mask = np.zeros(len(y0), dtype=bool)

    for v_in, v_out, f_mean in zip(V_in_data, V_out_data, freqs):
        z_in = fft.fft(v_in)[mask_pos]
        z_out = fft.fft(v_out)[mask_pos]
        P_in = np.abs(z_in) ** 2
        valid_bins = (y0 >= f_mean - f_std_2) & (y0 <= f_mean + f_std_2)
        S_xx[valid_bins] += P_in[valid_bins]
        S_xy[valid_bins] += (z_out * np.conj(z_in))[valid_bins]
        coverage_mask[valid_bins] = True

    H_global = np.zeros(len(y0), dtype=complex)
    H_global[coverage_mask] = S_xy[coverage_mask] / (S_xx[coverage_mask])
    return H_global, y0


# --- 2. Initialize Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/transmission_line_2dcnn.pth"

# Nominal values
L_0 = 330e-6
C_0 = 15e-9

print("Loading trained 2D CNN model...")
model = TransferFunction2DCNN(num_outputs=82).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- 3. Load Experimental Data ---
print("Loading experimental data...")
exp_dir = "data/electrical/gaussian_2/"
frequencies = np.linspace(20000, 140000, 30)
sigma = 0.00002

exp_v0_all = []
# Expecting all 10 nodes for the 2D CNN
target_nodes = np.arange(1, 11) * 4
exp_vout_all = {node: [] for node in target_nodes}
t_exp_common = None

for i in range(1, 31):
    file_path = os.path.join(exp_dir, f"FILTERED{i}.CSV")
    if not os.path.exists(file_path):
        file_path = os.path.join(exp_dir, f"FILTERED{i:02d}.CSV")

    if os.path.exists(file_path):
        data_exp = pd.read_csv(file_path)
        t_raw = data_exp.iloc[:, 0].to_numpy()
        mask_time = t_raw < (1e-3 - 1e-4)

        if t_exp_common is None:
            t_exp_common = t_raw[mask_time]

        exp_v0_all.append(data_exp.iloc[:, 1].to_numpy()[mask_time])

        # CRITICAL ASSUMPTION: Columns 2 through 11 contain Nodes 4, 8, ... 40
        for col_idx, node in enumerate(target_nodes):
            exp_vout_all[node].append(
                data_exp.iloc[:, col_idx + 2].to_numpy()[mask_time]
            )
    else:
        print(f"Warning: Experimental file {file_path} not found.")

# Calculate H(f) for all experimental nodes
H_exp_dict = {}
for node in target_nodes:
    H_exp, y0_exp = H_func_global(
        exp_v0_all, exp_vout_all[node], t_exp_common, frequencies, sigma
    )
    H_exp_dict[node] = H_exp

# --- 4. Interpolate Experimental Data to NN Grid ---
num_sim_bins = 160
# This must match exactly the frequency bounds your training target generated
freq_sim_target = np.linspace(y0_exp.min(), y0_exp.max(), num_sim_bins)

input_tensor_data = np.zeros((1, 20, num_sim_bins), dtype=np.float32)

print("Interpolating experimental data to model grid...")
for i, node in enumerate(target_nodes):
    H_mag = np.abs(H_exp_dict[node])
    H_phase = np.angle(H_exp_dict[node])

    f_mag = interp1d(
        y0_exp, H_mag, kind="cubic", bounds_error=False, fill_value="extrapolate"
    )
    f_phase = interp1d(
        y0_exp, H_phase, kind="cubic", bounds_error=False, fill_value="extrapolate"
    )

    input_tensor_data[0, 2 * i] = f_mag(freq_sim_target)
    input_tensor_data[0, 2 * i + 1] = f_phase(freq_sim_target)

input_tensor = torch.tensor(input_tensor_data).to(device)

# --- 5. Run Inference ---
print("Running neural network inference...")
with torch.no_grad():
    predictions = model(input_tensor)

predictions = predictions.cpu().numpy().flatten()
C_pred_physical = predictions[:41] * C_0
L_pred_physical = predictions[41:82] * L_0

print("\n--- Inference Results ---")
print(
    f"Mean Predicted L: {np.mean(L_pred_physical[:40]) * 1e6:.2f} µH (Nominal: 330 µH)"
)
print(f"Mean Predicted C: {np.mean(C_pred_physical) * 1e9:.2f} nF (Nominal: 15 nF)")

# --- 6. Forward Simulation with Predicted Components ---
print("\nRunning forward simulation to verify predictions...")
N = 41
R_in = 150
R_L_0 = 0.730
R_L = np.full(N - 1, R_L_0)  # Assuming nominal for nuisance params
power_rule_0 = 1.25
f_test, Q_test = 796e3, 65
R_ratio_test = 2 * np.pi * f_test * L_0 / Q_test / R_L_0
k_power = (R_ratio_test - 1) / f_test**power_rule_0

omega_c = 2.0 / np.sqrt(L_0 * C_0)
total_states = 2 * N - 1
A = np.zeros((total_states, total_states))
B = np.zeros(total_states)

C_sim = C_pred_physical
L_sim = L_pred_physical[:40]  # Drop the zero-padding at index 40

B[0] = 1 / (R_in * C_sim[0])
A[0, 0] = -1 / (R_in * C_sim[0])
A[N - 1, total_states - 1] = 1 / C_sim[-1]

for i in range(N - 1):
    A[i, N + i] = -1 / C_sim[i]
    A[i + 1, N + i] = 1 / C_sim[i + 1]
    A[N + i, i] = 1 / L_sim[i]
    A[N + i, i + 1] = -1 / L_sim[i]
    A[N + i, N + i] = -R_L[i] / L_sim[i]

V_in_sim_runs = []
V_out_sim_runs = []

duration = 1.0 / 500.0
t0 = duration / 2
amplitude = 5.0 / 2.0
num_points = 10000
t_eval_points = np.linspace(0, duration, num_points, endpoint=False)

for f_c in frequencies:
    omega = 2.0 * np.pi * f_c
    if omega >= omega_c:
        R_out = 1e6
    else:
        R_out = np.sqrt(L_0 / C_0) / np.sqrt(1 - (omega / omega_c) ** 2)

    A[N - 1, N - 1] = -1 / (R_out * C_sim[-1])
    R_L_AC = R_L * (1 + k_power * f_c**power_rule_0)
    for i in range(N - 1):
        A[N + i, N + i] = -R_L_AC[i] / L_sim[i]

    Y0 = np.zeros(total_states)
    Y_all = rk4_solve(t_eval_points, Y0, A, B, amplitude, t0, sigma, f_c)

    V_in_sim_runs.append(Y_all[0, :])
    V_out_sim_runs.append(Y_all[40, :])  # Final node 40

H_sim, y0_sim = H_func_global(
    V_in_sim_runs, V_out_sim_runs, t_eval_points, frequencies, sigma
)

# --- 7. Plotting Verification ---
plt.figure(figsize=(10, 6))
plt.plot(
    y0_exp / 1e3,
    np.abs(H_exp_dict[40]),
    label="Experimental $|H(f)|$ (Target)",
    color="black",
    linestyle="--",
)
plt.plot(
    y0_sim / 1e3, np.abs(H_sim), label="Simulated $|H(f)|$ (Predicted L&C)", color="red"
)
plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude $|V_{40}(f) / V_0(f)|$")
plt.title("Experimental vs. Neural Network Forward Simulation")
plt.xlim(0, 150)
plt.legend()
plt.grid(True)
plt.savefig("data/inverse_problem/inference_validation.png")
print(
    "Validation complete. Plot saved to data/inverse_problem/inference_validation.png"
)
