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

from src.inverse_problem.model_1d import TransferFunction1DCNN


# --- 1. Simulation Helper Functions ---
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


# --- 2. Initialize 1D Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "data/inverse_problem/models/transmission_line_1dcnn_node40.pth"

L_0 = 330e-6
C_0 = 15e-9

print("Loading trained 1D CNN model...")
model = TransferFunction1DCNN(in_channels=2, num_outputs=82).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- 3. Load Experimental Data (Node 40 Only) ---
print("Loading experimental data...")
exp_dir = "data/electrical/gaussian_2/"
frequencies = np.linspace(20000, 140000, 30)
sigma = 0.00002

exp_v0_all = []
exp_v40_all = []
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
        exp_v40_all.append(data_exp.iloc[:, 2].to_numpy()[mask_time])

# Calculate H(f) for Node 40
H_exp, y0_exp = H_func_global(exp_v0_all, exp_v40_all, t_exp_common, frequencies, sigma)

# --- 4. Interpolate to 160 Bins ---
num_sim_bins = 160
freq_sim_target = np.linspace(y0_exp.min(), y0_exp.max(), num_sim_bins)

# Shape: (Batch, 2 channels, 160 freq_bins)
input_tensor_data = np.zeros((1, 2, num_sim_bins), dtype=np.float32)

print("Interpolating Node 40 experimental data...")
f_mag = interp1d(
    y0_exp, np.abs(H_exp), kind="cubic", bounds_error=False, fill_value="extrapolate"
)
f_phase = interp1d(
    y0_exp, np.angle(H_exp), kind="cubic", bounds_error=False, fill_value="extrapolate"
)

input_tensor_data[0, 0] = f_mag(freq_sim_target)
input_tensor_data[0, 1] = f_phase(freq_sim_target)

input_tensor = torch.tensor(input_tensor_data).to(device)

# --- 5. Run Inference ---
print("Running neural network inference...")
with torch.no_grad():
    predictions = model(input_tensor)

predictions = predictions.cpu().numpy().flatten()
C_pred_physical = predictions[:41] * C_0
L_pred_physical = predictions[41:82] * L_0

print("\n--- Inference Results ---")
print(f"Mean Predicted L: {np.mean(L_pred_physical[:40]) * 1e6:.2f} µH")
print(f"Mean Predicted C: {np.mean(C_pred_physical) * 1e9:.2f} nF")
print(
    f"Std Predicted L: {np.std(L_pred_physical[:40]) / np.mean(L_pred_physical[:40]) * 100:.2f} %"
)
print(
    f"Std Predicted C: {np.std(C_pred_physical) / np.mean(C_pred_physical) * 100:.2f} %"
)

# --- 6. Forward Simulation ---
print("\nRunning forward verification simulation...")
N = 41
R_in, R_L_0, power_rule_0 = 150, 0.730, 1.25
f_test, Q_test = 796e3, 65
R_ratio_test = 2 * np.pi * f_test * L_0 / Q_test / R_L_0
k_power = (R_ratio_test - 1) / f_test**power_rule_0

omega_c = 2.0 / np.sqrt(L_0 * C_0)
total_states = 2 * N - 1
A = np.zeros((total_states, total_states))
B = np.zeros(total_states)

C_sim = C_pred_physical
L_sim = L_pred_physical[:40]
R_L = np.full(N - 1, R_L_0)

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
t_eval_points = np.linspace(0, duration, 10000, endpoint=False)

for f_c in frequencies:
    omega = 2.0 * np.pi * f_c
    R_out = (
        1e6
        if omega >= omega_c
        else np.sqrt(L_0 / C_0) / np.sqrt(1 - (omega / omega_c) ** 2)
    )

    A[N - 1, N - 1] = -1 / (R_out * C_sim[-1])
    R_L_AC = R_L * (1 + k_power * f_c**power_rule_0)
    for i in range(N - 1):
        A[N + i, N + i] = -R_L_AC[i] / L_sim[i]

    Y_all = rk4_solve(
        t_eval_points, np.zeros(total_states), A, B, amplitude, t0, sigma, f_c
    )

    V_in_sim_runs.append(Y_all[0, :])
    V_out_sim_runs.append(Y_all[40, :])

H_sim, y0_sim = H_func_global(
    V_in_sim_runs, V_out_sim_runs, t_eval_points, frequencies, sigma
)

# --- 7. Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(
    freq_sim_target / 1e3,
    f_mag(freq_sim_target),
    label="Experimental $|V_{40}(f) / V_0(f)|$",
    color="black",
    linestyle="--",
)
plt.plot(y0_sim / 1e3, np.abs(H_sim), label="Simulated (1D CNN Predicted)", color="red")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude")
plt.title("Experimental vs. 1D CNN Prediction Verification")
plt.xlim(0, 150)
plt.legend()
plt.grid(True)
plt.savefig("figures/inference_validation_1d.png")
plt.show()
print("Validation complete. Plot saved to data/figures/inference_validation_1d.png")
