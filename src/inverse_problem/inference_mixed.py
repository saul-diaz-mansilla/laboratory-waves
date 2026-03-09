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


# --- 1. Simulation Helper Functions (Pulse Specific) ---
@njit
def compute_deriv_pulse(t, Y, A, B, pulse_width, pulse_height, tr):
    # Trapezoidal top-hat pulse logic
    if t < tr:
        V_in_val = pulse_height * (t / tr)
    elif t < pulse_width - tr:
        V_in_val = pulse_height
    elif t < pulse_width:
        V_in_val = pulse_height * (1 - (t - (pulse_width - tr)) / tr)
    else:
        V_in_val = 0.0

    return A @ Y + B * V_in_val


@njit
def rk4_solve_pulse(t_eval, Y0, A, B, pulse_width, pulse_height, tr):
    n_points = len(t_eval)
    n_states = len(Y0)
    Y_out = np.zeros((n_states, n_points))
    Y_curr = Y0.copy()
    Y_out[:, 0] = Y_curr
    dt = t_eval[1] - t_eval[0]

    for i in range(1, n_points):
        t = t_eval[i - 1]
        k1 = compute_deriv_pulse(t, Y_curr, A, B, pulse_width, pulse_height, tr)
        k2 = compute_deriv_pulse(
            t + dt / 2, Y_curr + dt / 2 * k1, A, B, pulse_width, pulse_height, tr
        )
        k3 = compute_deriv_pulse(
            t + dt / 2, Y_curr + dt / 2 * k2, A, B, pulse_width, pulse_height, tr
        )
        k4 = compute_deriv_pulse(
            t + dt, Y_curr + dt * k3, A, B, pulse_width, pulse_height, tr
        )
        Y_curr = Y_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        Y_out[:, i] = Y_curr
    return Y_out


def H_func_pulse(V_in_data, V_out_data, t_array, pulse_width, notch_width_hz=670.0):
    """
    Extracts H(f) and completely removes data points around the sinc singularities.
    """
    dt = t_array[1] - t_array[0]
    y0 = fft.fftfreq(len(V_in_data), d=dt)

    # 1. Base mask: Positive frequencies up to 150 kHz
    mask_pos = (y0 > 0) & (y0 < 150e3)

    # 2. Singularity notch mask
    f_zero = 1.0 / pulse_width
    num_singularities = int(150e3 // f_zero)

    valid_bins = np.ones_like(y0, dtype=bool)
    for n in range(1, num_singularities + 1):
        singularity = n * f_zero
        # Find frequencies within the exclusion zone
        in_notch = (y0 > singularity - notch_width_hz) & (
            y0 < singularity + notch_width_hz
        )
        valid_bins[in_notch] = False  # Punch a hole in the mask

    # 3. Combine masks
    final_mask = mask_pos & valid_bins

    y0_filtered = y0[final_mask]
    z_in = fft.fft(V_in_data)[final_mask]
    z_out = fft.fft(V_out_data)[final_mask]

    # H(f) calculation is now safe from near-zero division
    H_global = z_out / z_in

    return H_global, y0_filtered


# --- 2. Initialize Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "data/inverse_problem/models/transmission_line_2dcnn.pth"

L_0 = 330e-6
C_0 = 15e-9

# Pulse definitions
duration = 1e-3
pulse_width = 2e-5
pulse_height = 5.0
tr = 1e-7
t_eval_points = np.linspace(0, duration, int(duration / tr))

print("Loading trained 2D CNN model...")
model = TransferFunction2DCNN(num_outputs=82).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- 3. Load Pulse Experimental Data ---
print("Loading experimental pulse data...")
exp_dir = "data/electrical/pulses/"
target_nodes = np.arange(1, 11) * 4  # Nodes 4, 8, ..., 40

exp_v0_all = None
exp_vout_all = {}
t_exp_common = None

# Mapping {i}=01,...,10 to nodes 40 down to 4
# FILTERED01.CSV -> Node 40, FILTERED02.CSV -> Node 36, ..., FILTERED10.CSV -> Node 4
for node in target_nodes:
    file_idx = 11 - (node // 4)
    file_path = os.path.join(exp_dir, f"FILTERED{file_idx:02d}.CSV")

    if os.path.exists(file_path):
        data_exp = pd.read_csv(file_path)
        t_raw = data_exp.iloc[:, 0].to_numpy()

        if t_exp_common is None:
            t_exp_common = t_raw
            exp_v0_all = data_exp.iloc[:, 1].to_numpy()  # Assuming V_in is in col 1

        exp_vout_all[node] = data_exp.iloc[
            :, 2
        ].to_numpy()  # Assuming V_out is in col 2
    else:
        print(f"Warning: Experimental file {file_path} not found.")

# Calculate H(f) for all nodes
H_exp_dict = {}
y0_exp = None
for node in target_nodes:
    H_exp, y0_exp_curr = H_func_pulse(
        exp_v0_all, exp_vout_all[node], t_exp_common, pulse_width
    )
    H_exp_dict[node] = H_exp
    y0_exp = y0_exp_curr

# --- 4. Interpolate to Model Grid ---
num_sim_bins = 160
freq_sim_target = np.linspace(20e3, 140e3, num_sim_bins)

input_tensor_data = np.zeros((1, 20, num_sim_bins), dtype=np.float32)

print("Interpolating to 160 frequency bins...")
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
print(f"Mean Predicted L: {np.mean(L_pred_physical[:40]) * 1e6:.2f} µH")
print(f"Mean Predicted C: {np.mean(C_pred_physical) * 1e9:.2f} nF")
print(
    f"Std Predicted L: {np.std(L_pred_physical[:40]) / np.mean(L_pred_physical[:40]) * 100:.2f} %"
)
print(
    f"Std Predicted C: {np.std(C_pred_physical) / np.mean(C_pred_physical) * 100:.2f} %"
)

# --- 6. Forward Pulse Simulation Verification ---
print("\nRunning forward pulse simulation...")
N = 41
R_in = 150
R_L_0 = 0.730
R_L = np.full(N - 1, R_L_0)

total_states = 2 * N - 1
A = np.zeros((total_states, total_states))
B = np.zeros(total_states)

C_sim = C_pred_physical
L_sim = L_pred_physical[:40]

B[0] = 1 / (R_in * C_sim[0])
A[0, 0] = -1 / (R_in * C_sim[0])
A[N - 1, total_states - 1] = 1 / C_sim[-1]

for i in range(N - 1):
    A[i, N + i] = -1 / C_sim[i]
    A[i + 1, N + i] = 1 / C_sim[i + 1]
    A[N + i, i] = 1 / L_sim[i]
    A[N + i, i + 1] = -1 / L_sim[i]
    A[N + i, N + i] = -R_L[i] / L_sim[i]

# STATIC Impedance Matching for Pulse (Nominal Z0)
R_out_static = np.sqrt(L_0 / C_0)
A[N - 1, N - 1] = -1 / (R_out_static * C_sim[-1])

# Run the single RK4 integration
Y0 = np.zeros(total_states)
Y_all = rk4_solve_pulse(t_eval_points, Y0, A, B, pulse_width, pulse_height, tr)

V_in_sim = Y_all[0, :]
V_out_sim = Y_all[40, :]  # Final node 40

H_sim, y0_sim = H_func_pulse(V_in_sim, V_out_sim, t_eval_points, pulse_width)

# --- 7. Plotting Verification ---
plt.figure(figsize=(10, 6))
plt.plot(
    y0_exp / 1e3,
    np.abs(H_exp_dict[40]),
    label="Experimental Pulse |H(f)| (Node 40)",
    color="black",
    linestyle="--",
)
plt.plot(
    y0_sim / 1e3,
    np.abs(H_sim),
    label="Simulated Pulse |H(f)| (Predicted L&C)",
    color="red",
)
plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude")
plt.title("Experimental Pulse vs. NN Prediction Forward Simulation")
plt.xlim(0, 150)
plt.legend()
plt.grid(True)
plt.savefig("figures/inference_mixed_validation.png")
plt.show()
print("Validation complete. Plot saved to figures/inference_mixed_validation.png")
