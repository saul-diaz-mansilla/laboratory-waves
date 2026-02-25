import numpy as np
from scipy.integrate import solve_ivp
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# 1. Circuit Parameters
L_0 = 330e-6
dL = 0.2 * L_0
C_0 = 15e-9
dC = 0.1 * C_0
C_end = 7.5e-9
dC_end = dC / 2
R_in = 150
V_in = 2.5
N = 41  # Nodes 0 to 40
f_c = (2 / np.sqrt(L_0 * C_0)) / (2 * np.pi)
# f_c_max = (2 / np.sqrt((L_0 - dL) * (C_0 - dC))) / (2 * np.pi)

frequencies = np.linspace(20, f_c, 100)
all_v_amps = []

# Match impedance
matched = 1

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

A_L = Mc_inv @ Ml

print("Starting numerical integration. This will take a moment...")

for idx, f in enumerate(frequencies):
    omega = 2 * np.pi * f
    R_out = np.sqrt(L_0 / C_0)
    if matched:
        R_out /= np.sqrt(1 - (f / f_c) ** 2)  # Matched impedance

    # Resistance Matrix (Mr)
    Mr = np.zeros((N, N))
    Mr[0, 0] = 1 / R_in
    Mr[-1, -1] = 1 / R_out

    A_R = Mc_inv @ Mr

    # Define the ODE function for solve_ivp
    def odefunc(t, Y):
        V = Y[:N]
        V_dot = Y[N:]

        # Calculate the forcing vector F(t)
        F = np.zeros(N)
        # derivative of V_gen = 1 * e^{i w t} is i * w * e^{i w t}
        F[0] = (V_in / R_in) * (omega * np.cos(omega * t))

        # State-space equations
        V_ddot = -A_R @ V_dot - A_L @ V + Mc_inv @ F

        return np.concatenate((V_dot, V_ddot))

    # Initial conditions (zero voltage, zero current)
    Y0 = np.zeros(2 * N)

    # Integrate for 1 millisecond (enough time for transients to decay) + 2 periods
    t_span = (0, 1e-3 + 2 / f)

    # We restrict the max_step to ensure the solver catches the fast oscillations
    sol = solve_ivp(odefunc, t_span, Y0, method="RK45", max_step=1 / (20 * f))

    # Extract the steady state (last 2 periods) to find the peak amplitude
    t_steady_start = sol.t[-1] - 2 / f
    mask = sol.t >= t_steady_start

    v_amps_for_freq = [np.max(np.abs(sol.y[i][mask])) for i in range(N)]
    all_v_amps.append(v_amps_for_freq)

    print(f"Computed {idx + 1}/{len(frequencies)}: f = {f / 1000:.1f} kHz")

# 4. Exporting data
print("Simulation finished. Exporting data to CSV...")
output_dir = "data/electrical/theory_system_random"
if matched:
    output_dir += "/matched"
else:
    output_dir += "/unmatched"
os.makedirs(output_dir, exist_ok=True)

existing_indices = []
for f in os.listdir(output_dir):
    if f.startswith("output_") and f.endswith(".csv"):
        try:
            existing_indices.append(int(f[7:-4]))
        except ValueError:
            continue

next_idx = max(existing_indices) + 1 if existing_indices else 1
output_path = os.path.join(output_dir, f"output_{next_idx}.csv")
L_path = os.path.join(output_dir, f"L_{next_idx}.csv")
C_path = os.path.join(output_dir, f"C_{next_idx}.csv")

pd.DataFrame(L, columns=["L"]).to_csv(L_path, index=False, float_format="%.6e")
pd.DataFrame(C, columns=["C"]).to_csv(C_path, index=False, float_format="%.6e")

# Create a DataFrame
results_df = pd.DataFrame(all_v_amps, columns=[f"V{i}_amp" for i in range(N)])
results_df.insert(0, "frequency", frequencies)

# Save to CSV
results_df.to_csv(output_path, index=False, float_format="%.6e")
print(f"Data saved to {output_path}")
print(f"Parameters saved to {L_path} and {C_path}")
