import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import os
import time

time_start = time.time()

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

# Pulse Parameters
duration = 1e-3  # Total duration of the waveform
pulse_width = 2e-5
pulse_height = 5.0
R_out = 130

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
    tr = 1e-7
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

# Integrate
t_span = (0, duration)

# Create a uniform time grid for the output
# Using 1e-7 to match your desired resolution (10,000 points)
t_eval_points = np.linspace(0, duration, int(duration / 1e-7) + 1)

sol = solve_ivp(odefunc, t_span, Y0, method="RK45", max_step=1e-7, t_eval=t_eval_points)

# 4. Exporting data
output_dir = "data/electrical/pulses"
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

data = {"time": sol.t}
for i in range(N):
    data[f"v_{i}"] = sol.y[i]
df = pd.DataFrame(data)
df.to_csv(output_path, index=False)
print(f"Data saved to {output_path}")
