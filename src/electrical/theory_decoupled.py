import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Circuit Parameters
L = 330e-6
C = 15e-9
C_end = 7.5e-9
R_in = 150
R_out = np.sqrt(L / C)
N_nodes = 41
N_inductors = 40

f_c = (2 / np.sqrt(L * C)) / (2 * np.pi)
f_test = f_c * 0.5  # Test at 50% of cutoff
omega = 2 * np.pi * f_test


# 2. First-Order ODE System
def circuit_odes(t, Y):
    # Y contains 41 voltages followed by 40 currents
    V = Y[:N_nodes]
    I = Y[N_nodes:]

    dV = np.zeros(N_nodes, dtype=complex)
    dI = np.zeros(N_inductors, dtype=complex)

    # Input generator (Complex exponential)
    V_gen = np.exp(1j * omega * t)

    # Node 0 (Input Boundary)
    dV[0] = ((V_gen - V[0]) / R_in - I[0]) / C_end

    # Nodes 1 to 39 (Bulk)
    for n in range(1, N_nodes - 1):
        dV[n] = (I[n - 1] - I[n]) / C

    # Node 40 (Output Boundary)
    dV[-1] = (I[-1] - V[-1] / R_out) / C_end

    # Inductors 1 to 40
    for n in range(N_inductors):
        dI[n] = (V[n] - V[n + 1]) / L

    return np.concatenate((dV, dI))


# 3. Integration Setup
# Initialize with completely zero energy
Y0 = np.zeros(N_nodes + N_inductors, dtype=complex)

# Run for 2 milliseconds to see both transient and steady state
t_span = (0, 2e-3)
t_eval = np.linspace(0, 2e-3, 2000)

print("Integrating V-I model...")
sol = solve_ivp(circuit_odes, t_span, Y0, t_eval=t_eval, method="RK45")

# 4. Plotting
plt.figure(figsize=(10, 5))

# Plot the real part of Node 38 to verify the AC oscillation is centered
plt.plot(sol.t * 1000, np.real(sol.y[38]), label="Re($V_{38}$) (Numerical)", color="b")

plt.title(f"Time-Domain Response at {f_test / 1000:.1f} kHz (V-I Model)", fontsize=14)
plt.xlabel("Time (ms)", fontsize=12)
plt.ylabel("Voltage (V)", fontsize=12)
plt.axhline(0, color="black", linewidth=1)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
