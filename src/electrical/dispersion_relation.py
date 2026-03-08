import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# 1. Circuit Parameters
L = 330e-6
C = 15e-9
C_end = 7.5e-9
R_in = 150
V_in = 2.5
N = 41  # Nodes 0 to 40
f_c = (2 / np.sqrt(L * C)) / (2 * np.pi)


@njit
def compute_deriv(t, Y, A_R, A_L, Mc_inv_00, V_in, R_in, omega, N):
    V = Y[:N]
    V_dot = Y[N:]

    F0 = (V_in / R_in) * (1j * omega * np.exp(1j * omega * t))

    # Safest matrix-vector product method in Numba for complex arrays
    V_ddot = -np.dot(A_R, V_dot) - np.dot(A_L, V)
    V_ddot[0] += Mc_inv_00 * F0

    dY = np.empty(2 * N, dtype=np.complex128)
    dY[:N] = V_dot
    dY[N:] = V_ddot
    return dY


@njit
def rk4_solve(t_eval, Y0, A_R, A_L, Mc_inv_00, V_in, R_in, omega, N):
    n_points = len(t_eval)
    n_states = len(Y0)
    Y_out = np.empty((n_states, n_points), dtype=np.complex128)

    Y_curr = Y0.copy()
    Y_out[:, 0] = Y_curr

    dt = t_eval[1] - t_eval[0]

    for i in range(1, n_points):
        t = t_eval[i - 1]

        k1 = compute_deriv(t, Y_curr, A_R, A_L, Mc_inv_00, V_in, R_in, omega, N)
        k2 = compute_deriv(
            t + dt / 2, Y_curr + dt / 2 * k1, A_R, A_L, Mc_inv_00, V_in, R_in, omega, N
        )
        k3 = compute_deriv(
            t + dt / 2, Y_curr + dt / 2 * k2, A_R, A_L, Mc_inv_00, V_in, R_in, omega, N
        )
        k4 = compute_deriv(
            t + dt, Y_curr + dt * k3, A_R, A_L, Mc_inv_00, V_in, R_in, omega, N
        )

        Y_curr = Y_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        Y_out[:, i] = Y_curr

    return Y_out


frequencies = np.linspace(20, f_c * 0.95, 100)
ratio_V40_V0 = []
phases = []

print("Initiating accelerated RK4 integration with strict KCL conditions...")

for idx, f in enumerate(frequencies):
    omega = 2 * np.pi * f
    R_out = np.sqrt(L / C) / np.sqrt(1 - (omega / (2 * np.pi * f_c)) ** 2)

    # 2. Build the Matrices
    Mc = np.diag([C_end] + [C] * (N - 2) + [C_end])
    Mr = np.zeros((N, N))
    Mr[0, 0] = 1 / R_in
    Mr[-1, -1] = 1 / R_out

    Ml = np.zeros((N, N))
    Ml[0, 0] = 1 / L
    Ml[0, 1] = -1 / L
    for i in range(1, N - 1):
        Ml[i, i - 1] = -1 / L
        Ml[i, i] = 2 / L
        Ml[i, i + 1] = -1 / L
    Ml[-1, -2] = -1 / L
    Ml[-1, -1] = 1 / L

    Mc_inv = np.linalg.inv(Mc)
    A_R = Mc_inv @ Mr
    A_L = Mc_inv @ Ml

    A_R = np.ascontiguousarray(A_R, dtype=np.complex128)
    A_L = np.ascontiguousarray(A_L, dtype=np.complex128)

    # 3. Apply Correct Initial Conditions
    Y0 = np.zeros(2 * N, dtype=complex)
    # V_dot_0(0) must satisfy KCL at t=0 due to the instantaneous input voltage
    Y0[N] = (V_in / R_in) / C_end

    t_end = 1.5e-3 + 5 / f

    # CRITICAL: Bulletproof stability. Limit dt to 0.1 microseconds max.
    n_points = max(int(t_end / 1e-7), int(t_end * f * 50))
    t_eval = np.linspace(0, t_end, n_points)

    Y_out = rk4_solve(t_eval, Y0, A_R, A_L, Mc_inv[0, 0], V_in, R_in, omega, N)

    t_steady_start = t_eval[-1] - 5 / f
    mask = t_eval >= t_steady_start

    V0_steady_ac = Y_out[0][mask] - np.mean(Y_out[0][mask])
    V40_steady_ac = Y_out[40][mask] - np.mean(Y_out[40][mask])

    carrier_end = np.exp(-1j * omega * t_eval[-1])

    V0_complex = V0_steady_ac[-1] * carrier_end
    V40_complex = V40_steady_ac[-1] * carrier_end

    V0_amp = np.abs(V0_complex)
    V40_amp = np.abs(V40_complex)

    ratio = V40_amp / V0_amp if V0_amp != 0 else 0
    ratio_V40_V0.append(ratio)

    phases.append(np.angle(V0_complex) - np.angle(V40_complex))

    print(f"Computed {idx + 1}/{len(frequencies)}: f = {f / 1000:.1f} kHz")
    # Plot the real part of the steady-state signals for the first frequency
    if idx == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(t_eval[mask], V0_steady_ac.real, label=r"Real($V_0$)")
        plt.plot(t_eval[mask], V40_steady_ac.real, label=r"Real($V_{40}$)")
        plt.title(
            f"Real Part of Steady-State Voltages at f = {f / 1000:.1f} kHz",
            fontsize=14,
        )
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Voltage (V)", fontsize=12)
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.show()
# 4. Plotting
plt.figure(figsize=(10, 6))
plt.plot(frequencies / 1000, ratio_V40_V0, "b-", linewidth=2, label="$|V_{40} / V_0|$")
plt.axvline(
    x=f_c / 1000,
    color="r",
    linestyle="--",
    label=f"Theoretical Cutoff ({f_c / 1000:.1f} kHz)",
)
plt.title("Steady-State Amplitude Ratio via Time-Domain", fontsize=14)
plt.xlabel("Frequency (kHz)", fontsize=12)
plt.ylabel("Amplitude Ratio $|V_{40} / V_0|$", fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

phases_unwrapped = np.unwrap(phases)
k = phases_unwrapped / (N - 1)
k_th = np.linspace(0, np.max(k), 500)

plt.figure(figsize=(10, 6))
plt.plot(
    k_th,
    f_c * np.sin(k_th / 2) / 1e3,
    "r-",
    linewidth=2,
    label="Infinite line",
)
plt.plot(k, frequencies / 1000, "b--", linewidth=2, label="Finite line")
plt.title("Dispersion Relation", fontsize=14)
plt.xlabel("Wavenumber $k$ (sections$^{-1}$)", fontsize=12)
plt.ylabel("Frequency (kHz)", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("figures/dispersion_comparison.pdf")
plt.savefig("figures/dispersion_comparison.png")
plt.show()
