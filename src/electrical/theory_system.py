import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Circuit Parameters
L = 330e-6
C = 15e-9
C_end = 7.5e-9
R_in = 150
V_in = 2.5
N = 41  # Nodes 0 to 40
f_c = (2 / np.sqrt(L * C)) / (2 * np.pi)

# 3. Setup the Integration
# 30 points to keep computation time reasonable
frequencies = np.linspace(20, f_c, 30)
ratio_V38_V0 = []

print("Starting numerical integration. This will take a moment...")

for idx, f in enumerate(frequencies):
    omega = 2 * np.pi * f
    R_out = np.sqrt(L / C) / np.sqrt(
        1 - (omega / (2 * np.pi * f_c)) ** 2
    )  # Matched impedance

    # 2. Build the Matrices
    # Capacitance Matrix (Mc)
    Mc = np.diag([C_end] + [C] * (N - 2) + [C_end])

    # Resistance Matrix (Mr)
    Mr = np.zeros((N, N))
    Mr[0, 0] = 1 / R_in
    Mr[-1, -1] = 1 / R_out

    # Inductance Matrix (Ml)
    Ml = np.zeros((N, N))
    Ml[0, 0] = 1 / L
    Ml[0, 1] = -1 / L
    for i in range(1, N - 1):
        Ml[i, i - 1] = -1 / L
        Ml[i, i] = 2 / L
        Ml[i, i + 1] = -1 / L
    Ml[-1, -2] = -1 / L
    Ml[-1, -1] = 1 / L

    # Isolate the second derivative (create A_R and A_L)
    Mc_inv = np.linalg.inv(Mc)
    A_R = Mc_inv @ Mr
    A_L = Mc_inv @ Ml

    # Define the ODE function for solve_ivp
    def odefunc(t, Y):
        V = Y[:N]
        V_dot = Y[N:]

        # Calculate the forcing vector F(t)
        # // F = np.zeros(N, dtype=complex)
        F = np.zeros(N)
        # derivative of V_gen = 1 * e^{i w t} is i * w * e^{i w t}
        # // F[0] = (V_in / R_in) * (1j * omega * np.exp(1j * omega * t))
        F[0] = (V_in / R_in) * (omega * np.cos(omega * t))

        # State-space equations
        V_ddot = -A_R @ V_dot - A_L @ V + Mc_inv @ F

        return np.concatenate((V_dot, V_ddot))

    # Initial conditions (zero voltage, zero current)
    # // Y0 = np.zeros(2 * N, dtype=complex)
    Y0 = np.zeros(2 * N)

    # Integrate for 1 millisecond (enough time for transients to decay) + 10 periods
    t_span = (0, 1e-3 + 10 / f)

    # We restrict the max_step to ensure the solver catches the fast oscillations
    sol = solve_ivp(odefunc, t_span, Y0, method="RK45", max_step=1 / (20 * f))

    # Extract the steady state (last 5 periods) to find the peak amplitude
    t_steady_start = sol.t[-1] - 5 / f
    mask = sol.t >= t_steady_start

    V0_amp = np.max(np.abs(sol.y[0][mask]))
    V38_amp = np.max(np.abs(sol.y[38][mask]))

    ratio = V38_amp / V0_amp if V0_amp != 0 else 0
    ratio_V38_V0.append(ratio)
    print(f"Computed {idx + 1}/30: f = {f / 1000:.1f} kHz")

    # if idx == 15:
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(sol.t * 1000, sol.y[0], label="$V_0$")
    #     plt.plot(sol.t * 1000, sol.y[38], label="$V_{38}$")
    #     plt.title(f"Time Domain Response at f = {f / 1000:.1f} kHz")
    #     plt.xlabel("Time (ms)")
    #     plt.ylabel("Voltage (V)")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig("figures/theory_system_time.pdf")

# 4. Plotting
plt.figure(figsize=(10, 6))
plt.plot(frequencies / 1000, ratio_V38_V0, "b-o", linewidth=2, label="$|V_{38} / V_0|$")
plt.axvline(
    x=f_c / 1000,
    color="r",
    linestyle="--",
    label=f"Theoretical Cutoff ({f_c / 1000:.1f} kHz)",
)

plt.title(
    "Steady-State Amplitude Ratio via Time-Domain Numerical Integration", fontsize=14
)
plt.xlabel("Frequency (kHz)", fontsize=12)
plt.ylabel("Amplitude Ratio $|V_{38} / V_0|$", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("figures/theory_system.pdf")
plt.show()
