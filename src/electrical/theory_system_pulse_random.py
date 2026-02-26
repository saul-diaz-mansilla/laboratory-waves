import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.fft as fft

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

# Simulation Parameters
matched = True

# Oscilloscope AWG Parameters
awg_frequency = (
    100.0  # The repetition frequency set on the scope (e.g., 100 Hz = 10 ms duration)
)
vpp = 5.0  # Peak-to-Peak voltage (e.g., 10V swings from -5V to +5V)
num_points = 10000  # Resolution: how many points make up one full cycle

# Pulse Parameters
duration = 1.0 / awg_frequency  # Total duration of the waveform
f = 200.0  # Carrier frequency (200 Hz)
t0 = duration / 2  # Time center of the Gaussian pulse
sigma = 0.0001  # Width of the envelope
amplitude = vpp / 2.0

omega = 2 * np.pi * f

R_out = np.sqrt(L_0 / C_0)

if matched:
    R_out /= np.sqrt(
        1 - (omega / (2 * np.pi * f_c)) ** 2
    )  # Matched impedance at carrier frequency

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

    # Square wave approximation: V_gen = amplitude * tanh(k * sin(omega * t))
    k_smooth = 50.0
    sin_wt = np.sin(omega * t)
    cos_wt = np.cos(omega * t)
    tanh_val = np.tanh(k_smooth * sin_wt)
    sech2_val = 1.0 - tanh_val**2

    # dV_gen/dt
    d_Vin = amplitude * sech2_val * k_smooth * omega * cos_wt

    F[0] = (1 / R_in) * d_Vin

    # State-space equations
    V_ddot = -A_R @ V_dot - A_L @ V + Mc_inv @ F

    return np.concatenate((V_dot, V_ddot))


# Initial conditions (zero voltage, zero current)
Y0 = np.zeros(2 * N)

# Integrate
t_span = (0, duration)
sol = solve_ivp(odefunc, t_span, Y0, method="RK45", max_step=1 / f / 200)
v_0 = sol.y[0]
v_38 = sol.y[38]

# 4. Plotting
plt.figure(figsize=(10, 6))
plt.plot(sol.t * 1000, sol.y[0], label="$V_0$ (Input)")
plt.plot(sol.t * 1000, sol.y[38], label="$V_{38}$ (Output)")
plt.title("Square Wave Propagation")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/theory_system_gaussian.pdf")
plt.show()

dt = sol.t[1] - sol.t[0]
z_0 = fft.fft(v_0)
y_0 = fft.fftfreq(len(v_0), d=dt)
mask = y_0 > 0
y_0 = y_0[mask]
z_0 = z_0[mask]
z_38 = fft.fft(v_38)
y_38 = fft.fftfreq(len(v_38), d=dt)
mask = y_38 > 0
y_38 = y_38[mask]
z_38 = z_38[mask]

plt.figure()
plt.plot(y_0, np.log(np.abs(z_0) / len(v_0) * 2), label=r"$V_0$")
plt.plot(y_38, np.log(np.abs(z_38) / len(v_38) * 2), label=r"$V_{38}$")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
# plt.xlim(0, 200000)
plt.legend()
plt.show()
