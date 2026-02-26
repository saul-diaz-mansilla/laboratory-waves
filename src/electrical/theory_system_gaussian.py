import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.fft as fft

# 1. Circuit Parameters
L = 330e-6
C = 15e-9
C_end = 7.5e-9
R_in = 150
V_in = 2.5
N = 41  # Nodes 0 to 40
f_c = (2 / np.sqrt(L * C)) / (2 * np.pi)

# Simulation Parameters
matched = True

# Oscilloscope AWG Parameters
awg_frequency = (
    500.0  # The repetition frequency set on the scope (e.g., 100 Hz = 10 ms duration)
)
vpp = 5.0  # Peak-to-Peak voltage (e.g., 10V swings from -5V to +5V)
num_points = 10000  # Resolution: how many points make up one full cycle

# Pulse Parameters
duration = 1.0 / awg_frequency  # Total duration of the waveform
f = 50e3  # Carrier frequency (50 kHz)
t0 = duration / 2  # Time center of the Gaussian pulse
sigma = 0.00002  # Width of the envelope
amplitude = vpp / 2.0

omega = 2 * np.pi * f

R_out = np.sqrt(L / C)

if matched:
    R_out /= np.sqrt(
        1 - (omega / (2 * np.pi * f_c)) ** 2
    )  # Matched impedance at carrier frequency

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
    F = np.zeros(N)

    # Derivative of V_gen = V_in * exp(-0.5*((t-t0)/sigma)**2) * sin(omega*t)
    envelope = amplitude * np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    d_envelope = amplitude * envelope * (-1.0 * (t - t0) / (sigma**2))

    # Product rule: d(uv) = u'v + uv'
    d_Vin = V_in * (
        d_envelope * np.sin(omega * t) + envelope * omega * np.cos(omega * t)
    )

    F[0] = (1 / R_in) * d_Vin

    # State-space equations
    V_ddot = -A_R @ V_dot - A_L @ V + Mc_inv @ F

    return np.concatenate((V_dot, V_ddot))


# Initial conditions (zero voltage, zero current)
Y0 = np.zeros(2 * N)

# Integrate
t_span = (0, duration)
sol = solve_ivp(odefunc, t_span, Y0, method="RK45", max_step=1 / f / 20)
v_0 = sol.y[0]
v_38 = sol.y[38]

# 4. Plotting
plt.figure(figsize=(10, 6))
plt.plot(sol.t * 1000, v_0, label="$V_0$ (Input)")
plt.plot(sol.t * 1000, v_38, label="$V_{38}$ (Output)")
plt.title("Gaussian Pulse Propagation")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/theory_system_gaussian.pdf")

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
plt.plot(y_0, np.abs(z_0) / len(v_0) * 2, label=r"$V_0$")
plt.plot(y_38, np.abs(z_38) / len(v_38) * 2, label=r"$V_{38}$")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 200000)
plt.legend()
plt.show()
