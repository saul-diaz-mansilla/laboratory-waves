import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import scipy.optimize as opt
import scipy.interpolate as interp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import src.utils.landaubeta as lb

lb.use_latex_fonts()

# Length of RL circuits & transmission line
l = 38  # sections
l_0 = 1 # sections

# Inductance and capacitance
L = 330e-6  # H / section
dL = .2 * L
C = .015e-6 # F / section
dC = .1 * C

# Cutoff frequency
w_c = 2 / np.sqrt(L * C)

# Manual data
path = "data/electrical/task_2_6.xlsx"
df = pd.read_excel(path, header=None, engine='openpyxl')

n = df.iloc[2:35, 0].to_numpy()
in_out = df.iloc[2:35, 1].to_numpy()
f_m = df.iloc[2:35, 4].to_numpy()
v0_m = df.iloc[2:35, 5].to_numpy()
dv0_m = df.iloc[2:35, 6].to_numpy()
v38_m = df.iloc[2:35, 7].to_numpy()
dv38_m = df.iloc[2:35, 8].to_numpy()

w_m = 2 * np.pi * f_m
v0_m = v0_m / 2
dv0_m = dv0_m / 2
v38_m = v38_m / 2
dv38_m = dv38_m / 2
k = np.pi * n / l

# Fitting function
def sine(t, A, w, phi, c):
    return A * np.sin(w * t + phi) + c

# Loop over oscilloscope data to find more precise values
w = []
dw = []
vmax_0 = []
vmax_38 = []
vmax_in = []
dvmax_0 = []
dvmax_38 = []
dvmax_in = []
phase_diffs = []

for i in range(len(n)):
    file_path = "data/electrical/task_2_6/" + f"AMPPUL{i:02d}.CSV"
    data = pd.read_csv(file_path)
    t = data.iloc[:, 0].to_numpy()
    v_0 = data.iloc[:, 1].to_numpy()
    v_38 = data.iloc[:, 2].to_numpy()
    v_in = data.iloc[:, 3].to_numpy()

    popt0, pcov0 = opt.curve_fit(sine, t, v_0, p0=[v0_m[i], w_m[i], 0, 0])
    popt38, pcov38 = opt.curve_fit(sine, t, v_38, p0=[v38_m[i], w_m[i], 0, 0])
    popt_in, pcov_in = opt.curve_fit(sine, t, v_in, p0=[np.max(v_in), w_m[i], 0, 0])

    if i == None:
        plt.figure()
        plt.plot(t, v_0, label='v0 data')
        plt.plot(t, sine(t, *popt0), label='v0 fit')
        plt.plot(t, v_38, label='v38 data')
        plt.plot(t, sine(t, *popt38), label='v38 fit')
        plt.legend()
        plt.show()

    w.append(popt0[1])
    dw.append(np.sqrt(pcov0[1, 1]))
    vmax_0.append(np.abs(popt0[0]))
    vmax_38.append(np.abs(popt38[0]))
    vmax_in.append(np.abs(popt_in[0]))
    dvmax_0.append(np.sqrt(pcov0[0, 0]))
    dvmax_38.append(np.sqrt(pcov38[0, 0]))
    dvmax_in.append(np.sqrt(pcov_in[0, 0]))

    phi0 = popt0[2] if popt0[0] > 0 else popt0[2] + np.pi
    phi38 = popt38[2] if popt38[0] > 0 else popt38[2] + np.pi
    phase_diffs.append((phi38 - phi0) % (2 * np.pi))

w = np.array(w)
dw = np.array(dw)
vmax_0 = np.array(vmax_0)
vmax_38 = np.array(vmax_38)
vmax_in = np.array(vmax_in)
dvmax_0 = np.array(dvmax_0)
dvmax_38 = np.array(dvmax_38)
dvmax_in = np.array(dvmax_in)
phase_diffs = np.array(phase_diffs)

ratio = vmax_38 / vmax_0
d_ratio = ratio * np.sqrt((dvmax_38 / vmax_38)**2 + (dvmax_0 / vmax_0)**2)

fig_dir = "figures/"

plt.figure()
plt.errorbar(w, ratio, yerr=d_ratio, fmt='.', label="Experimental")
plt.title("Ratio of amplitudes")
plt.xlabel(r"$\omega$ (s$^{-1}$)")
plt.ylabel(r"$v_{38}/v_{0}$")
plt.legend()
plt.savefig(fig_dir + "ratio.pdf")

plt.figure()
plt.errorbar(k, w, yerr=dw, fmt='.', label="Experimental")
k_theo = np.linspace(0, np.pi, 1000)
plt.plot(k_theo, w_c * np.abs(np.sin(k_theo * l_0 / 2)), label="Theoretical (calc)")

def dispersion_func(k, wc):
    return wc * np.abs(np.sin(k * l_0 / 2))

popt_wc, _ = opt.curve_fit(dispersion_func, k, w, p0=[w_c])
wc_fit = popt_wc[0]
plt.plot(k_theo, dispersion_func(k_theo, wc_fit), '--', label="Theoretical (fit)")

plt.title("Dispersion relation")
plt.xlabel(r"k (sections$^{-1}$)")
plt.ylabel(r"$\omega$ (s$^{-1}$)")
plt.legend()
plt.savefig(fig_dir + "dispersion.pdf")

sort_idx = np.argsort(k)
k_sort = k[sort_idx]
w_sort = w[sort_idx]
cs = interp.CubicSpline(k_sort, w_sort)
k_new = np.linspace(k_sort.min(), k_sort.max(), 100)
w_new = cs(k_new)

vp = w_new / k_new
vg = np.gradient(w_new, k_new)

w_fit_new = dispersion_func(k_new, wc_fit)
vg_fit = wc_fit * (l_0 / 2) * np.cos(k_new * l_0 / 2)

plt.figure()
plt.plot(w_new, vp, label='Phase velocity')
plt.plot(w_new, vg, label='Group velocity')
plt.plot(w_fit_new, vg_fit, '--', label='Group velocity (fit)')
plt.xlabel(r"$\omega$ (s$^{-1}$)")
plt.ylabel(r"Velocity (sections/s)")
plt.legend()
plt.savefig(fig_dir + "velocities.pdf")

plt.figure()
plt.errorbar(w, vmax_in, yerr=dvmax_in, fmt='.', label=r'$v_{in}$')
plt.errorbar(w, vmax_0, yerr=dvmax_0, fmt='.', label=r'$v_{0}$')
plt.errorbar(w, vmax_38, yerr=dvmax_38, fmt='.', label=r'$v_{38}$')
plt.xlabel(r"$\omega$ (s$^{-1}$)")
plt.ylabel("Amplitude (V)")
plt.legend()
plt.savefig(fig_dir + "amplitudes.pdf")
plt.show()
