# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 09:38:31 2026

@author: sd3525
"""
import pandas as pd
import re
import scipy.signal as sps
import scipy.optimize as spo
import numpy as np
import matplotlib.pyplot as plt
import os

T = 30
w = 2*np.pi/T
dx = 5e-3
xi = np.arange(0, 8) * dx
D = 34.1e-6

def sine_f(t, ph, w, T0, a, y):
    return T0 + y*t + a * np.sin(-w*t + ph)

def load_dataset(path):
    data = pd.read_csv(path, header=3)
    timestamp = data.iloc[:, 0].to_numpy()
    output_voltage = data.iloc[:, 1].to_numpy()
    output_current = data.iloc[:, 2].to_numpy()
    thermistor_temperatures = data.iloc[:, 3:].to_numpy()
    comments = re.search(r"Comments: (.*)$", open(path).read(), re.MULTILINE)[1]
    return timestamp, output_voltage, output_current, thermistor_temperatures, comments

timestamp, output_voltage, output_current, thermistor_temperatures, comments = (load_dataset(os.path.join(os.path.dirname(__file__), "../../data/thermal/task_1_6_"+str(int(T))+".csv")))

total = len(timestamp)
#timestamp = timestamp[8*total//9:]
#thermistor_temperatures = thermistor_temperatures[8*total//9:, :]

popt, pcov = spo.curve_fit(sine_f, timestamp, thermistor_temperatures[:, 0], p0=[0, w, np.mean(thermistor_temperatures[:, 0]), 1, 0.0008], bounds=([-np.inf, 0, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]))
phi0 = popt[0] % (2*np.pi)
a0 = popt[3]

deltaph = []
gamma = []
for i in range(8):#0.0008
    phase_guess = phi0 + np.sqrt(w/2/D)*xi[i]
    popt, pcov = spo.curve_fit(sine_f, timestamp, thermistor_temperatures[:, i], p0=[phase_guess, w, np.mean(thermistor_temperatures[:, i]), .5, 0.0008], bounds=([-np.inf, 0, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]))
    phase = popt[0] % (2*np.pi)
    if i==0:
        phi0 = phase
        a0 = popt[3]
    deltaph.append((phase-phi0) % (2*np.pi))
    gamma.append(popt[3]/ a0)

for i in range(7):
    print(w*xi[i+1]**2/2/np.log(gamma[i+1])**2)
    print(w*xi[i+1]**2/2/deltaph[i+1]**2)
    
plt.figure()
plt.plot(xi, gamma, '.')
plt.title("gamma")
plt.xlabel("distance")
plt.ylabel("gamma")

plt.figure()
plt.plot(xi, deltaph, '.')
plt.title("phase")
plt.xlabel("distance")
plt.ylabel("phase")

plt.show()