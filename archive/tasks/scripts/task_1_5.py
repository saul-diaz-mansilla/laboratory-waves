# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 09:38:31 2026

@author: sd3525
"""

import pandas as pd
import re
import scipy.optimize as spo
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

w = 2 * np.pi / 10
dx = 5e-3
xi = np.arange(0, 8) * dx


def sine_f(t, ph, w, T0, a, y):
    return T0 + y * t + a * np.sin(w * t + ph)


def load_dataset(path):
    data = pd.read_csv(path, header=3)
    timestamp = data.iloc[:, 0].to_numpy()
    output_voltage = data.iloc[:, 1].to_numpy()
    output_current = data.iloc[:, 2].to_numpy()
    thermistor_temperatures = data.iloc[:, 3:].to_numpy()
    comments = re.search(r"Comments: (.*)$", open(path).read(), re.MULTILINE)[1]
    return timestamp, output_voltage, output_current, thermistor_temperatures, comments


timestamp, output_voltage, output_current, thermistor_temperatures, comments = (
    load_dataset("archive/thermal/data/task_1_5.csv")
)

total = len(timestamp)
timestamp = timestamp[8 * total // 9 :]
thermistor_temperatures = thermistor_temperatures[8 * total // 9 :, :]

deltaph = []
gamma = []
for i in range(8):
    popt, pcov = spo.curve_fit(
        sine_f,
        timestamp,
        thermistor_temperatures[:, i],
        p0=[6, w, 32, 0.5, 0.0008],
        bounds=(
            [0, 0, -np.inf, 0, -np.inf],
            [2 * np.pi, np.inf, np.inf, np.inf, np.inf],
        ),
    )
    if i == 0:
        phi0 = popt[0]
        a0 = popt[3]
    deltaph.append(popt[0] - phi0)
    gamma.append(popt[3] / a0)

for i in range(7):
    print(w * xi[i + 1] ** 2 / 2 / np.log(gamma[i + 1]) ** 2)
    print(w * xi[i + 1] ** 2 / 2 / deltaph[i + 1] ** 2)

plt.figure()
plt.plot(timestamp, thermistor_temperatures[:, -1])
plt.plot(timestamp, sine_f(timestamp, *popt))
plt.show()
