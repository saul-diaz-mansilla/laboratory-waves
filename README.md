# laboratory-waves
An experimental project mainly on electrical waves through a discrete LC transmission line. We studied the transmission line's deviations from the model and resonances, attempting to invert its signals to obtain back the experimental distribution of inductances L and capacitances C using a neural network.

See final presentation in "Modelling of Transmission Line through Monte Carlo Simulations.pptx" (no report).

Install "requirements.txt" before running the source code (compatible with Python 3.10 and further versions).

Directories:
- "src/": source code for the project.
- "data/": experimental and simulation data
- "figures/": images for main results of studied system
- "latex/": contains sources and references. As there is no report, it just contains the bibliography.
See READMEs inside the folders for more details.

General outline of the project:
- "electrical/": Study of several input signals in a coupled LC transmission line.
- "inverse_problem/": Attempts to invert the signals from the transmission line to find the line's parameters.
- "thermal/": Study of thermal waves in a brass cylinder. Secondary part of the project, unrelated to the previous ones.+

Generating filtered data:
- In directories "electrical/" and "inverse_problem/", the experimental data from the oscilloscope is passed through a Butterworth filter to remove high-frequency noise. Run "electrical/data_filter.py" to filter the desired raw data and save it.
