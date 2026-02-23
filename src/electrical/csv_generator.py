import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Oscilloscope AWG Parameters
awg_frequency = (
    500.0  # The repetition frequency set on the scope (e.g., 100 Hz = 10 ms duration)
)
vpp = 5.0  # Peak-to-Peak voltage (e.g., 10V swings from -5V to +5V)
num_points = 10000  # Resolution: how many points make up one full cycle

# 1. Generate the Waveform Data
duration = 1.0 / awg_frequency  # Total duration of the waveform
t = np.linspace(0, duration, num_points, endpoint=False)

frequencies = np.linspace(20000, 140000, 10)
print(frequencies)
t0 = duration / 2  # Time center of the Gaussian pulse
sigma = 0.0001  # Width of the envelope
amplitude = vpp / 2.0

plt.figure(figsize=(10, 6))

for i in range(len(frequencies)):
    f = frequencies[i]
    # Calculate the Gaussian envelope and the carrier wave
    envelope = np.exp(-0.5 * ((t - t0) / sigma) ** 2)  # Gaussian envelope
    carrier = np.sin(2 * np.pi * f * t)

    # Combine and normalize
    waveform = carrier * envelope
    waveform_normalized = (waveform / np.max(np.abs(waveform))) * amplitude

    # 2. Export to CSV for USB Upload
    # Most R&S instruments prefer a simple Time(s), Voltage(V) format.
    file_name = f"data/electrical/gaussian/GEN_{i + 1}.csv"

    try:
        # Create a DataFrame for clean formatting
        df = pd.DataFrame({"Time_s": t, "Voltage_V": waveform_normalized})

        # Exporting without index and header is often safer for older firmware,
        # but modern R&S software (like RT-ZVC or ARB modules) can handle headers.
        df.to_csv(file_name, index=False, header=False)

        print(f"Success! File '{file_name}' is ready for USB transfer.")
        print(f"Total points: {len(df)}")

    except Exception as e:
        print(f"Error saving file: {e}")

    # 3. Plot the waveform
    plt.plot(t * 1000, waveform_normalized, label=f"{int(f / 1000)} kHz")

plt.title("Generated Gaussian Pulses")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.legend()
plt.show()
