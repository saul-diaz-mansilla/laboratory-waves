import numpy as np

# from RsInstrument import *
import RsInstrument

# 1. Generate the Waveform Data
sample_rate = 1e6  # 1 MS/s
duration = 0.01  # 10 ms duration
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

f = 5000  # Carrier frequency (5 kHz)
t0 = duration / 2  # Time center of the Gaussian pulse
sigma = 0.001  # Width of the envelope

# Calculate the Gaussian envelope and the carrier wave
envelope = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
carrier = np.sin(2 * np.pi * f * t)

# Combine and normalize the waveform to the range [-1.0, 1.0]
waveform = carrier * envelope
waveform_normalized = waveform / np.max(np.abs(waveform))

# Convert to 32-bit floating point format for SCPI transfer
waveform_bytes = waveform_normalized.astype(np.float32).tobytes()

# 2. Transfer Data to the Instrument
try:
    # Initialize connection (replace with your oscilloscope's IP address or VISA resource string)
    instr = RsInstrument.RsInstrument(
        "TCPIP::xxxxxxxxxxx::SOCKET ",
        id_query=True,
        reset=False,
        options="SelectVisa='pyvisa-py'",
    )
    print(f"Connected to: {instr.idn_string}")

    # Note: Exact SCPI commands vary by specific R&S oscilloscope models and AWG modules.
    # The following is a standard format for writing a binary block to a generic AWG.
    # Consult your specific instrument's programmer manual for the exact node.
    instr.write_bin_block("WGEN:ARB:DATA ", waveform_bytes)

    # Enable the arbitrary waveform generator output
    instr.write("WGEN:OUTPut ON")

    print("Waveform successfully uploaded to the AWG.")
    instr.close()

except Exception as e:
    print(f"Communication error: {e}")
