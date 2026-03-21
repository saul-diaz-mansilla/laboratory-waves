import numpy as np
import scipy.fft as fft


def H_gaussian(V_in_data, V_out_data, t_array, freqs, t_std):
    """
    Computes a global transfer function for gaussian wavepackets using Cross-Spectral Density,
    masking out the frequencies outside a range of half a standard deviation to each side.
    """
    dt = t_array[1] - t_array[0]
    # Calculate the frequency standard deviation theoretically
    f_std_2 = 1.0 / (2 * np.pi * t_std) / 2

    if not isinstance(V_in_data, list):
        V_in_data = [V_in_data]
        V_out_data = [V_out_data]

    y0 = fft.fftfreq(len(V_in_data[0]), d=dt)
    mask_pos = (y0 > (freqs[0] - f_std_2)) & (y0 < 150e3)
    y0 = y0[mask_pos]

    S_xx = np.zeros(len(y0))
    S_xy = np.zeros(len(y0), dtype=complex)

    # Track which frequency bins received valid data to avoid zero-division later
    coverage_mask = np.zeros(len(y0), dtype=bool)

    for v_in, v_out, f_mean in zip(V_in_data, V_out_data, freqs):
        z_in = fft.fft(v_in)[mask_pos]
        z_out = fft.fft(v_out)[mask_pos]

        # Calculate the Power Spectral Density (PSD) of the input
        P_in = np.abs(z_in) ** 2

        # Identify the high-energy region of this specific wavepacket
        valid_bins = (y0 >= f_mean - f_std_2) & (y0 <= f_mean + f_std_2)

        # Accumulate only the significant bins (cutting off the noisy tails)
        S_xx[valid_bins] += P_in[valid_bins]
        S_xy[valid_bins] += (z_out * np.conj(z_in))[valid_bins]

        # Mark these frequencies as covered
        coverage_mask[valid_bins] = True

    # Calculate global transfer function safely
    H_global = np.zeros(len(y0), dtype=complex)

    H_global[coverage_mask] = S_xy[coverage_mask] / (S_xx[coverage_mask])

    return H_global, y0


# TODO: Obtain H for sines and pulses.


def H_sine():
    return None


def H_pulse():
    return None
