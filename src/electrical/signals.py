import numpy as np
import scipy.fft as fft
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


def fit_sine(t, A, f, phi, C):
    return A * np.sin(2 * np.pi * f * t + phi) + C


def remove_pulse_offset(data):
    """
    Removes the DC offset of a pulse based data using the median absolute deviation.
    """
    baseline = np.median(data)
    aux = np.abs(data - baseline)
    mad = np.median(aux)
    quiet_mask = aux < (3 * mad)
    return data - np.mean(data[quiet_mask])


def remove_sine_offset(data):
    """
    Removes the DC offset of a sine or other periodic function by fitting to a sine plus a
    constant and removing the constant.
    """
    popt, _ = curve_fit(fit_sine, np.arange(len(data)), data, p0=[np.max(data), 1, 0])
    return data - popt[3]


def H_gaussian(V_in_data, V_out_data, t_array, freqs, t_std):
    """
    Computes a global transfer function for gaussian wavepackets using Cross-Spectral Density,
    masking out the frequencies outside a range of half a standard deviation to each side.

    Inputs:
    - V_in_data (array): all of the time series data of node 0 for each frequency
    - V_out_data (array): all of the time series data of the desired node for each frequency
    - t_array (array): common time axis
    - freqs (array): mean frequencies for each gaussian
    - t_std (float): standard deviation of all of the gaussians in the time domain

    Outputs:
    - H_global (array): global transfer function for windowed frequencies
    - freqs_global (array): frequencies at which the transfer function is evaluated
    """
    dt = t_array[1] - t_array[0]

    # Use a window of half a frequency std to each side
    half_window = 1.0 / (2 * np.pi * t_std) / 2

    V_in_data = np.atleast_2d(V_in_data)
    V_out_data = np.atleast_2d(V_out_data)

    # Mask out regions not covered by any window
    freqs_global = fft.fftfreq(len(V_in_data[0]), d=dt)
    mask_pos = (freqs_global >= (freqs[0] - half_window)) & (
        freqs_global <= (freqs[-1] + half_window)
    )
    freqs_global = freqs_global[mask_pos]

    # Calculate V_out / V_in = V_out * conj(V_in) / |V_in|^2 = S_xy / S_xx
    S_xx = np.zeros(len(freqs_global))
    S_xy = np.zeros(len(freqs_global), dtype=complex)

    # Track which frequency bins received valid data to avoid zero-division later
    windowed_mask = np.zeros(len(freqs_global), dtype=bool)

    for v_in, v_out, f_mean in zip(V_in_data, V_out_data, freqs):
        # Apply Fourier transform
        z_in = fft.fft(v_in)[mask_pos]
        z_out = fft.fft(v_out)[mask_pos]

        P_in = np.abs(z_in) ** 2

        # Window this specific wavepacket
        in_window = (freqs_global >= f_mean - half_window) & (
            freqs_global <= f_mean + half_window
        )

        S_xx[in_window] += P_in[in_window]
        S_xy[in_window] += (z_out * np.conj(z_in))[in_window]

        windowed_mask[in_window] = True

    # Calculate global transfer function safely
    H_global = np.zeros(len(freqs_global), dtype=complex)

    H_global[windowed_mask] = S_xy[windowed_mask] / S_xx[windowed_mask]

    return H_global[windowed_mask], freqs_global[windowed_mask]


def H_sine(V_in_data, V_out_data, t_array, freqs):
    """
    Computes transfer function for sines as the ratio of their amplitudes for each frequency.

    Inputs:
    - V_in_data (array): all of the time series data of node 0 for each frequency
    - V_out_data (array): all of the time series data of the desired node for each frequency
    - t_array (array): common time axis
    - freqs (array): mean frequencies for each gaussian

    Outputs:
    - H_sine (array): transfer function for sine frequencies
    - freqs (array): sine frequencies (same array as input)
    """

    H_sine = []
    for v_in, v_out, freq in zip(V_in_data, V_out_data, freqs):
        # Fit data to sine curves
        popt_in, _ = curve_fit(fit_sine, t_array, v_in, p0=[np.max(v_in), freq, 0])
        popt_out, _ = curve_fit(fit_sine, t_array, v_out, p0=[np.max(v_out), freq, 0])

        # Extract amplitude & phase info
        A_in, _, phi_in = popt_in
        A_out, _, phi_out = popt_out

        if A_in < 0:
            A_in = -A_in
            phi_in += np.pi
        if A_out < 0:
            A_out = -A_out
            phi_out += np.pi

        # Obtain full complex expression for H
        H_complex = (A_out / A_in) * np.exp(1j * (phi_out - phi_in))
        H_sine.append(H_complex)

    H_sine = np.array(H_sine)
    return H_sine, freqs


def H_pulse(V_in, V_out, t, pulse_width):
    """
    Computes transfer function for pulses via Cross-Spectral Density, deleting the data at poles.

    Inputs:
    - V_in_data (array): all of the time series data of node 0 for each frequency
    - V_out_data (array): all of the time series data of the desired node for each frequency
    - t_array (array): common time axis
    - pulse_width (float): width of the pulse in the time domain

    Outputs:
    - H_clean (array): transfer function after deleting poles
    - freqs_global (array): frequencies at which the transfer function is evaluated
    """
    # Fourier transform the data
    dt = t[1] - t[0]
    z_in = fft.fft(V_in)
    z_out = fft.fft(V_out)
    freqs_global = fft.fftfreq(len(V_in), d=dt)

    # Keep positive freqs
    mask_pos = freqs_global > 0
    freqs_global = freqs_global[mask_pos]
    z_in = z_in[mask_pos]
    z_out = z_out[mask_pos]

    P_in = np.abs(z_in) ** 2

    # Obtain H via CSD: multiply numerator & denominator by z_in conjugated
    H_raw = z_out * np.conj(z_in) / P_in

    # Drop freqs at multiples of the inverse width
    first_pole = 1 / pulse_width
    to_drop = np.arange(1, np.ceil(freqs_global[-1] / first_pole)) * first_pole

    # Delete nearby data as well
    mask_bad = np.any(
        [np.isclose(freqs_global, f, atol=first_pole / 100) for f in to_drop], axis=0
    )

    mask_good = ~mask_bad

    # Interpolate data near poles
    f_interp_real = interp1d(
        freqs_global[mask_good],
        np.real(H_raw)[mask_good],
        kind="cubic",
        fill_value="extrapolate",
    )

    f_interp_imag = interp1d(
        freqs_global[mask_good],
        np.imag(H_raw)[mask_good],
        kind="cubic",
        fill_value="extrapolate",
    )

    H_clean = np.empty_like(H_raw)

    H_clean[mask_good] = H_raw[mask_good]

    H_clean[mask_bad] = f_interp_real(freqs_global[mask_bad]) + 1j * f_interp_imag(
        freqs_global[mask_bad]
    )

    return H_clean, freqs_global
