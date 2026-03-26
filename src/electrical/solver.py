import numpy as np
from numba import njit


@njit(cache=True)
def compute_deriv(Y, C, L, R_L_AC, R_in, R_out, V_in_array):
    num_freqs = Y.shape[0]
    N = len(C)
    dY = np.zeros_like(Y)

    for f in range(num_freqs):
        dY[f, 0] = (V_in_array[f] - Y[f, 0]) / (R_in * C[0]) - Y[f, N] / C[0]
        for i in range(1, N - 1):
            dY[f, i] = (Y[f, N + i - 1] - Y[f, N + i]) / C[i]
        dY[f, N - 1] = Y[f, 2 * N - 2] / C[N - 1] - Y[f, N - 1] / (R_out[f] * C[N - 1])

        for i in range(N - 1):
            dY[f, N + i] = (Y[f, i] - Y[f, i + 1] - R_L_AC[f, i] * Y[f, N + i]) / L[i]

    return dY


@njit(cache=True)
def rk4_solve(t_eval, Y0, C, L, R_L_AC, R_in, R_out, V_in_array, Y_out):
    n_points = len(t_eval)

    Y_curr = Y0.copy()
    Y_out[:, :, 0] = Y_curr

    dt = t_eval[1] - t_eval[0]

    for i in range(1, n_points):
        v_t = V_in_array[:, i - 1]
        v_t_half = (V_in_array[:, i - 1] + V_in_array[:, i]) / 2.0
        v_t_next = V_in_array[:, i]

        k1 = compute_deriv(Y_curr, C, L, R_L_AC, R_in, R_out, v_t)
        k2 = compute_deriv(Y_curr + dt / 2 * k1, C, L, R_L_AC, R_in, R_out, v_t_half)
        k3 = compute_deriv(Y_curr + dt / 2 * k2, C, L, R_L_AC, R_in, R_out, v_t_half)
        k4 = compute_deriv(Y_curr + dt * k3, C, L, R_L_AC, R_in, R_out, v_t_next)

        Y_curr = Y_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        Y_out[:, :, i] = Y_curr

    return Y_out
