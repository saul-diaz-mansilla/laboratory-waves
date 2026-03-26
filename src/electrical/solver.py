import numpy as np
from numba import njit


@njit
def compute_deriv(Y, C, L, R_L_AC, R_in, R_out, V_in_val):
    N = len(C)
    dY = np.zeros_like(Y)

    dY[0] = (V_in_val - Y[0]) / (R_in * C[0]) - Y[N] / C[0]
    for i in range(1, N - 1):
        dY[i] = (Y[N + i - 1] - Y[N + i]) / C[i]
    dY[N - 1] = Y[2 * N - 2] / C[N - 1] - Y[N - 1] / (R_out * C[N - 1])

    for i in range(N - 1):
        dY[N + i] = (Y[i] - Y[i + 1] - R_L_AC[i] * Y[N + i]) / L[i]

    return dY


@njit
def rk4_solve(t_eval, Y0, C, L, R_L_AC, R_in, R_out, V_in_array):
    n_points = len(t_eval)
    n_states = len(Y0)
    Y_out = np.zeros((n_states, n_points))

    Y_curr = Y0.copy()
    Y_out[:, 0] = Y_curr

    dt = t_eval[1] - t_eval[0]

    for i in range(1, n_points):
        v_t = V_in_array[i - 1]
        v_t_half = (V_in_array[i - 1] + V_in_array[i]) / 2.0
        v_t_next = V_in_array[i]

        k1 = compute_deriv(Y_curr, C, L, R_L_AC, R_in, R_out, v_t)
        k2 = compute_deriv(Y_curr + dt / 2 * k1, C, L, R_L_AC, R_in, R_out, v_t_half)
        k3 = compute_deriv(Y_curr + dt / 2 * k2, C, L, R_L_AC, R_in, R_out, v_t_half)
        k4 = compute_deriv(Y_curr + dt * k3, C, L, R_L_AC, R_in, R_out, v_t_next)

        Y_curr = Y_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        Y_out[:, i] = Y_curr

    return Y_out
