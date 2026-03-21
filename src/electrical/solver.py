import numpy as np
from numba import njit


@njit
def compute_deriv(Y, A, B, V_in_val):
    return A @ Y + B * V_in_val


@njit
def rk4_solve(t_eval, Y0, A, B, V_in_array):
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

        k1 = compute_deriv(Y_curr, A, B, v_t)
        k2 = compute_deriv(Y_curr + dt / 2 * k1, A, B, v_t_half)
        k3 = compute_deriv(Y_curr + dt / 2 * k2, A, B, v_t_half)
        k4 = compute_deriv(Y_curr + dt * k3, A, B, v_t_next)

        Y_curr = Y_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        Y_out[:, i] = Y_curr

    return Y_out
