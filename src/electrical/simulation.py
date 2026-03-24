import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import src.electrical.solver as solver
import src.electrical.signals as signals
import src.utils.io as io


def simulate(config_path):
    """
    Main simulation core. Includes internal resistances in inductors and frequency dependence in R with a power rule.
    Uses Monte Carlo methods to add variations to the base parameters.

    Inputs:
    - config_path (str): path to the ".yaml" master config file (config/experiment)

    Outputs:
    - all_targets (dict): Dictionary containing the randomized parameters for each simulation. Contains "C_norm", "L_norm",
    "R_L_norm", "power_rule", "R_in_norm", "R_out_mult", "noise_std_mV", "global_temp_drift", "C_batch_factor",
    and "L_batch_factor" keys.
    - all_results (dict): Dictionary containing the simulation results (time series or transfer functions). Contains
    "Frequency (Hz)", "H_Mag_{node}", and "H_Phase_{node}" keys for all target nodes.

    """

    # Import master config from yaml file
    exp_parameters = io.load_config(config_path)
    circuit_path = exp_parameters["paths"]["circuit_config"]
    sim_path = exp_parameters["paths"]["simulation_config"]

    # Import circuit and input parameters from yaml files
    circuit_parameters = io.load_config(circuit_path)
    sim_parameters = io.load_config(sim_path)

    # Simulation config
    num_points = sim_parameters["execution"]["num_points"]
    num_simulations = sim_parameters["execution"]["num_simulations"]
    output_mode = sim_parameters["execution"]["output_mode"]
    target_nodes = sim_parameters["execution"]["target_nodes"]

    if output_mode not in ["time_series", "transfer_function"]:
        raise ValueError(
            "Invalid output mode. Must be 'time_series' or 'transfer_function'."
        )

    # Circuit Parameters
    L_0 = circuit_parameters["L"]["value"]
    dL = circuit_parameters["L"]["tolerance"]
    C_0 = circuit_parameters["C"]["value"]
    dC = circuit_parameters["C"]["tolerance"]
    C_end = circuit_parameters["C_end"]["value"]
    dC_end = circuit_parameters["C_end"]["tolerance"]
    R_in_0 = circuit_parameters["R_in"]["value"]
    dR_in = circuit_parameters["R_in"]["tolerance"]
    R_out_0 = circuit_parameters["R_out"]["value"]
    R_out_tol = circuit_parameters["R_out"]["tolerance"]

    N = circuit_parameters["N"]["value"]

    # Deviations from ideal behavior
    # Parasitic resistances in L
    R_L_0 = circuit_parameters["R_L"]["value"]
    dR_L = circuit_parameters["R_L"]["tolerance"]

    # Resistance ratio at 796 kHz from quality factor
    f_test = circuit_parameters["f_test"]["value"]
    R_L_test = circuit_parameters["R_L_test"]["value"]
    R_L_ratio = R_L_test / R_L_0

    # Model R_AC = R_DC * (1 + k_power * f**power_rule)
    power_rule_0 = circuit_parameters["power_rule"]["value"]
    dpower_rule = circuit_parameters["power_rule"]["tolerance"]

    # Domain randomization configs
    noise_min = sim_parameters["randomization"]["noise_min"]
    noise_max = sim_parameters["randomization"]["noise_max"]
    temp_drift = sim_parameters["randomization"]["temp_drift"]
    L_batch_max = sim_parameters["randomization"]["L_batch_max"]
    C_batch_max = sim_parameters["randomization"]["C_batch_max"]

    # Input function shape and parameters
    waveform = exp_parameters["input"]["waveform"]
    duration = exp_parameters["input"]["duration"]
    match_impedance = exp_parameters["input"]["match_impedance"]

    # RK4 parameters
    t_eval_points = np.linspace(0, duration, num_points, endpoint=False)

    # Input function values for all frequencies and times
    if waveform == "gaussian":
        f_start = exp_parameters["input"]["f_start"]
        f_end = exp_parameters["input"]["f_end"]
        num_inputs = exp_parameters["input"]["num_inputs"]
        frequencies = np.linspace(f_start, f_end, num_inputs)
        t0 = exp_parameters["input"]["t0"]
        sigma = exp_parameters["input"]["sigma"]
        amplitude = exp_parameters["input"]["amplitude"]

        V_gen_all_freqs = {
            frequency: (
                amplitude
                * np.exp(-((t_eval_points - t0) ** 2) / (2 * sigma**2))
                * np.cos(2 * np.pi * frequency * t_eval_points)
            )
            for frequency in frequencies
        }
    elif waveform == "sine":
        f_start = exp_parameters["input"]["f_start"]
        f_end = exp_parameters["input"]["f_end"]
        num_inputs = exp_parameters["input"]["num_inputs"]
        frequencies = np.linspace(f_start, f_end, num_inputs)
        amplitude = exp_parameters["input"]["amplitude"]

        V_gen_all_freqs = {
            frequency: (amplitude * np.sin(2 * np.pi * frequency * t_eval_points))
            for frequency in frequencies
        }
    elif waveform == "pulse":
        frequencies = np.array([0])
        pulse_width = exp_parameters["input"]["width"]
        pulse_height = exp_parameters["input"]["amplitude"]
        t_rise = exp_parameters["input"]["t_rise"]

        rise_section = np.where(t_eval_points < t_rise)[0]
        plateau_section = np.where(
            (t_eval_points >= t_rise) & (t_eval_points < pulse_width)
        )[0]
        fall_section = np.where(
            (t_eval_points >= pulse_width) & (t_eval_points < pulse_width + t_rise)
        )[0]

        V_pulse = np.zeros_like(t_eval_points)
        V_pulse[rise_section] = (pulse_height / t_rise) * t_eval_points[rise_section]
        V_pulse[plateau_section] = pulse_height
        V_pulse[fall_section] = pulse_height - (pulse_height / t_rise) * (
            t_eval_points[fall_section] - pulse_width
        )

        V_gen_all_freqs = {frequency: V_pulse for frequency in frequencies}

    else:
        raise ValueError(
            f"Unknown waveform type: {waveform}. Must be 'gaussian', 'sine' or 'pulse'."
        )

    all_targets = {
        "C_norm": [],
        "L_norm": [],
        "R_L_norm": [],
        "power_rule": [],
        "R_in_norm": [],
        "R_out_mult": [],
        "noise_std_mV": [],
        "global_temp_drift": [],
        "C_batch_factor": [],
        "L_batch_factor": [],
    }
    all_results = {}
    for node in target_nodes:
        all_results[f"H_Mag_{node}"] = []
        all_results[f"H_Phase_{node}"] = []

    # * ------ START SIMULATION LOOP ------
    for sim_num in range(num_simulations):
        # Domain Randomization Parameters
        power_rule = np.random.normal(power_rule_0, dpower_rule)
        R_in = np.random.normal(R_in_0, dR_in)
        R_out_mult = np.random.normal(1.0, R_out_tol)
        noise_std = np.random.uniform(noise_min, noise_max)
        global_temp_drift = np.random.uniform(1.0 - temp_drift, 1.0 + temp_drift)
        L_batch_factor = np.random.uniform(1.0, L_batch_max)
        C_batch_factor = np.random.uniform(1.0, C_batch_max)

        k_power = (R_L_ratio - 1) / f_test**power_rule_0

        # Build the Matrices (randomized)
        C = np.concatenate(
            (
                [np.random.normal(C_end, dC_end / C_batch_factor)],
                np.random.normal(C_0, dC / C_batch_factor, N - 2),
                [np.random.normal(C_end, dC_end / C_batch_factor)],
            )
        )
        C *= global_temp_drift

        L = np.random.normal(L_0, dL / L_batch_factor, N - 1)
        R_L = np.random.normal(R_L_0, dR_L, N - 1)

        # Calculate theoretical cutoff angular frequency based on nominal values
        omega_c = 2.0 / np.sqrt(L_0 * C_0)

        # Build First-Order V-I State-Space Matrices
        total_states = 2 * N - 1
        A = np.zeros((total_states, total_states))
        B = np.zeros(total_states)

        B[0] = 1 / (R_in * C[0])
        A[0, 0] = -1 / (R_in * C[0])
        A[N - 1, total_states - 1] = 1 / C[-1]

        i = np.arange(N - 1)

        A[i, N + i] = -1 / C[i]
        A[i + 1, N + i] = 1 / C[i + 1]
        A[N + i, i] = 1 / L[i]
        A[N + i, i + 1] = -1 / L[i]
        A[N + i, N + i] = -R_L[i] / L[i]

        V_in_all_runs = []
        V_out_nodes_all_runs = {node: [] for node in np.arange(1, 41)}

        # * ------ START FREQUENCY LOOP ------
        # Integrate 30 times (Dynamic Impedance Matching)
        for f_current in frequencies:
            omega = 2.0 * np.pi * f_current

            if match_impedance:
                # Calculate frequency-dependent matched impedance
                if omega >= omega_c:
                    # Prevent complex numbers/division by zero if approaching or exceeding cutoff
                    R_out = 1e6
                else:
                    R_out = np.sqrt(L_0 / C_0) / np.sqrt(1 - (omega / omega_c) ** 2)
            else:
                R_out = R_out_0

            R_out *= R_out_mult

            # Update the load boundary condition in the A matrix for this specific frequency
            A[N - 1, N - 1] = -1 / (R_out * C[-1])

            # Update Inductor AC Losses
            R_L_AC = R_L * (1 + k_power * f_current**power_rule)
            idx = np.arange(N, 2 * N - 1)
            A[idx, idx] = -R_L_AC / L
            Y0 = np.zeros(total_states)

            V_gen = V_gen_all_freqs[f_current]

            Y_all = solver.rk4_solve(t_eval_points, Y0, A, B, V_gen)
            V_clean = Y_all[:N, :]

            V_noisy = V_clean + np.random.normal(0, noise_std, V_clean.shape)

            V_in_all_runs.append(V_noisy[0, :])
            for node in V_out_nodes_all_runs.keys():
                V_out_nodes_all_runs[node].append(V_noisy[node, :])

        # * ------ END FREQUENCY LOOP ------

        # Save targets (L, C, and new ML parameters padded/broadcasted to length N)
        all_targets["C_norm"].append(C / C_0)
        all_targets["L_norm"].append(np.pad(L / L_0, (0, 1), constant_values=0))
        all_targets["R_L_norm"].append(np.pad(R_L / R_L_0, (0, 1), constant_values=0))
        all_targets["power_rule"].append(np.full(N, power_rule))
        all_targets["R_in_norm"].append(np.full(N, R_in / R_in_0))
        all_targets["R_out_mult"].append(np.full(N, R_out_mult))
        all_targets["noise_std_mV"].append(np.full(N, noise_std / 1e-3))
        all_targets["global_temp_drift"].append(np.full(N, global_temp_drift))
        all_targets["C_batch_factor"].append(np.full(N, C_batch_factor))
        all_targets["L_batch_factor"].append(np.full(N, L_batch_factor))

        if output_mode == "transfer_function":
            # Calculate Global Transfer Functions for Target Nodes
            for node in target_nodes:
                if waveform == "gaussian":
                    H, freqs_global = signals.H_gaussian(
                        V_in_all_runs,
                        V_out_nodes_all_runs[node],
                        t_eval_points,
                        frequencies,
                        sigma,
                    )  # TODO: update parameter inputs when new H functions are implemented
                elif waveform == "sine":
                    H, freqs_global = signals.H_sine(
                        V_in_all_runs,
                        V_out_nodes_all_runs[node],
                        t_eval_points,
                        frequencies,
                    )  # TODO: update parameter inputs when new H functions are implemented
                elif waveform == "pulse":
                    H, freqs_global = signals.H_pulse(
                        V_in_all_runs,
                        V_out_nodes_all_runs[node],
                        t_eval_points,
                    )

                all_results[f"H_Mag_{node}"].append(np.abs(H).tolist())
                all_results[f"H_Phase_{node}"].append(np.angle(H).tolist())

        # TODO: Update time series output
        elif output_mode == "time_series":
            all_results[sim_num] = {
                "V_in": V_in_all_runs,
                "V_out": V_out_nodes_all_runs,
            }

    # * ------ END SIMULATION LOOP ------
    all_freqs = {"freqs_global": [freqs_global.tolist()]}
    return all_targets, all_results, all_freqs
