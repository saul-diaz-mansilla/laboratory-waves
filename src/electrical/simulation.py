import numpy as np
import os
import sys
import time
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import src.electrical.solver as solver
import src.electrical.signals as signals
import src.utils.io as io


global_sim_args = None


def init_worker(shared_args):
    global global_sim_args
    global_sim_args = shared_args


def run_single_simulation(sim_num):
    global global_sim_args
    sim_args = global_sim_args

    # Create a unique random generator for this process worker
    rng = np.random.default_rng((int(time.time() * 1000) + sim_num) % (2**32 - 1))

    power_rule_0 = sim_args["power_rule_0"]
    dpower_rule = sim_args["dpower_rule"]
    R_in_0 = sim_args["R_in_0"]
    dR_in = sim_args["dR_in"]
    R_out_0 = sim_args["R_out_0"]
    R_out_tol = sim_args["R_out_tol"]
    noise_min = sim_args["noise_min"]
    noise_max = sim_args["noise_max"]
    temp_drift = sim_args["temp_drift"]
    L_batch_max = sim_args["L_batch_max"]
    C_batch_max = sim_args["C_batch_max"]
    R_L_ratio = sim_args["R_L_ratio"]
    f_test = sim_args["f_test"]
    C_end = sim_args["C_end"]
    dC_end = sim_args["dC_end"]
    C_0 = sim_args["C_0"]
    dC = sim_args["dC"]
    N = sim_args["N"]
    L_0 = sim_args["L_0"]
    dL = sim_args["dL"]
    R_L_0 = sim_args["R_L_0"]
    dR_L = sim_args["dR_L"]
    frequencies = sim_args["frequencies"]
    match_impedance = sim_args["match_impedance"]
    waveform = sim_args["waveform"]
    t_eval_points = sim_args["t_eval_points"]
    V_gen_all_freqs = sim_args["V_gen_all_freqs"]
    target_nodes = sim_args["target_nodes"]
    output_mode = sim_args["output_mode"]
    L_custom = sim_args.get("L_custom")
    C_custom = sim_args.get("C_custom")

    # Domain Randomization Parameters
    power_rule = rng.normal(power_rule_0, dpower_rule)
    R_in = rng.normal(R_in_0, dR_in)
    R_out_mult = rng.normal(1.0, R_out_tol)
    noise_std = rng.uniform(noise_min, noise_max)
    global_temp_drift = rng.uniform(1.0 - temp_drift, 1.0 + temp_drift)
    L_batch_factor = rng.uniform(1.0, L_batch_max)
    C_batch_factor = rng.uniform(1.0, C_batch_max)

    k_power = (R_L_ratio - 1) / f_test**power_rule_0

    if C_custom is not None:
        C = np.array(C_custom)
    else:
        C = np.concatenate(
            (
                [rng.normal(C_end, dC_end / C_batch_factor)],
                rng.normal(C_0, dC / C_batch_factor, N - 2),
                [rng.normal(C_end, dC_end / C_batch_factor)],
            )
        )
        C *= global_temp_drift

    if L_custom is not None:
        L = np.array(L_custom)
    else:
        L = rng.normal(L_0, dL / L_batch_factor, N - 1)
    R_L = rng.normal(R_L_0, dR_L, N - 1)

    omega_c = 2.0 / np.sqrt(L_0 * C_0)
    total_states = 2 * N - 1

    V_in_all_runs = []
    V_out_nodes_all_runs = {node: [] for node in np.arange(1, 41)}

    num_freqs = len(frequencies)
    R_out_array = np.zeros(num_freqs)
    R_L_AC_array = np.zeros((num_freqs, N - 1))

    for f_idx, f_current in enumerate(frequencies):
        omega = 2.0 * np.pi * f_current

        if match_impedance:
            if omega >= omega_c:
                R_out = 1e6
            else:
                R_out = np.sqrt(L_0 / C_0) / np.sqrt(1 - (omega / omega_c) ** 2)
        else:
            R_out = R_out_0

        R_out *= R_out_mult
        R_out_array[f_idx] = R_out
        R_L_AC_array[f_idx] = R_L * (1 + k_power * f_current**power_rule)

    # Pre-allocate output array for the RK4 solver
    Y0_array = np.zeros((num_freqs, total_states))
    Y_out_buffer = np.zeros((num_freqs, total_states, len(t_eval_points)))

    # In-place numerical integration across all frequencies
    Y_all_freqs = solver.rk4_solve(
        t_eval_points,
        Y0_array,
        C,
        L,
        R_L_AC_array,
        R_in,
        R_out_array,
        V_gen_all_freqs,
        Y_out_buffer,
    )

    for f_idx in range(num_freqs):
        V_clean = Y_all_freqs[f_idx, :N, :]
        V_noisy = V_clean + rng.normal(0, noise_std, V_clean.shape)

        V_in_all_runs.append(V_noisy[0, :])
        for node in V_out_nodes_all_runs.keys():
            V_out_nodes_all_runs[node].append(V_noisy[node, :])

    R_L_norm = R_L / R_L_0 if R_L_0 != 0 else np.zeros_like(R_L)
    R_in_norm = R_in / R_in_0 if R_in_0 != 0 else np.zeros_like(R_in)

    target_data = {
        "C_norm": C / C_0,
        "L_norm": np.pad(L / L_0, (0, 1), constant_values=0),
        "R_L_norm": np.pad(R_L_norm, (0, 1), constant_values=0),
        "power_rule": np.full(N, power_rule),
        "R_in_norm": np.full(N, R_in_norm),
        "R_out_mult": np.full(N, R_out_mult),
        "noise_std_mV": np.full(N, noise_std / 1e-3),
        "global_temp_drift": np.full(N, global_temp_drift),
        "C_batch_factor": np.full(N, C_batch_factor),
        "L_batch_factor": np.full(N, L_batch_factor),
    }

    result_data = {}
    freqs_global = None

    if output_mode == "transfer_function":
        for node in target_nodes:
            if waveform == "gaussian":
                H, freqs_global = signals.H_gaussian(
                    V_in_all_runs,
                    V_out_nodes_all_runs[node],
                    t_eval_points,
                    frequencies,
                    sim_args["sigma"],
                )
            elif waveform == "sine":
                H, freqs_global = signals.H_sine(
                    V_in_all_runs,
                    V_out_nodes_all_runs[node],
                    t_eval_points,
                    frequencies,
                )
            elif waveform == "pulse":
                H, freqs_global = signals.H_pulse(
                    V_in_all_runs,
                    V_out_nodes_all_runs[node],
                    t_eval_points,
                    sim_args["pulse_width"],
                )

            result_data[f"H_Mag_{node}"] = np.abs(H).tolist()
            result_data[f"H_Phase_{node}"] = np.angle(H).tolist()

    elif output_mode == "time_series":
        result_data["V_0"] = V_in_all_runs
        for node in target_nodes:
            result_data[f"V_{node}"] = V_out_nodes_all_runs[node]

    return target_data, result_data, freqs_global


def simulate(config_path, L_custom=None, C_custom=None):
    """
    Main simulation core. Includes internal resistances in inductors and frequency dependence in R with a power rule.
    Uses Monte Carlo methods to add variations to the base parameters.

    Inputs:
    - config_path (str): path to the ".yaml" master config file (config/experiment)
    - L_custom (list or np.ndarray): custom inductance array (bypasses randomizer). Default is None.
    - C_custom (list or np.ndarray): custom capacitance array (bypasses randomizer). Default is None.

    Outputs:
    - all_targets (dict): Dictionary containing the randomized parameters for each simulation. Contains "C_norm", "L_norm",
    "R_L_norm", "power_rule", "R_in_norm", "R_out_mult", "noise_std_mV", "global_temp_drift", "C_batch_factor",
    and "L_batch_factor" keys.
    - all_results (dict): Dictionary containing the simulation results (time series or transfer functions). For transfer
    function mode, contains "H_Mag_{node}", and "H_Phase_{node}" keys for all target nodes. For time series mode,
    contains "V_0" and "V_{node}" keys for all target nodes.
    - all_freqs (dict): Dictionary containing "freqs_global", which contains the common frequency axis for transfer funcions.
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

    # Handle ideal case
    if R_L_0 != 0:
        R_L_ratio = R_L_test / R_L_0
    else:
        R_L_ratio = 1

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

        V_gen_all_freqs = np.array(
            [
                amplitude
                * np.exp(-((t_eval_points - t0) ** 2) / (2 * sigma**2))
                * np.cos(2 * np.pi * frequency * t_eval_points)
                for frequency in frequencies
            ]
        )
    elif waveform == "sine":
        f_start = exp_parameters["input"]["f_start"]
        f_end = exp_parameters["input"]["f_end"]
        num_inputs = exp_parameters["input"]["num_inputs"]
        frequencies = np.linspace(f_start, f_end, num_inputs)
        amplitude = exp_parameters["input"]["amplitude"]

        V_gen_all_freqs = np.array(
            [
                amplitude * np.sin(2 * np.pi * frequency * t_eval_points)
                for frequency in frequencies
            ]
        )
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

        V_gen_all_freqs = np.array([V_pulse for _ in frequencies])

    else:
        raise ValueError(
            f"Unknown waveform type: {waveform}. Must be 'gaussian', 'sine' or 'pulse'."
        )

    # Bundle all arguments needed by the worker function
    sim_args = {
        "power_rule_0": power_rule_0,
        "dpower_rule": dpower_rule,
        "R_in_0": R_in_0,
        "dR_in": dR_in,
        "R_out_0": R_out_0,
        "R_out_tol": R_out_tol,
        "noise_min": noise_min,
        "noise_max": noise_max,
        "temp_drift": temp_drift,
        "L_batch_max": L_batch_max,
        "C_batch_max": C_batch_max,
        "R_L_ratio": R_L_ratio,
        "f_test": f_test,
        "C_end": C_end,
        "dC_end": dC_end,
        "C_0": C_0,
        "dC": dC,
        "N": N,
        "L_0": L_0,
        "dL": dL,
        "R_L_0": R_L_0,
        "dR_L": dR_L,
        "frequencies": frequencies,
        "match_impedance": match_impedance,
        "waveform": waveform,
        "t_eval_points": t_eval_points,
        "V_gen_all_freqs": V_gen_all_freqs,
        "target_nodes": target_nodes,
        "output_mode": output_mode,
        "L_custom": L_custom,
        "C_custom": C_custom,
    }

    if waveform == "gaussian":
        sim_args["sigma"] = sigma
    elif waveform == "pulse":
        sim_args["pulse_width"] = pulse_width

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
    if output_mode == "transfer_function":
        for node in target_nodes:
            all_results[f"H_Mag_{node}"] = []
            all_results[f"H_Phase_{node}"] = []
    elif output_mode == "time_series":
        all_results["V_0"] = []
        for node in target_nodes:
            all_results[f"V_{node}"] = []

    freqs_global_final = np.array([])

    with concurrent.futures.ProcessPoolExecutor(
        initializer=init_worker, initargs=(sim_args,)
    ) as executor:
        results = executor.map(
            run_single_simulation,
            range(num_simulations),
            chunksize=max(1, num_simulations // (os.cpu_count() or 16)),
        )

        for target_data, result_data, freqs_global_ret in results:
            for key in all_targets:
                all_targets[key].append(target_data[key])
            for key in all_results:
                all_results[key].append(result_data[key])
            if freqs_global_ret is not None:
                freqs_global_final = freqs_global_ret

    all_freqs = {"freqs_global": [freqs_global_final.tolist()]}
    return all_targets, all_results, all_freqs
