import matplotlib.pyplot as plt


def apply_standard_style(nrows=1, ncols=1, **kwargs):
    """
    Applies global matplotlib style parameters for consistent formatting. Apply inside
    plt.subplots using "**".

    Input:
    - nrows (int): number of rows of the subplot grid
    - ncols (int): number of columns of the subplot grid
    - **kwargs: additional arguments for plt.subplots

    Output:
    - config (dict): arguments that plt.subplots will take
    """
    plt.rcParams.update(
        {
            # "text.usetex": True,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
        }
    )
    config = {
        "nrows": nrows,
        "ncols": ncols,
        "figsize": (6.4 * ncols, 4.8 * nrows),
        "layout": "constrained",
    }
    config.update(kwargs)
    return config


def plot_style(
    ax,
    x_sim=None,
    y_sim=None,
    x_exp=None,
    y_exp=None,
    x_sim_trend=None,
    y_sim_trend=None,
    x_exp_trend=None,
    y_exp_trend=None,
    dy_sim=None,
    dy_exp=None,
    vline_x=None,
    vline_label=None,
):
    """
    Plots simulation vs experimental transfer functions on a given axis
    with strictly ordered legends and consistent colors: blue for simulations
    and red for experimental data.

    Inputs:
    - ax: matplotlib ax of the subplot
    - x_sim: simulated x values
    - y_sim: simulated y values
    - dy_sim: optional errorbars for simulated data
    - x_exp: experimental x values
    - y_exp: experimental y values
    - dy_exp: optional errorbars for experimental data
    - x_sim_trend: simulated trend x values
    - y_sim_trend: simulated trend y values
    - x_exp_trend: experimental trend x values
    - y_exp_trend: experimental trend y values
    - vline_x: x value for vertical line
    - vline_label: label for vertical line
    """
    if (x_sim is None or y_sim is None) and (x_exp is None or y_exp is None):
        raise ValueError(
            "You must provide either simulation data (x_sim, y_sim) or experimental data (x_exp, y_exp)."
        )

    handles = []

    # Plot Simulation (Always Blue)
    if dy_sim is not None:
        sim_line = ax.errorbar(x_sim, y_sim, yerr=dy_sim, fmt="bo-", label="Simulation")
        handles.append(sim_line)
    elif x_sim is not None and y_sim is not None:
        (sim_line,) = ax.plot(x_sim, y_sim, "bo-", label="Simulation")
        handles.append(sim_line)

    # Plot Experimental (Always Red)
    if dy_exp is not None:
        exp_line = ax.errorbar(x_exp, y_exp, yerr=dy_exp, fmt="ro-", label="Experiment")
        handles.append(exp_line)
    elif x_exp is not None and y_exp is not None:
        (exp_line,) = ax.plot(x_exp, y_exp, "ro-", label="Experiment")
        handles.append(exp_line)

    # Plot the simulation trend
    if x_sim_trend is not None:
        (sim_trend_line,) = ax.plot(
            x_sim_trend, y_sim_trend, "b--", label="Simulation trend"
        )
        handles.append(sim_trend_line)

    # Plot the experimental trend
    if x_exp_trend is not None:
        (exp_trend_line,) = ax.plot(
            x_exp_trend, y_exp_trend, "r--", label="Experiment trend"
        )
        handles.append(exp_trend_line)

    # Plot Cut-off Frequency
    if vline_x is not None:
        vline = ax.axvline(x=vline_x, color="k", linestyle="--", label=vline_label)
        handles.append(vline)

    # Extract strictly ordered handles and labels
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="lower left")


def axes_transfer_function(ax, x_lim=None, y_lim=None):
    """
    Sets axes labels and limits for transfer function plots
    """
    # Apply Consistent Formatting
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel(r"$\left|\tilde{V}_{40}(f) / \tilde{V}_0(f)\right|$")
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    else:
        ax.set_ylim(0, 1.2)
