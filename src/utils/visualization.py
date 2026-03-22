import matplotlib.pyplot as plt
import numpy as np


def apply_standard_style():
    """Applies global matplotlib style parameters for consistent formatting."""
    plt.rcParams.update(
        {
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "figure.figsize": (6.4, 4.8),  # Default single plot size
        }
    )


def plot_transfer_comparison(
    ax, f_sim, H_sim, f_exp, H_exp, dH_exp=None, f_cutoff=None, x_lim=None
):
    """
    Plots simulation vs experimental transfer functions on a given axis
    with strictly ordered legends and consistent colors.
    """
    handles = []

    # 1. Plot Simulation (Always Blue)
    (sim_line,) = ax.plot(f_sim, np.abs(H_sim), "bo-", label="Simulation")
    handles.append(sim_line)

    # 2. Plot Experimental (Always Red, optional error bars)
    if dH_exp is not None:
        exp_line = ax.errorbar(
            f_exp, np.abs(H_exp), yerr=dH_exp, fmt="ro-", label="Experimental data"
        )
        handles.append(exp_line)
    else:
        (exp_line,) = ax.plot(f_exp, np.abs(H_exp), "ro-", label="Experimental data")
        handles.append(exp_line)

    # 3. Plot Cut-off Frequency (Always Black Dashed)
    if f_cutoff is not None:
        vline = ax.axvline(
            x=f_cutoff, color="k", linestyle="--", label=r"Cut-off frequency $f_c$"
        )
        handles.append(vline)

    # 4. Apply Consistent Formatting
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("$V_{40}(f) / V_0(f)$")
    ax.set_ylim(0, 1.2)
    if x_lim:
        ax.set_xlim(x_lim)

    # 5. Extract strictly ordered handles and labels
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="lower left")
