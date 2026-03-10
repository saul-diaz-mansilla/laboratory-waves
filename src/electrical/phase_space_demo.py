import numpy as np
import matplotlib.pyplot as plt
import random


def main():
    # 1. Parameters
    frequency = 50.0  # Hz
    amplitude = 5.0  # Volts
    duration = 0.1  # Seconds (enough for a few cycles)
    sampling_rate = 10000  # Hz

    # Time array
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    omega = 2 * np.pi * frequency

    # 2. Generate Waves
    # Initial wave V0
    v0 = amplitude * np.sin(omega * t)

    # Wave 1: In phase (dphi = 0)
    v_in_phase = amplitude * np.sin(omega * t + 0.01 * random.random()) * 0.8

    # Wave 2: Out of phase (dphi = pi)
    v_out_phase = amplitude * np.sin(omega * t + np.pi + 0.01 * random.random()) * 1.15

    # Wave 3: pi/2 out of phase (dphi = pi/2)
    v_quadrature = amplitude * np.sin(omega * t + (np.pi / 2) - 0.3) * 0.63

    # 3. Plot Phase Space (Lissajous Figures)
    plt.figure()

    # Plot V0 vs V_in_phase (Straight line, slope +1)
    plt.plot(v0, v_in_phase, color="r", label=r"In Phase ($\Delta\phi=0$)", linewidth=2)

    # Plot V0 vs V_out_phase (Straight line, slope -1)
    plt.plot(
        v0,
        v_out_phase,
        label=r"Out of Phase ($\Delta\phi=\pi$)",
        color="b",
        linewidth=2,
    )

    # Plot V0 vs V_quadrature (Circle)
    plt.plot(
        v0,
        v_quadrature,
        label=r"Other $\Delta\phi$",
        color="k",
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel(r"$V_{0}$ [V]", fontsize=16)
    plt.ylabel(r"$V_{40}$ [V]", fontsize=16)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="upper right", fontsize=14)
    plt.tight_layout()
    plt.savefig("figures/phase_space_demo.png")
    plt.show()


if __name__ == "__main__":
    main()
