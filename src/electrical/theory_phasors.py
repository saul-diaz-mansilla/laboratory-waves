import numpy as np
import matplotlib.pyplot as plt

# 1. Circuit Parameters
L = 330e-6             # Inductance: 330 uH
C = 15e-9              # Bulk Capacitance: 15 nF
C_end = 7.5e-9         # Boundary Capacitance: 7.5 nF
N = 41                 # Number of unknown nodes (V_1 to V_40)

# Calculate characteristic impedance at low frequencies to set Rout
# This prevents extreme reflections at the boundary
R_out = np.sqrt(L / C) # ~148.3 ohms

# 2. Frequency Range Setup
omega_c = 2 / np.sqrt(L * C)       # Cut-off angular frequency
f_c = omega_c / (2 * np.pi)        # Cut-off frequency in Hz (~71.5 kHz)

# Sweep from 100 Hz up to the cut-off frequency
frequencies = np.linspace(100, f_c, 1000)
ratio_V38_V0 = []

# 3. Solve the System for Each Frequency
for f in frequencies:
    omega = 2 * np.pi * f
    
    # Initialize the tridiagonal matrix M (complex to handle phase shifts from Rout)
    M = np.zeros((N, N), dtype=complex)
    
    # Initialize the forcing vector b 
    # (Assuming V_0 = 1, so the result is automatically the ratio V_n / V_0)
    b = np.zeros(N, dtype=complex)
    b[0] = 1 / L  
    
    # Fill Node 0
    M[0, 0] = 2 / L - omega**2 * C
    M[0, 1] = -1 / L
    
    # Fill Bulk Nodes 2 to 39
    for i in range(1, N - 1):
        M[i, i - 1] = -1 / L
        M[i, i]     = 2 / L - omega**2 * C
        M[i, i + 1] = -1 / L
        
    # Fill Output Node 40
    M[N - 1, N - 2] = -1 / L
    M[N - 1, N - 1] = 1 / L - omega**2 * C_end + 1j * omega / R_out
    
    # Solve the linear system M * V = b directly
    V = np.linalg.solve(M, b)
    
    # Extract Node 38
    V38 = np.abs(V[38])
    ratio_V38_V0.append(V38)

# 4. Plotting the Results
plt.figure(figsize=(10, 6))
plt.plot(frequencies / 1000, ratio_V38_V0, color='b', linewidth=2, label='$|V_{38} / V_0|$')
plt.axvline(x=f_c / 1000, color='r', linestyle='--', label=f'Theoretical Cutoff ({f_c/1000:.1f} kHz)')

plt.title('Amplitude Ratio at Node 38 vs. Frequency', fontsize=14)
plt.xlabel('Frequency (kHz)', fontsize=12)
plt.ylabel('Amplitude Ratio $|V_{38} / V_0|$', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()