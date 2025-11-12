# PHYS345 Numerical Project: Simulation of Thin Film Growth
#
# A simplified script to model surface growth via the KPZ equation.
# This code focuses on the core requirements of the project brief:
# 1. Numerically solve the equation using a finite difference method.
# 2. Visualize the final surface profile with a 2D cross-section.
# 3. Analyze the roughness using a Fast-Fourier Transform to get the PSD.

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- All parameters are defined in one place for easy modification ---
PARAMS = {
    'grid_size': 256,        # The N x N size of the simulation grid
    'total_time': 10.0,     # How long the simulation runs in physical time
    'time_step': 0.01,       # The size of each discrete time step (dt)
    'dx': 1.0,               # The spatial step size
    'nu': 1.0,               # ν: The surface tension / smoothing term
    'lambda': 4.0,           # λ: The non-linear growth term (key for KPZ)
    'noise_strength': 0.5,   # D: The magnitude of the random noise
}

# --- 1. The Main Simulation ---

# Start with a nearly flat surface with small random fluctuations
h = np.random.uniform(0, 0.1, size=(PARAMS['grid_size'], PARAMS['grid_size']))

# Calculate the total number of steps to run
num_steps = int(PARAMS['total_time'] / PARAMS['time_step'])

print("Starting simulation...")
# Loop through time to evolve the surface
for _ in tqdm(range(num_steps), desc="Simulating Growth"):
    # Calculate derivatives using shifted arrays for efficiency and periodic boundaries
    h_plus_x = np.roll(h, -1, axis=1)
    h_minus_x = np.roll(h, 1, axis=1)
    h_plus_y = np.roll(h, -1, axis=0)
    h_minus_y = np.roll(h, 1, axis=0)
    
    # The Laplacian (∇²h), for the smoothing term
    laplacian_h = (h_plus_x + h_minus_x + h_plus_y + h_minus_y - 4 * h) / (PARAMS['dx']**2)

    # The gradient squared (|∇h|²) for the non-linear term
    dh_dx = (h_plus_x - h_minus_x) / (2 * PARAMS['dx'])
    dh_dy = (h_plus_y - h_minus_y) / (2 * PARAMS['dx'])
    grad_h_sq = dh_dx**2 + dh_dy**2

    # The stochastic noise term (η)
    noise = np.random.randn(PARAMS['grid_size'], PARAMS['grid_size']) * np.sqrt(2 * PARAMS['noise_strength'] * PARAMS['time_step']) / PARAMS['dx']
    
    # The full KPZ equation in discrete form: dh/dt = ...
    dh_dt = (PARAMS['nu'] * laplacian_h + 0.5 * PARAMS['lambda'] * grad_h_sq + noise)
    
    # Update the height field using a simple forward Euler step
    h += dh_dt * PARAMS['time_step']

print("Simulation finished.")

# --- 2. Visualize the Final Surface Profile ---

print("Generating plots...")
# Take a slice right through the middle of the final surface
cross_section = h[PARAMS['grid_size'] // 2, :]
x_positions = np.arange(PARAMS['grid_size'])

plt.figure(figsize=(10, 6))
plt.plot(x_positions, cross_section, color='darkblue')
plt.title('Final Surface Profile (2D Cross-Section)')
plt.xlabel('Position (x)')
plt.ylabel('Height')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- 3. Analyze the Roughness with Power Spectral Density ---

# Calculate the 2D Fast-Fourier Transform of the final surface
h_k = np.fft.fft2(h)
# Calculate the Power Spectrum (the squared magnitude of the Fourier components)
psd_2d = np.abs(h_k)**2

# Define the spatial frequencies (q) for the grid
q_freq = np.fft.fftfreq(PARAMS['grid_size'], d=PARAMS['dx'])
q_x, q_y = np.meshgrid(q_freq, q_freq)
q_radial = np.sqrt(q_x**2 + q_y**2)

# Radially average the 2D PSD into 1D bins
# This gives the average power for each spatial frequency magnitude
bin_width = q_freq.max() / (PARAMS['grid_size'] / 2)
q_bins = np.arange(0, q_radial.max(), bin_width)
psd_1d = np.zeros(len(q_bins) - 1)
q_vals = 0.5 * (q_bins[1:] + q_bins[:-1])

for i in range(len(q_bins) - 1):
    mask = (q_radial >= q_bins[i]) & (q_radial < q_bins[i+1])
    if np.any(mask):
        psd_1d[i] = psd_2d[mask].mean()

# Plot the PSD on a log-log scale to identify power-law behavior
plt.figure(figsize=(8, 6))
plt.loglog(q_vals[1:], psd_1d[1:], 'o', markersize=5, label='Simulated PSD')

# Add a reference line to guide the eye and show what a power law looks like
q_fit = q_vals[q_vals > 2]
if len(q_fit) > 0:
    fit = q_fit**(-4) * psd_1d[q_vals > 2][0] / q_fit[0]**(-3)
    plt.plot(q_fit, fit, 'r--', label=r'Reference line ($q^{-4}$)')

plt.xlabel('Spatial Frequency, q')
plt.ylabel('Power Spectral Density, PSD(q)')
plt.title('Roughness Analysis: Power Spectrum')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()

print("Script finished.")