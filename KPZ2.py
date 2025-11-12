# PHYS345 Numerical Project: Simulation and Analysis of Surface Growth
#
# This script simulates the evolution of a surface according to the KPZ equation
# and then visualizes the result in three ways:
# 1. A 3D surface plot for overall morphology.
# 2. A 2D cross-section plot (position vs. height) for a detailed profile.
# 3. A Power Spectral Density (PSD) plot for statistical analysis of roughness.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm # A library for creating smart progress bars

def simulate_surface_growth(params):
    """Runs the main simulation loop for surface growth."""
    # Unpack parameters
    N, total_time, dt = params['grid_size'], params['total_time'], params['time_step']
    nu, lam, gamma = params['nu'], params['lambda'], params['gamma']
    noise_strength, growth_rate, dx = params['noise_strength'], params['growth_rate'], params['dx']

    # --- Initialization ---
    h = np.random.uniform(0, 0.1, size=(N, N))
    num_steps = int(total_time / dt)
    
    print("Starting simulation...")
    for step in tqdm(range(num_steps), desc="Simulating Growth"):
        # --- Calculate Spatial Derivatives (Vectorized) ---
        h_plus_x = np.roll(h, -1, axis=1)
        h_minus_x = np.roll(h, 1, axis=1)
        h_plus_y = np.roll(h, -1, axis=0)
        h_minus_y = np.roll(h, 1, axis=0)
        
        laplacian_h = (h_plus_x + h_minus_x + h_plus_y + h_minus_y - 4 * h) / (dx**2)
        
        dh_dx = (h_plus_x - h_minus_x) / (2 * dx)
        dh_dy = (h_plus_y - h_minus_y) / (2 * dx)
        grad_h_sq = dh_dx**2 + dh_dy**2

        # --- Generate Stochastic Noise Term (η) ---
        noise = np.random.randn(N, N) * np.sqrt(2 * noise_strength * dt) / dx
        
        # --- Assemble the Equation ---
        dh_dt = (nu * laplacian_h + 0.5 * lam * grad_h_sq + noise + growth_rate)
        
        # --- Update Height Field ---
        h += dh_dt * dt

    print("Simulation finished.")
    return h

def plot_surface(h, title="Final Surface Morphology (3D View)"):
    """Creates a 3D plot of the surface height field."""
    N = h.shape[0]
    x, y = np.arange(0, N, 1), np.arange(0, N, 1)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(h, cmap=plt.cm.viridis, vert_exag=0.1, blend_mode='soft')
    
    ax.plot_surface(X, Y, h, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)
    ax.set_xlabel('X position'); ax.set_ylabel('Y position'); ax.set_zlabel('Height')
    ax.set_title(title)
    plt.show()

def plot_cross_section(h, title="Final Surface Cross-Section (2D View)"):
    """
    Plots a 2D cross-section of the surface height field.
    This shows the height profile along the central horizontal line.

    Args:
        h (numpy.ndarray): The 2D height field.
        title (str): The title for the plot.
    """
    N = h.shape[0]
    # Take a slice through the middle of the grid (at y = N/2)
    cross_section_data = h[N // 2, :]
    # Create an array for the x-axis positions
    x_positions = np.arange(N)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_positions, cross_section_data, color='darkblue')

    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Height')
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.6)
    # Set a tight layout to make sure all labels fit
    plt.tight_layout()
    plt.show()


def calculate_and_plot_psd(h, dx, title="Power Spectral Density"):
    """
    Calculates and plots the radially averaged Power Spectral Density (PSD)
    of the final surface.
    """
    N = h.shape[0]
    h_k = np.fft.fft2(h)
    psd_2d = np.abs(h_k)**2
    q_freq = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    q_x, q_y = np.meshgrid(q_freq, q_freq)
    q_radial = np.sqrt(q_x**2 + q_y**2)
    
    q_bins = np.arange(0, q_radial.max(), 2 * np.pi / (N * dx))
    psd_1d = np.zeros(len(q_bins) - 1)
    
    for i in range(len(q_bins) - 1):
        mask = (q_radial >= q_bins[i]) & (q_radial < q_bins[i+1])
        if np.any(mask):
            psd_1d[i] = psd_2d[mask].mean()
            
    q_vals = 0.5 * (q_bins[1:] + q_bins[:-1])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(q_vals[1:], psd_1d[1:], 'o', label='Simulated PSD')
    
    q_fit = q_vals[q_vals > 0.5]
    if len(q_fit) > 0 and len(psd_1d[q_vals > 0.5]) > 0:
        fit = q_fit**(-3) * psd_1d[q_vals > 0.5][0] / q_fit[0]**(-3)
        ax.plot(q_fit, fit, 'r--', label=r'Reference line ($q^{-3}$)')
    
    ax.set_xlabel('Spatial Frequency, q (radians/unit length)')
    ax.set_ylabel('Power Spectral Density, PSD(q)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--")
    plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    
    simulation_parameters = {
        'grid_size': 256,
        'total_time': 100.0,
        'time_step': 0.01,
        'dx': 1.0,
        'nu': 1.0,
        'lambda': 4.0,
        'gamma': 0.0,
        'noise_strength': 0.5,
        'growth_rate': 0.0,
    }

    # Run the simulation
    final_surface = simulate_surface_growth(simulation_parameters)
    
    # Visualize the final result in 3D
    plot_surface(final_surface)
    
    # --- NEW ---
    # Visualize the final result as a 2D cross-section
    plot_cross_section(final_surface)

    # Calculate and plot the PSD as required by the brief
    calculate_and_plot_psd(final_surface, simulation_parameters['dx'])