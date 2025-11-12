# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 12:53:22 2025

@author: adamf
"""

# PHYS345 Numerical Project: 1D Thin-Film Growth Simulation
#
# A clear, procedural script to solve the 1D stochastic growth equation.
# This code implements key concepts from the project guide, including a direct
# comparison between the KPZ and Edwards-Wilkinson (EW) models.

import numpy as np
import matplotlib.pyplot as plt

def run_1d_simulation(L, T, dt, dx, v_coeff, lam_coeff, D_coeff, growth_rate):
    """
    Solves the 1D stochastic growth equation using a finite difference method.

    This function simulates the evolution of the surface height profile 'h' over time.
    It's designed to be called with different parameters to model various physical cases.
    """
    # Calculate grid points and time steps from the physical parameters
    nx = int(L / dx)  # Number of spatial points
    nt = int(T / dt)  # Number of time steps

    # Initialize the surface. Starting with a tiny amount of random noise helps
    # to seed the growth process, which is more realistic than a perfect flat line.
    h = 0.01 * np.random.randn(nx)
    
    # Pre-calculate the noise amplitude for efficiency.
    # This scaling ensures the noise term is physically correct.
    noise_amplitude = np.sqrt(2 * D_coeff / dx) * np.sqrt(dt)

    # The main simulation loop, stepping forward in time
    for _ in range(nt):
        # Use np.roll for periodic boundary conditions, which is efficient
        h_plus = np.roll(h, -1)
        h_minus = np.roll(h, 1)

        # Calculate derivatives using central finite differences
        # 1. The gradient, for the non-linear term
        grad_h = (h_plus - h_minus) / (2 * dx)
        # 2. The Laplacian, for the smoothing/diffusion term
        laplacian_h = (h_plus - 2 * h + h_minus) / (dx**2)

        # The non-linear growth term is proportional to the gradient squared
        grad_h_sq = grad_h**2

        # The stochastic noise term combines a random fluctuation and an average growth rate
        eta = noise_amplitude * np.random.randn(nx) + growth_rate

        # The full growth equation (Euler method update step)
        # Note that if lam_coeff is zero, the non-linear term vanishes (EW case)
        h += dt * (v_coeff * laplacian_h + (lam_coeff / 2) * grad_h_sq + eta)

    return h


# --- Main script execution block ---
if __name__ == "__main__":
    
    ### PART 1: Run the Simulations ###
    
    # Define the core physical parameters for our thin-film system
    # This is done once, so both simulations are comparable.
    SIM_PARAMS = {
        'L': 512.0, 'T': 100.0, 'dt': 0.005, 'dx': 1.0,
        'v_coeff': 1.0, 'D_coeff': 1.0, 'growth_rate': 0.1
    }

    print("--- Running KPZ Simulation (with non-linear term) ---")
    # For the KPZ case, we include the non-linear lambda term.
    h_final_kpz = run_1d_simulation(lam_coeff=1.0, **SIM_PARAMS)

    print("--- Running EW Simulation (linear only) for comparison ---")
    # For the EW case, we set lambda to zero, turning off the non-linear growth.
    h_final_ew = run_1d_simulation(lam_coeff=0.0, **SIM_PARAMS)
    
    print("Simulations finished. Generating plots...")

    ### PART 2: Plotting and Analysis ###

    # Create a single figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Comparison of 1D Thin-Film Growth Models', fontsize=16)

    # --- Left Panel: Final Surface Profiles ---
    x_axis = np.arange(len(h_final_kpz)) * SIM_PARAMS['dx']
    ax1.plot(x_axis, h_final_kpz, label='KPZ Model (Non-linear)', color='darkblue')
    ax1.plot(x_axis, h_final_ew, label='EW Model (Linear)', color='red', alpha=0.7)
    ax1.set_title(f'Final Surface Profile at t={SIM_PARAMS["T"]}s')
    ax1.set_xlabel('Position (nm)')
    ax1.set_ylabel('Height (nm)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Right Panel: Power Spectral Density (PSD) Analysis ---
    # We analyze the KPZ case to check for its characteristic roughness.
    
    # First, remove the mean height (DC offset) for a cleaner spectrum
    h_kpz_detrended = h_final_kpz - np.mean(h_final_kpz)
    
    # Calculate the FFT and then the Power Spectrum
    fft_h = np.fft.fft(h_kpz_detrended)
    psd = np.abs(fft_h)**2 / SIM_PARAMS['L'] # Normalize by length

    # Get the corresponding spatial frequencies
    # We only need the first half (positive frequencies)
    freq = np.fft.fftfreq(len(h_kpz_detrended), d=SIM_PARAMS['dx'])
    positive_freq_mask = freq > 0
    freq = freq[positive_freq_mask]
    psd = psd[positive_freq_mask]

    # Plot the PSD on a log-log scale to identify power-law behavior
    ax2.loglog(freq, psd, 'o', markersize=4, label='KPZ Spectrum')
    
    # Add a reference line to guide the eye. The theoretical slope for 1D KPZ
    # is -5/3, but a q^-2 slope is often used for visual comparison.
    # We scale it to match the data's magnitude for a nice fit.
    ref_slope = 1.0 * freq**(-2) # A simple q^-2 line
    ax2.loglog(freq, ref_slope, '--', color='gray', label='Reference Slope ($q^{-2}$)')
    
    ax2.set_title('Power Spectral Density of KPZ Surface')
    ax2.set_xlabel('Spatial Frequency, q (1/nm)')
    ax2.set_ylabel('Power')
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    # Show the final plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()