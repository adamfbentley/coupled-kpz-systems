# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 12:01:44 2025

@author: adamf
"""

# PHYS345 Numerical Project: Simulation and Analysis of Surface Growth
#
# This version includes a function to generate a live, animated plot
# of the surface cross-section as the simulation runs.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def simulate_surface_growth_animated(params, plot_interval=50):
    """
    Runs the simulation and displays an animated plot of the central
    cross-section evolving over time.

    Args:
        params (dict): Dictionary of simulation parameters.
        plot_interval (int): Update the plot every N steps. A larger number
                             makes the simulation run faster but the animation
                             will be less smooth.
    """
    # --- Unpack Parameters ---
    N, total_time, dt = params['grid_size'], params['total_time'], params['time_step']
    nu, lam, noise_strength, dx = params['nu'], params['lambda'], params['noise_strength'], params['dx']
    
    # --- Initialization ---
    h = np.random.uniform(0, 0.1, size=(N, N))
    num_steps = int(total_time / dt)

    # --- Set up the animation plot ---
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create the initial line object. We will only update its data later.
    x_positions = np.arange(N) * dx
    cross_section_data = h[N // 2, :]
    line, = ax.plot(x_positions, cross_section_data, color='darkblue') # The comma is important!
    
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Height')
    # Set fixed y-axis limits to prevent the plot from rescaling distractingly
    # You may need to adjust these based on your parameters
    ax.set_ylim(-15, 15) 
    ax.grid(True, which="both", ls="--", alpha=0.6)
    
    print("Starting animated simulation...")
    
    # --- Main Simulation Loop ---
    for step in tqdm(range(num_steps), desc="Simulating and Animating"):
        # Physics calculations (same as before)
        h_plus_x = np.roll(h, -1, axis=1)
        h_minus_x = np.roll(h, 1, axis=1)
        h_plus_y = np.roll(h, -1, axis=0)
        h_minus_y = np.roll(h, 1, axis=0)
        
        laplacian_h = (h_plus_x + h_minus_x + h_plus_y + h_minus_y - 4 * h) / (dx**2)
        dh_dx = (h_plus_x - h_minus_x) / (2 * dx)
        dh_dy = (h_plus_y - h_minus_y) / (2 * dx)
        grad_h_sq = dh_dx**2 + dh_dy**2
        noise = np.random.randn(N, N) * np.sqrt(2 * noise_strength * dt) / dx
        dh_dt = (nu * laplacian_h + 0.5 * lam * grad_h_sq + noise)
        h += dh_dt * dt
        
        # --- Update the plot periodically ---
        if step % plot_interval == 0:
            # Get the current cross-section
            cross_section_data = h[N // 2, :]
            
            # Update the y-data of the line object
            line.set_ydata(cross_section_data)
            
            # Update the title to show the current time
            current_time = step * dt
            ax.set_title(f"Surface Cross-Section at Time = {current_time:.2f}")
            
            # Redraw the canvas
            fig.canvas.draw()
            # Give the GUI a moment to process the update
            plt.pause(0.001)

    plt.ioff() # Turn off interactive mode
    ax.set_title(f"Final Surface Cross-Section at Time = {total_time:.2f}")
    print("Animation finished. The final plot is now displayed.")
    plt.show() # Keep the final plot window open
    
    return h

# The other plotting functions (plot_surface, calculate_and_plot_psd) remain the same.
# (You can copy them from the previous response)

# --- Main execution block ---
if __name__ == "__main__":
    
    simulation_parameters = {
        'grid_size': 256,
        'total_time': 200.0,      # A longer time makes for a better animation
        'time_step': 0.01,
        'dx': 1.0,
        'nu': 1.0,
        'lambda': 4.0,
        'gamma': 0.0,
        'noise_strength': 0.5,
        'growth_rate': 0.0,
    }

    # --- Run the ANIMATED simulation ---
    # This function will now produce the live plot as it runs.
    # The plot_interval determines how often the frame updates.
    # A smaller value gives a smoother animation but slows down the calculation.
    final_surface = simulate_surface_growth_animated(simulation_parameters, plot_interval=100)
    
    # You can still generate the final 3D and PSD plots afterwards
    # plot_surface(final_surface)
    # calculate_and_plot_psd(final_surface, simulation_parameters['dx'])