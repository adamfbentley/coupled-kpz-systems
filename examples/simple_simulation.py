"""
Simple Coupled KPZ Simulation Example
=====================================

This script demonstrates basic usage of the coupled KPZ simulator
with symmetric coupling.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from coupled_kpz_simulation import CoupledKPZSimulator
from analysis import compute_interface_width, scaling_analysis, cross_correlation
from visualization import plot_interface_evolution, plot_scaling_analysis

# Set random seed for reproducibility
np.random.seed(42)

def run_simple_simulation():
    """Run a basic coupled KPZ simulation with symmetric coupling"""
    
    print("=" * 60)
    print("SIMPLE COUPLED KPZ SIMULATION")
    print("=" * 60)
    
    # Initialize simulator
    sim = CoupledKPZSimulator(L=128, dx=1.0, dt=0.01)
    
    # Physical parameters
    nu1 = nu2 = 1.0          # Surface tension
    lambda1 = lambda2 = 1.0   # KPZ nonlinearity
    lambda12 = lambda21 = 0.5 # Symmetric coupling
    D11 = D22 = 1.0          # Noise strength
    
    print(f"\nSimulation Parameters:")
    print(f"  System size: L = {sim.L}")
    print(f"  Coupling: λ₁₂ = λ₂₁ = {lambda12} (symmetric)")
    print(f"  Time step: dt = {sim.dt}")
    
    # Run simulation
    t_max = 50.0
    save_interval = 2.0
    
    print(f"\nRunning simulation to t = {t_max}...")
    
    results = sim.run_simulation(
        t_max, nu1, lambda1, lambda12, 
        nu2, lambda2, lambda21,
        D11, D22, D12=0,
        save_interval=save_interval
    )
    
    # Extract results
    times = np.array(results['times'])
    widths_h1 = []
    widths_h2 = []
    cross_corrs = []
    
    for h1, h2 in zip(results['h1_series'], results['h2_series']):
        widths_h1.append(compute_interface_width(h1))
        widths_h2.append(compute_interface_width(h2))
        cross_corrs.append(cross_correlation(h1, h2))
    
    widths_h1 = np.array(widths_h1)
    widths_h2 = np.array(widths_h2)
    cross_corrs = np.array(cross_corrs)
    
    # Analyze results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    print("\nInterface 1:")
    scaling_h1 = scaling_analysis(widths_h1, times, verbose=True)
    
    print("\nInterface 2:")
    scaling_h2 = scaling_analysis(widths_h2, times, verbose=True)
    
    print("\nCross-Correlation:")
    print(f"Mean: {np.mean(cross_corrs):.4f} ± {np.std(cross_corrs):.4f}")
    
    # Visualize
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    plot_interface_evolution(
        times, widths_h1, widths_h2,
        title="Symmetric Coupling (γ=0.5)",
        save_path="simple_evolution.png"
    )
    
    if scaling_h1:
        plot_scaling_analysis(
            times, widths_h1, scaling_h1,
            interface_name="Interface 1",
            save_path="simple_scaling_h1.png"
        )
    
    if scaling_h2:
        plot_scaling_analysis(
            times, widths_h2, scaling_h2,
            interface_name="Interface 2",
            save_path="simple_scaling_h2.png"
        )
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - simple_evolution.png")
    print("  - simple_scaling_h1.png")
    print("  - simple_scaling_h2.png")

if __name__ == "__main__":
    run_simple_simulation()
