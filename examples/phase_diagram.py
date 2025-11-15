"""
Phase Diagram Generation Example
=================================

This script explores how cross-correlations depend on coupling strength,
generating a phase diagram for coupled KPZ systems.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from coupled_kpz_simulation import CoupledKPZSimulator
from analysis import compute_interface_width, cross_correlation
from visualization import plot_phase_diagram
import matplotlib.pyplot as plt

np.random.seed(42)

def explore_coupling_space():
    """Explore different coupling strengths and measure correlations"""
    
    print("=" * 60)
    print("PHASE DIAGRAM: COUPLING VS CORRELATION")
    print("=" * 60)
    
    # Define coupling strengths to explore
    coupling_strengths = np.linspace(-1.0, 1.0, 11)
    
    # Storage for results
    mean_correlations = []
    std_correlations = []
    
    # Fixed parameters
    L = 128
    t_max = 30.0
    nu = 1.0
    lam = 1.0
    D = 1.0
    
    print(f"\nExploring {len(coupling_strengths)} coupling values...")
    print(f"System size: L = {L}")
    print(f"Evolution time: t = {t_max}")
    
    for gamma in coupling_strengths:
        print(f"\nγ = {gamma:+.2f}...", end=" ")
        
        # Initialize simulator
        sim = CoupledKPZSimulator(L=L, dx=1.0, dt=0.01)
        
        # Symmetric coupling: gamma_12 = gamma_21 = gamma
        results = sim.run_simulation(
            t_max, nu, lam, gamma,
            nu, lam, gamma,
            D, D, D12=0,
            save_interval=2.0
        )
        
        # Calculate cross-correlations
        cross_corrs = []
        for h1, h2 in zip(results['h1_series'], results['h2_series']):
            corr = cross_correlation(h1, h2)
            cross_corrs.append(corr)
        
        cross_corrs = np.array(cross_corrs)
        mean_corr = np.mean(cross_corrs)
        std_corr = np.std(cross_corrs)
        
        mean_correlations.append(mean_corr)
        std_correlations.append(std_corr)
        
        print(f"⟨C₁₂⟩ = {mean_corr:+.4f} ± {std_corr:.4f}")
    
    # Plot phase diagram
    print("\n" + "=" * 60)
    print("GENERATING PHASE DIAGRAM")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(coupling_strengths, mean_correlations, yerr=std_correlations,
                fmt='bo-', linewidth=2, markersize=8, capsize=5, capthick=2)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Coupling Strength γ (symmetric)', fontsize=13)
    ax.set_ylabel('Mean Cross-Correlation ⟨C₁₂⟩', fontsize=13)
    ax.set_title('Phase Diagram: Correlation vs Coupling', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add shaded regions
    ax.axhspan(0, max(mean_correlations) * 1.1, alpha=0.1, color='blue', 
               label='Positive correlation region')
    ax.axhspan(min(mean_correlations) * 1.1, 0, alpha=0.1, color='red',
               label='Negative correlation region')
    
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('phase_diagram.png', dpi=300)
    print("Saved: phase_diagram.png")
    
    # Save data
    np.savez('phase_diagram_data.npz',
             coupling_strengths=coupling_strengths,
             mean_correlations=mean_correlations,
             std_correlations=std_correlations)
    print("Saved: phase_diagram_data.npz")
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE DIAGRAM SUMMARY")
    print("=" * 60)
    
    # Find transitions
    positive_mask = np.array(mean_correlations) > 0
    negative_mask = np.array(mean_correlations) < 0
    
    if np.any(positive_mask) and np.any(negative_mask):
        # Find approximate transition point
        zero_crossings = np.where(np.diff(np.sign(mean_correlations)))[0]
        if len(zero_crossings) > 0:
            transition_idx = zero_crossings[0]
            transition_gamma = (coupling_strengths[transition_idx] + 
                              coupling_strengths[transition_idx + 1]) / 2
            print(f"\nCorrelation changes sign near γ ≈ {transition_gamma:.2f}")
    
    print(f"\nPositive correlation for γ > 0: {np.sum(positive_mask & (coupling_strengths > 0))}/{np.sum(coupling_strengths > 0)} cases")
    print(f"Negative correlation for γ < 0: {np.sum(negative_mask & (coupling_strengths < 0))}/{np.sum(coupling_strengths < 0)} cases")
    
    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    explore_coupling_space()
