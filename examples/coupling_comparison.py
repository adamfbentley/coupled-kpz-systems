"""
Comparison of Coupling Types
============================

This script compares symmetric, antisymmetric, and asymmetric coupling
to demonstrate different synchronization behaviors.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from coupled_kpz_simulation import CoupledKPZSimulator
from analysis import compute_interface_width, cross_correlation, analyze_correlation_evolution
from visualization import plot_comprehensive_analysis
import matplotlib.pyplot as plt

np.random.seed(42)

def compare_coupling_types():
    """Compare different coupling configurations"""
    
    print("=" * 60)
    print("COUPLING TYPE COMPARISON")
    print("=" * 60)
    
    # Define coupling scenarios
    scenarios = {
        'Symmetric': {'gamma_12': 0.5, 'gamma_21': 0.5},
        'Antisymmetric': {'gamma_12': 0.5, 'gamma_21': -0.5},
        'Asymmetric': {'gamma_12': 0.8, 'gamma_21': 0.2}
    }
    
    # Common parameters
    L = 128
    t_max = 40.0
    nu = 1.0
    lam = 1.0
    D = 1.0
    
    # Storage for results
    all_results = {}
    
    # Run simulations
    for scenario_name, params in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Running: {scenario_name} Coupling")
        print(f"  γ₁₂ = {params['gamma_12']}, γ₂₁ = {params['gamma_21']}")
        print(f"{'='*60}")
        
        sim = CoupledKPZSimulator(L=L, dx=1.0, dt=0.01)
        
        results = sim.run_simulation(
            t_max, nu, lam, params['gamma_12'],
            nu, lam, params['gamma_21'],
            D, D, D12=0,
            save_interval=2.0
        )
        
        # Process results
        times = np.array(results['times'])
        widths_h1 = np.array([compute_interface_width(h) for h in results['h1_series']])
        widths_h2 = np.array([compute_interface_width(h) for h in results['h2_series']])
        cross_corrs = np.array([cross_correlation(h1, h2) 
                               for h1, h2 in zip(results['h1_series'], results['h2_series'])])
        
        mean_corr = np.mean(cross_corrs)
        std_corr = np.std(cross_corrs)
        
        print(f"\nCross-correlation: {mean_corr:+.4f} ± {std_corr:.4f}")
        
        all_results[scenario_name] = {
            'times': times,
            'widths_h1': widths_h1,
            'widths_h2': widths_h2,
            'cross_corrs': cross_corrs,
            'params': params,
            'mean_corr': mean_corr,
            'std_corr': std_corr
        }
    
    # Create comparison plot
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    colors = {'Symmetric': 'blue', 'Antisymmetric': 'red', 'Asymmetric': 'green'}
    
    for idx, (scenario_name, data) in enumerate(all_results.items()):
        times = data['times']
        color = colors[scenario_name]
        
        # Row 1: Interface widths
        axes[0, idx].plot(times, data['widths_h1'], color=color, linewidth=2, alpha=0.7, label='h₁')
        axes[0, idx].plot(times, data['widths_h2'], color='orange', linewidth=2, alpha=0.7, label='h₂')
        axes[0, idx].set_title(f'{scenario_name}\nγ₁₂={data["params"]["gamma_12"]}, γ₂₁={data["params"]["gamma_21"]}',
                              fontsize=11, fontweight='bold')
        axes[0, idx].set_xlabel('Time', fontsize=10)
        axes[0, idx].set_ylabel('Width W(t)', fontsize=10)
        axes[0, idx].legend(fontsize=9)
        axes[0, idx].grid(True, alpha=0.3)
        
        # Row 2: Log-log scaling
        valid_mask = (times > 0) & (data['widths_h1'] > 0)
        axes[1, idx].loglog(times[valid_mask], data['widths_h1'][valid_mask], 
                           'o', color=color, markersize=4, alpha=0.5)
        axes[1, idx].set_xlabel('Time (log)', fontsize=10)
        axes[1, idx].set_ylabel('W(t) (log)', fontsize=10)
        axes[1, idx].set_title('Scaling Analysis', fontsize=10)
        axes[1, idx].grid(True, alpha=0.3, which='both')
        
        # Row 3: Cross-correlations
        axes[2, idx].plot(times, data['cross_corrs'], color=color, linewidth=2)
        axes[2, idx].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[2, idx].axhline(y=data['mean_corr'], color='red', linestyle='--', linewidth=2,
                            label=f"Mean={data['mean_corr']:+.3f}")
        axes[2, idx].set_xlabel('Time', fontsize=10)
        axes[2, idx].set_ylabel('Cross-Corr C₁₂', fontsize=10)
        axes[2, idx].set_title('Cross-Correlation', fontsize=10)
        axes[2, idx].legend(fontsize=9)
        axes[2, idx].grid(True, alpha=0.3)
    
    fig.suptitle('Coupled KPZ: Comparison of Coupling Types', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('coupling_comparison.png', dpi=300)
    print("Saved: coupling_comparison.png")
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"\n{'Scenario':<15} {'γ₁₂':<8} {'γ₂₁':<8} {'⟨C₁₂⟩':<12} {'Behavior'}")
    print("-" * 60)
    
    for scenario_name, data in all_results.items():
        mean_corr = data['mean_corr']
        
        if mean_corr > 0.05:
            behavior = "Positive correlation"
        elif mean_corr < -0.05:
            behavior = "Negative correlation"
        else:
            behavior = "Weak/no correlation"
        
        print(f"{scenario_name:<15} {data['params']['gamma_12']:<8.2f} "
              f"{data['params']['gamma_21']:<8.2f} {mean_corr:+.4f} ± {data['std_corr']:.4f}  "
              f"{behavior}")
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    compare_coupling_types()
