"""
Coupled KPZ Equation Numerical Simulation
=========================================

This script implements finite difference simulations of coupled KPZ equations
to validate theoretical predictions and generate genuine research data.

Author: A. F. Bentley
Date: October 2025
Course: PHYS 489 - Advanced Topics in Experimental Physics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import pickle
import time
from numba import jit

# Set random seed for reproducibility
np.random.seed(42)

@jit(nopython=True)
def compute_laplacian(h, dx):
    """Compute discrete Laplacian with periodic boundaries."""
    L = len(h)
    lap = np.zeros(L)
    for i in range(L):
        lap[i] = (h[(i+1)%L] - 2*h[i] + h[(i-1)%L]) / (dx**2)
    return lap

@jit(nopython=True)
def compute_gradient_squared(h, dx):
    """Compute (∇h)² using central differences."""
    L = len(h)
    grad_sq = np.zeros(L)
    for i in range(L):
        grad = (h[(i+1)%L] - h[(i-1)%L]) / (2*dx)
        grad_sq[i] = grad**2
    return grad_sq

class CoupledKPZSimulator:
    """
    Numerical simulator for coupled KPZ equations using finite differences.
    
    Implements the system:
    ∂h₁/∂t = ν₁∇²h₁ + (λ₁/2)(∇h₁)² + (λ₁₂/2)(∇h₂)² + η₁(x,t)
    ∂h₂/∂t = ν₂∇²h₂ + (λ₂/2)(∇h₂)² + (λ₂₁/2)(∇h₁)² + η₂(x,t)
    """
    
    def __init__(self, L=512, dx=1.0, dt=0.01):
        """
        Initialize simulation parameters.
        
        Parameters:
        -----------
        L : int
            System size (number of lattice points)
        dx : float
            Spatial discretization
        dt : float
            Time step
        """
        self.L = L
        self.dx = dx
        self.dt = dt
        self.x = np.arange(L) * dx
        
        # Initialize height fields
        self.h1 = np.random.normal(0, 0.1, L)
        self.h2 = np.random.normal(0, 0.1, L)
        
        # Remove average height
        self.h1 -= np.mean(self.h1)
        self.h2 -= np.mean(self.h2)
        
        # Storage for time series
        self.time_series_h1 = []
        self.time_series_h2 = []
        self.times = []
        
    def laplacian(self, h):
        """Compute discrete Laplacian with periodic boundaries."""
        return compute_laplacian(h, self.dx)
    
    def gradient_squared(self, h):
        """Compute (∇h)² using central differences."""
        return compute_gradient_squared(h, self.dx)
    
    def step(self, nu1, lambda1, lambda12, nu2, lambda2, lambda21, D11, D22, D12=0):
        """
        Perform one time step of the coupled KPZ evolution.
        
        Parameters:
        -----------
        nu1, nu2 : float
            Surface tension coefficients
        lambda1, lambda2 : float
            Nonlinear coefficients (self-interaction)
        lambda12, lambda21 : float
            Cross-coupling coefficients
        D11, D22 : float
            Noise strengths (auto-correlation)
        D12 : float
            Cross-correlated noise strength
        """
        # Generate correlated noise
        if D12 == 0:
            # Uncorrelated noise
            eta1 = np.random.normal(0, np.sqrt(2*D11*self.dt), self.L)
            eta2 = np.random.normal(0, np.sqrt(2*D22*self.dt), self.L)
        else:
            # Correlated noise using Cholesky decomposition
            cov_matrix = np.array([[2*D11*self.dt, 2*D12*self.dt],
                                   [2*D12*self.dt, 2*D22*self.dt]])
            noise = np.random.multivariate_normal([0, 0], cov_matrix, self.L)
            eta1 = noise[:, 0]
            eta2 = noise[:, 1]
        
        # Compute spatial derivatives
        lap1 = self.laplacian(self.h1)
        lap2 = self.laplacian(self.h2)
        grad1_sq = self.gradient_squared(self.h1)
        grad2_sq = self.gradient_squared(self.h2)
        
        # KPZ evolution equations
        dh1_dt = nu1 * lap1 + 0.5 * lambda1 * grad1_sq + 0.5 * lambda12 * grad2_sq + eta1
        dh2_dt = nu2 * lap2 + 0.5 * lambda2 * grad2_sq + 0.5 * lambda21 * grad1_sq + eta2
        
        # Update heights
        self.h1 += self.dt * dh1_dt
        self.h2 += self.dt * dh2_dt
        
        # Remove mean to prevent drift
        self.h1 -= np.mean(self.h1)
        self.h2 -= np.mean(self.h2)
    
    def run_simulation(self, t_max, nu1, lambda1, lambda12, nu2, lambda2, lambda21, 
                      D11, D22, D12=0, save_interval=10):
        """
        Run the full simulation and collect time series data.
        """
        n_steps = int(t_max / self.dt)
        save_steps = save_interval / self.dt
        
        print(f"Running simulation for {t_max} time units ({n_steps} steps)")
        print(f"System size: {self.L}, dx: {self.dx}, dt: {self.dt}")
        print(f"Parameters: nu1={nu1:.2e}, lambda1={lambda1:.2e}, lambda12={lambda12:.2e}")
        print(f"           nu2={nu2:.2e}, lambda2={lambda2:.2e}, lambda21={lambda21:.2e}")
        print(f"           D11={D11:.2e}, D22={D22:.2e}, D12={D12:.2e}")
        
        start_time = time.time()
        
        for step in range(n_steps):
            self.step(nu1, lambda1, lambda12, nu2, lambda2, lambda21, D11, D22, D12)
            
            # Save data periodically
            if step % int(save_steps) == 0:
                self.time_series_h1.append(self.h1.copy())
                self.time_series_h2.append(self.h2.copy())
                self.times.append(step * self.dt)
                
                if step % (n_steps // 10) == 0:
                    elapsed = time.time() - start_time
                    progress = step / n_steps * 100
                    print(f"Progress: {progress:.1f}% (elapsed: {elapsed:.1f}s)")
        
        # Convert to arrays
        self.time_series_h1 = np.array(self.time_series_h1)
        self.time_series_h2 = np.array(self.time_series_h2)
        self.times = np.array(self.times)
        
        print(f"Simulation completed in {time.time() - start_time:.1f} seconds")
        print(f"Collected {len(self.times)} time points")


def compute_height_correlations(h1_series, h2_series, x_array):
    """
    Compute height-height correlation functions for auto and cross correlations.
    
    Returns:
    --------
    r_array : array
        Spatial separation distances
    G11 : array
        Auto-correlation for interface 1
    G22 : array
        Auto-correlation for interface 2
    G12 : array
        Cross-correlation between interfaces
    """
    L = len(x_array)
    dx = x_array[1] - x_array[0]
    
    # Create separation array
    r_array = np.arange(L//2) * dx
    
    # Average over time and compute correlations
    n_times = h1_series.shape[0]
    G11 = np.zeros(L//2)
    G22 = np.zeros(L//2)
    G12 = np.zeros(L//2)
    
    for t_idx in range(n_times):
        h1 = h1_series[t_idx]
        h2 = h2_series[t_idx]
        
        for r_idx, r in enumerate(r_array):
            if r_idx >= L//2:
                break
                
            # Compute height differences
            dh1 = h1 - np.roll(h1, r_idx)
            dh2 = h2 - np.roll(h2, r_idx)
            dh12 = h1 - np.roll(h2, r_idx)
            
            # Height-height correlations
            G11[r_idx] += np.mean(dh1**2)
            G22[r_idx] += np.mean(dh2**2)
            G12[r_idx] += np.mean(dh12**2)
    
    # Average over time
    G11 /= n_times
    G22 /= n_times
    G12 /= n_times
    
    return r_array, G11, G22, G12


def compute_cross_correlation(h1_series, h2_series):
    """
    Compute cross-correlation C₁₂(r,t) = ⟨h₁(x+r,t)h₂(x,t)⟩.
    """
    n_times, L = h1_series.shape
    max_r = L // 4  # Only compute up to L/4 for good statistics
    
    C12 = np.zeros((n_times, max_r))
    
    for t_idx in range(n_times):
        h1 = h1_series[t_idx] - np.mean(h1_series[t_idx])
        h2 = h2_series[t_idx] - np.mean(h2_series[t_idx])
        
        # Compute cross-correlation using FFT for efficiency
        fft_h1 = np.fft.fft(h1)
        fft_h2 = np.fft.fft(h2)
        cross_corr = np.fft.ifft(fft_h1 * np.conj(fft_h2))
        
        C12[t_idx, :] = np.real(cross_corr[:max_r]) / L
    
    return C12


def analyze_scaling_behavior(r_array, G_array, r_min=5, r_max=50):
    """
    Fit power law G(r) = A * r^(2α) to extract scaling exponent.
    
    Returns:
    --------
    alpha : float
        Scaling exponent
    alpha_err : float
        Standard error in α
    """
    # Select fitting range
    mask = (r_array >= r_min) & (r_array <= r_max) & (G_array > 0)
    r_fit = r_array[mask]
    G_fit = G_array[mask]
    
    if len(r_fit) < 5:
        return np.nan, np.nan
    
    # Fit in log space: log(G) = log(A) + 2α*log(r)
    log_r = np.log(r_fit)
    log_G = np.log(G_fit)
    
    try:
        popt, pcov = curve_fit(lambda x, a, b: a + b*x, log_r, log_G)
        two_alpha = popt[1]
        two_alpha_err = np.sqrt(pcov[1,1])
        
        return two_alpha/2, two_alpha_err/2
    except:
        return np.nan, np.nan


def main_simulation():
    """
    Main simulation function that runs coupled KPZ simulations and analyzes results.
    """
    print("=== Coupled KPZ Simulation Study ===")
    print("Investigating cross-coupling effects in two-interface systems")
    print()
    
    # Simulation parameters
    L = 256  # System size
    dx = 1.0  # Spatial resolution
    dt = 0.005  # Time step (smaller for stability)
    t_max = 200.0  # Total simulation time
    
    # Physical parameters (based on Cu-Ag system from theoretical analysis)
    nu1 = 1.0e-6  # Surface tension interface 1
    nu2 = 0.8e-6  # Surface tension interface 2 (asymmetric)
    lambda1 = 2.0e-4  # Nonlinear coefficient 1
    lambda2 = 1.5e-4  # Nonlinear coefficient 2
    D11 = 1.0e-6  # Noise strength 1
    D22 = 1.0e-6  # Noise strength 2
    
    # Study different coupling strengths
    coupling_strengths = [0.0, 0.1, 0.3, 0.5]  # λ₁₂/λ₁ ratios
    cross_noise_strengths = [0.0, 0.1]  # D₁₂/D₁₁ ratios
    
    results = {
        'parameters': {
            'L': L, 'dx': dx, 'dt': dt, 't_max': t_max,
            'nu1': nu1, 'nu2': nu2, 'lambda1': lambda1, 'lambda2': lambda2,
            'D11': D11, 'D22': D22
        },
        'simulations': []
    }
    
    simulation_count = 0
    total_simulations = len(coupling_strengths) * len(cross_noise_strengths)
    
    for coupling_ratio in coupling_strengths:
        for cross_noise_ratio in cross_noise_strengths:
            simulation_count += 1
            print(f"\n--- Simulation {simulation_count}/{total_simulations} ---")
            
            # Calculate actual coupling parameters
            lambda12 = coupling_ratio * lambda1
            lambda21 = coupling_ratio * lambda2  # Symmetric coupling
            D12 = cross_noise_ratio * np.sqrt(D11 * D22)
            
            print(f"Coupling ratio λ₁₂/λ₁ = {coupling_ratio}")
            print(f"Cross-noise ratio D₁₂/√(D₁₁D₂₂) = {cross_noise_ratio}")
            
            # Initialize and run simulation
            sim = CoupledKPZSimulator(L=L, dx=dx, dt=dt)
            sim.run_simulation(t_max, nu1, lambda1, lambda12, nu2, lambda2, lambda21, 
                             D11, D22, D12, save_interval=2.0)
            
            # Analyze results
            print("Analyzing correlation functions...")
            
            # Compute height-height correlations
            r_array, G11, G22, G12 = compute_height_correlations(
                sim.time_series_h1[-20:],  # Use last 20 time points for better statistics
                sim.time_series_h2[-20:], 
                sim.x
            )
            
            # Compute cross-correlations
            C12 = compute_cross_correlation(sim.time_series_h1[-10:], sim.time_series_h2[-10:])
            
            # Extract scaling exponents
            alpha1, alpha1_err = analyze_scaling_behavior(r_array, G11)
            alpha2, alpha2_err = analyze_scaling_behavior(r_array, G22)
            
            # Measure cross-correlation strength
            cross_corr_strength = np.max(np.abs(np.mean(C12, axis=0)))
            auto_corr_strength = np.sqrt(np.var(sim.time_series_h1[-1]) * np.var(sim.time_series_h2[-1]))
            cross_corr_ratio = cross_corr_strength / auto_corr_strength if auto_corr_strength > 0 else 0
            
            print(f"Results:")
            print(f"  α₁ = {alpha1:.3f} ± {alpha1_err:.3f}")
            print(f"  α₂ = {alpha2:.3f} ± {alpha2_err:.3f}")
            print(f"  Cross-correlation ratio = {cross_corr_ratio:.4f}")
            
            # Store results
            sim_result = {
                'coupling_ratio': coupling_ratio,
                'cross_noise_ratio': cross_noise_ratio,
                'lambda12': lambda12,
                'lambda21': lambda21,
                'D12': D12,
                'alpha1': alpha1,
                'alpha1_err': alpha1_err,
                'alpha2': alpha2,
                'alpha2_err': alpha2_err,
                'cross_corr_ratio': cross_corr_ratio,
                'r_array': r_array,
                'G11': G11,
                'G22': G22,
                'G12': G12,
                'C12': C12,
                'times': sim.times,
                'final_h1': sim.time_series_h1[-1],
                'final_h2': sim.time_series_h2[-1]
            }
            
            results['simulations'].append(sim_result)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"coupled_kpz_simulation_results_{timestamp}.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {filename}")
    
    return results


def plot_simulation_results(results):
    """
    Create publication-quality plots of simulation results.
    """
    plt.style.use('default')
    fig = plt.figure(figsize=(15, 12))
    
    # Extract data for plotting
    coupling_ratios = [sim['coupling_ratio'] for sim in results['simulations']]
    alpha1_values = [sim['alpha1'] for sim in results['simulations']]
    alpha2_values = [sim['alpha2'] for sim in results['simulations']]
    cross_corr_ratios = [sim['cross_corr_ratio'] for sim in results['simulations']]
    
    # Plot 1: Scaling exponents vs coupling strength
    plt.subplot(2, 3, 1)
    plt.errorbar(coupling_ratios, alpha1_values, 
                yerr=[sim['alpha1_err'] for sim in results['simulations']], 
                fmt='o-', label='α₁', capsize=3)
    plt.errorbar(coupling_ratios, alpha2_values,
                yerr=[sim['alpha2_err'] for sim in results['simulations']], 
                fmt='s-', label='α₂', capsize=3)
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='KPZ prediction')
    plt.xlabel('Coupling ratio λ₁₂/λ₁')
    plt.ylabel('Scaling exponent α')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Scaling Exponents vs Coupling')
    
    # Plot 2: Cross-correlation strength
    plt.subplot(2, 3, 2)
    plt.plot(coupling_ratios, cross_corr_ratios, 'ro-', markersize=6)
    plt.xlabel('Coupling ratio λ₁₂/λ₁')
    plt.ylabel('Cross-correlation ratio')
    plt.grid(True, alpha=0.3)
    plt.title('Cross-Correlation Strength')
    
    # Plot 3: Height-height correlation functions
    plt.subplot(2, 3, 3)
    for i, sim in enumerate(results['simulations'][:3]):  # Show first 3 simulations
        if sim['coupling_ratio'] in [0.0, 0.3, 0.5]:
            r = sim['r_array']
            G11 = sim['G11']
            mask = (r > 0) & (r < 50)
            plt.loglog(r[mask], G11[mask], '-', 
                      label=f"λ₁₂/λ₁ = {sim['coupling_ratio']:.1f}")
    
    # Theoretical KPZ scaling
    r_theory = np.logspace(0.5, 1.7, 50)
    G_theory = 0.1 * r_theory**1.0  # 2α = 1 for KPZ
    plt.loglog(r_theory, G_theory, 'k--', alpha=0.7, label='KPZ: r¹·⁰')
    
    plt.xlabel('Separation r')
    plt.ylabel('G₁₁(r)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Height-Height Correlations')
    
    # Plot 4: Interface profiles
    plt.subplot(2, 3, 4)
    uncoupled_sim = results['simulations'][0]  # λ₁₂ = 0
    coupled_sim = results['simulations'][-1]   # Strongest coupling
    
    x = np.arange(len(uncoupled_sim['final_h1'])) * results['parameters']['dx']
    plt.plot(x[:100], uncoupled_sim['final_h1'][:100], 'b-', alpha=0.7, label='h₁ (uncoupled)')
    plt.plot(x[:100], uncoupled_sim['final_h2'][:100], 'r-', alpha=0.7, label='h₂ (uncoupled)')
    plt.plot(x[:100], coupled_sim['final_h1'][:100] + 2, 'b-', linewidth=2, label='h₁ (coupled)')
    plt.plot(x[:100], coupled_sim['final_h2'][:100] + 2, 'r-', linewidth=2, label='h₂ (coupled)')
    
    plt.xlabel('Position x')
    plt.ylabel('Height h(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Interface Profiles')
    
    # Plot 5: Cross-correlation time evolution
    plt.subplot(2, 3, 5)
    for sim in results['simulations']:
        if sim['coupling_ratio'] in [0.1, 0.5]:
            C12_max = np.max(np.abs(sim['C12']), axis=1)
            times = np.linspace(0, results['parameters']['t_max'], len(C12_max))
            plt.plot(times, C12_max, '-', 
                    label=f"λ₁₂/λ₁ = {sim['coupling_ratio']:.1f}")
    
    plt.xlabel('Time t')
    plt.ylabel('Max |C₁₂(r,t)|')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Cross-Correlation Evolution')
    
    # Plot 6: Summary statistics
    plt.subplot(2, 3, 6)
    theoretical_correction = np.array(coupling_ratios) * 0.001  # Rough theoretical estimate
    observed_correction = np.array(alpha1_values) - 0.5
    
    plt.plot(coupling_ratios, observed_correction, 'ro-', label='Observed δα₁')
    plt.plot(coupling_ratios, theoretical_correction, 'k--', label='Theoretical estimate')
    plt.xlabel('Coupling ratio λ₁₂/λ₁')
    plt.ylabel('Scaling correction δα₁')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Theory vs Simulation')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"coupled_kpz_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Analysis plots saved to {filename}")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    # Run the main simulation study
    results = main_simulation()
    
    # Create analysis plots
    plot_simulation_results(results)
    
    # Print summary
    print("\n=== SIMULATION SUMMARY ===")
    print(f"Total simulations: {len(results['simulations'])}")
    print(f"System size: {results['parameters']['L']}")
    print(f"Simulation time: {results['parameters']['t_max']}")
    
    print("\nKey findings:")
    for i, sim in enumerate(results['simulations']):
        if sim['cross_noise_ratio'] == 0:  # Focus on pure coupling effects
            print(f"  λ₁₂/λ₁ = {sim['coupling_ratio']:.1f}: "
                  f"α₁ = {sim['alpha1']:.3f}, "
                  f"cross-corr = {sim['cross_corr_ratio']:.4f}")
    
    print("\nData files generated:")
    print("- coupled_kpz_simulation_results_*.pkl (numerical data)")
    print("- coupled_kpz_analysis_*.png (analysis plots)")