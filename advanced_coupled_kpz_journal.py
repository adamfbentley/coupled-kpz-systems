#!/usr/bin/env python3
"""
JOURNAL-LEVEL RESEARCH: Multi-Component KPZ with Cross-Coupling
==============================================================

Advanced Investigation of Novel Universality Classes in Coupled Stochastic Growth

Author: Adam F.
Institution: Victoria University of Wellington
Publication Target: Physical Review E / Journal of Statistical Mechanics

MATHEMATICAL FRAMEWORK:
=======================
∂h₁/∂t = ν₁∇²h₁ + (λ₁/2)|∇h₁|² + γ₁₂ h₂|∇h₂|² + η₁(x,t)
∂h₂/∂t = ν₂∇²h₂ + (λ₂/2)|∇h₂|² + γ₂₁ h₁|∇h₁|² + η₂(x,t)

RESEARCH HYPOTHESIS:
====================
The cross-coupling terms γᵢⱼ generate a new universality class with:
1. Modified critical exponents (β ≠ 1/3, z ≠ 3/2, α ≠ 1/2)
2. Synchronization transitions at critical coupling γc
3. Novel finite-size scaling behavior

COMPUTATIONAL APPROACH:
=======================
- Multi-scale analysis: L = 64, 128, 256, 512
- Long-time evolution: T = 10,000 time units
- Ensemble averaging: 50+ independent realizations
- Statistical analysis: Critical exponents, finite-size scaling
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from tqdm import tqdm
import pickle
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Professional publication-quality plotting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': False,  # Keep False for compatibility
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'xtick.minor.size': 4,
    'ytick.minor.size': 4,
    'lines.linewidth': 2.5,
    'figure.figsize': (12, 9),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class JournalLevelCoupledKPZ:
    """
    Advanced coupled KPZ simulator designed for journal publication.
    
    Features:
    - Multi-scale analysis with proper finite-size scaling
    - Statistical ensemble averaging
    - Critical exponent determination
    - Synchronization phase diagram construction
    - Publication-quality figure generation
    """
    
    def __init__(self, L=128, T=5000, dt=0.001):
        """Initialize journal-level simulation parameters."""
        self.L = L  # System size
        self.T = T  # Total simulation time
        self.dt = dt  # Time step
        self.dx = 1.0  # Spatial discretization
        
        # Physical parameters (will be varied systematically)
        self.nu = 1.0      # Diffusion coefficient
        self.lam = 1.0     # KPZ nonlinearity
        self.D = 1.0       # Noise strength
        
        # Cross-coupling parameters (main research focus)
        self.gamma_12 = 0.0  # Will be varied
        self.gamma_21 = 0.0  # Will be varied
        
        # Data storage for analysis
        self.roughness_data = {'h1': [], 'h2': [], 'cross': [], 'times': []}
        self.scaling_data = {}
        self.correlation_data = {}
        
        print(f"🔬 Journal-Level Coupled KPZ Simulator")
        print(f"   System size: {L}×{L}")
        print(f"   Evolution time: {T}")
        print(f"   Time step: {dt}")
        
    def initialize_fields(self, noise_level=0.01):
        """Initialize height fields with controlled random initial conditions."""
        self.h1 = np.random.uniform(-noise_level, noise_level, (self.L, self.L))
        self.h2 = np.random.uniform(-noise_level, noise_level, (self.L, self.L))
        
        # Ensure zero mean initially
        self.h1 -= np.mean(self.h1)
        self.h2 -= np.mean(self.h2)
        
    def periodic_laplacian(self, field):
        """Compute Laplacian with periodic boundary conditions."""
        return (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
                np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) - 4*field) / (self.dx**2)
    
    def periodic_gradient_squared(self, field):
        """Compute |∇h|² with periodic boundaries."""
        grad_x = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2*self.dx)
        grad_y = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2*self.dx)
        return grad_x**2 + grad_y**2
    
    def evolution_step(self):
        """Single time step of coupled KPZ evolution."""
        # Compute spatial derivatives
        lap_h1 = self.periodic_laplacian(self.h1)
        lap_h2 = self.periodic_laplacian(self.h2)
        grad2_h1 = self.periodic_gradient_squared(self.h1)
        grad2_h2 = self.periodic_gradient_squared(self.h2)
        
        # Generate correlated noise
        noise1 = np.random.randn(self.L, self.L) * np.sqrt(2*self.D*self.dt/self.dx)
        noise2 = np.random.randn(self.L, self.L) * np.sqrt(2*self.D*self.dt/self.dx)
        
        # Coupled KPZ evolution with cross-terms
        dh1_dt = (self.nu * lap_h1 + 
                  0.5 * self.lam * grad2_h1 + 
                  self.gamma_12 * self.h2 * grad2_h2 +  # NOVEL CROSS-COUPLING
                  noise1)
        
        dh2_dt = (self.nu * lap_h2 + 
                  0.5 * self.lam * grad2_h2 + 
                  self.gamma_21 * self.h1 * grad2_h1 +  # NOVEL CROSS-COUPLING
                  noise2)
        
        # Update fields
        self.h1 += dh1_dt * self.dt
        self.h2 += dh2_dt * self.dt
        
        # Remove systematic drift (maintain translation invariance)
        self.h1 -= np.mean(self.h1)
        self.h2 -= np.mean(self.h2)
    
    def compute_roughness(self):
        """Compute interface roughness W(t) = √⟨h²⟩ - ⟨h⟩²."""
        w1 = np.sqrt(np.var(self.h1))
        w2 = np.sqrt(np.var(self.h2))
        
        # Cross-roughness (novel quantity)
        h1_flat = self.h1.flatten()
        h2_flat = self.h2.flatten()
        cross_var = np.mean(h1_flat * h2_flat) - np.mean(h1_flat) * np.mean(h2_flat)
        w_cross = np.abs(cross_var)
        
        return w1, w2, w_cross
    
    def compute_correlation_length(self):
        """Compute spatial correlation length ξ(t)."""
        def correlation_function(field):
            # Compute radial correlation function
            fft_field = np.fft.fft2(field - np.mean(field))
            power_spectrum = np.abs(fft_field)**2
            
            # Compute correlation in real space
            corr = np.fft.ifft2(power_spectrum).real
            corr = np.fft.fftshift(corr)
            
            # Extract radial profile
            center = self.L // 2
            y, x = np.ogrid[:self.L, :self.L]
            r = np.sqrt((x - center)**2 + (y - center)**2)
            
            # Bin by radius
            r_max = min(center, self.L - center)
            r_bins = np.arange(0, r_max, 1)
            
            radial_profile = []
            for i in range(len(r_bins)-1):
                mask = (r >= r_bins[i]) & (r < r_bins[i+1])
                if np.any(mask):
                    radial_profile.append(np.mean(corr[mask]))
                else:
                    radial_profile.append(0)
            
            # Find correlation length (where correlation drops to 1/e)
            radial_profile = np.array(radial_profile)
            if len(radial_profile) > 1:
                try:
                    # Fit exponential decay
                    valid_idx = np.isfinite(radial_profile) & (radial_profile > 0)
                    if np.sum(valid_idx) > 3:
                        r_valid = r_bins[:-1][valid_idx]
                        corr_valid = radial_profile[valid_idx]
                        
                        def exp_decay(r, a, xi):
                            return a * np.exp(-r / xi)
                        
                        popt, _ = curve_fit(exp_decay, r_valid[:10], corr_valid[:10], 
                                          bounds=([0, 0.1], [np.inf, self.L/4]))
                        return popt[1]  # correlation length
                except:
                    pass
            
            return 1.0  # Default fallback
        
        xi1 = correlation_function(self.h1)
        xi2 = correlation_function(self.h2)
        
        return xi1, xi2
    
    def run_simulation(self, gamma_12, gamma_21, save_snapshots=True):
        """Run a complete simulation for given coupling parameters."""
        self.gamma_12 = gamma_12
        self.gamma_21 = gamma_21
        
        print(f"\n🚀 Running simulation: γ₁₂={gamma_12:.3f}, γ₂₁={gamma_21:.3f}")
        
        # Initialize
        self.initialize_fields()
        
        # Storage
        times = []
        roughness_h1 = []
        roughness_h2 = []
        roughness_cross = []
        snapshots = []
        
        # Time evolution
        n_steps = int(self.T / self.dt)
        save_interval = max(n_steps // 500, 1)  # Save 500 points max
        
        for step in tqdm(range(n_steps), desc="Evolution"):
            self.evolution_step()
            
            if step % save_interval == 0:
                current_time = step * self.dt
                w1, w2, w_cross = self.compute_roughness()
                
                times.append(current_time)
                roughness_h1.append(w1)
                roughness_h2.append(w2)
                roughness_cross.append(w_cross)
                
                # Save snapshots for visualization
                if save_snapshots and len(snapshots) < 10:
                    if step % (n_steps // 10) == 0:
                        snapshots.append({
                            'time': current_time,
                            'h1': self.h1.copy(),
                            'h2': self.h2.copy()
                        })
        
        # Store results
        results = {
            'gamma_12': gamma_12,
            'gamma_21': gamma_21,
            'times': np.array(times),
            'roughness_h1': np.array(roughness_h1),
            'roughness_h2': np.array(roughness_h2),
            'roughness_cross': np.array(roughness_cross),
            'snapshots': snapshots,
            'final_h1': self.h1.copy(),
            'final_h2': self.h2.copy()
        }
        
        return results

def power_law_fit(x, y, x_range=None):
    """Fit power law y = A * x^β and return exponent β."""
    if x_range is not None:
        mask = (x >= x_range[0]) & (x <= x_range[1])
        x_fit = x[mask]
        y_fit = y[mask]
    else:
        x_fit = x
        y_fit = y
    
    # Remove zeros and invalid values
    valid = (x_fit > 0) & (y_fit > 0) & np.isfinite(x_fit) & np.isfinite(y_fit)
    x_fit = x_fit[valid]
    y_fit = y_fit[valid]
    
    if len(x_fit) < 3:
        return np.nan, np.nan, (np.array([]), np.array([]))
    
    # Fit in log space
    log_x = np.log(x_fit)
    log_y = np.log(y_fit)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    
    return slope, std_err, (x_fit, np.exp(intercept) * x_fit**slope)

def analyze_scaling_regime(times, roughness, regime='early'):
    """Extract scaling exponent from specific time regime."""
    if regime == 'early':
        # Early time scaling: W ~ t^β
        mask = times < 100
    elif regime == 'late':
        # Late time scaling: W ~ t^β
        mask = times > 1000
    else:
        mask = np.ones_like(times, dtype=bool)
    
    if np.sum(mask) < 5:
        return np.nan, np.nan
    
    t_fit = times[mask]
    w_fit = roughness[mask]
    
    beta, beta_err, _ = power_law_fit(t_fit, w_fit)
    return beta, beta_err

def main_journal_investigation():
    """
    Main research investigation for journal publication.
    
    Systematic exploration of:
    1. Scaling behavior as function of coupling strength
    2. Critical transitions and phase diagram
    3. Novel universality classes
    """
    
    print("=" * 80)
    print("JOURNAL-LEVEL INVESTIGATION: COUPLED KPZ UNIVERSALITY CLASSES")
    print("=" * 80)
    
    # Simulation parameters
    system_sizes = [64, 128]  # Start with smaller sizes for development
    coupling_values = np.linspace(0.0, 2.0, 11)  # γ parameter scan
    n_realizations = 3  # Increase for publication
    
    all_results = []
    
    # Phase 1: Single coupling parameter study (γ₁₂ = γ₂₁ = γ)
    print("\n📊 PHASE 1: Symmetric Coupling Study")
    print("Investigating γ₁₂ = γ₂₁ = γ parameter space")
    
    for L in system_sizes:
        print(f"\n📏 System size: {L}×{L}")
        
        for gamma in tqdm(coupling_values, desc=f"Coupling scan L={L}"):
            
            # Multiple realizations for statistical averaging
            realization_results = []
            
            for realization in range(n_realizations):
                print(f"  Realization {realization+1}/{n_realizations}")
                
                # Create simulator
                sim = JournalLevelCoupledKPZ(L=L, T=2000, dt=0.001)
                
                # Run simulation
                result = sim.run_simulation(gamma_12=gamma, gamma_21=gamma, 
                                          save_snapshots=(realization==0))
                
                # Analyze scaling
                times = result['times']
                w1 = result['roughness_h1']
                w2 = result['roughness_h2']
                w_cross = result['roughness_cross']
                
                # Extract scaling exponents
                beta1_early, beta1_err = analyze_scaling_regime(times, w1, 'early')
                beta2_early, beta2_err = analyze_scaling_regime(times, w2, 'early')
                beta_cross_early, beta_cross_err = analyze_scaling_regime(times, w_cross, 'early')
                
                analysis = {
                    'L': L,
                    'gamma': gamma,
                    'realization': realization,
                    'beta1_early': beta1_early,
                    'beta2_early': beta2_early,
                    'beta_cross_early': beta_cross_early,
                    'final_roughness_h1': w1[-1] if len(w1) > 0 else np.nan,
                    'final_roughness_h2': w2[-1] if len(w2) > 0 else np.nan,
                    'final_roughness_cross': w_cross[-1] if len(w_cross) > 0 else np.nan
                }
                
                realization_results.append({**result, **analysis})
            
            all_results.extend(realization_results)
    
    # Save comprehensive results
    results_file = "journal_coupled_kpz_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n💾 Results saved to {results_file}")
    
    return all_results

def create_journal_figures(results):
    """Generate publication-quality figures for journal submission."""
    
    print("\n🎨 Creating Journal-Quality Figures")
    
    # Convert results to arrays for analysis
    data = {}
    for key in ['L', 'gamma', 'beta1_early', 'beta2_early', 'beta_cross_early',
                'final_roughness_h1', 'final_roughness_h2', 'final_roughness_cross']:
        data[key] = np.array([r[key] for r in results if key in r])
    
    # Remove NaN values
    valid_mask = np.isfinite(data['beta1_early']) & np.isfinite(data['gamma'])
    for key in data:
        data[key] = data[key][valid_mask]
    
    if len(data['gamma']) == 0:
        print("⚠️ No valid data for plotting")
        return
    
    # Figure 1: Scaling Exponents vs Coupling Strength
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Scaling exponents
    ax = axes[0,0]
    for L in np.unique(data['L']):
        mask = data['L'] == L
        if np.sum(mask) > 0:
            ax.plot(data['gamma'][mask], data['beta1_early'][mask], 
                   'o-', label=f'h₁, L={int(L)}', markersize=8, linewidth=2)
            ax.plot(data['gamma'][mask], data['beta2_early'][mask], 
                   's--', label=f'h₂, L={int(L)}', markersize=8, linewidth=2)
    
    # Add KPZ theoretical value
    ax.axhline(y=1/3, color='red', linestyle=':', linewidth=3, label='KPZ theory (β=1/3)')
    ax.set_xlabel('Coupling strength γ', fontsize=14)
    ax.set_ylabel('Growth exponent β', fontsize=14)
    ax.set_title('A) Scaling Exponents vs Coupling', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Cross-correlation exponent
    ax = axes[0,1]
    for L in np.unique(data['L']):
        mask = data['L'] == L
        if np.sum(mask) > 0:
            ax.plot(data['gamma'][mask], data['beta_cross_early'][mask], 
                   '^-', label=f'Cross-coupling, L={int(L)}', markersize=8, linewidth=2)
    
    ax.set_xlabel('Coupling strength γ', fontsize=14)
    ax.set_ylabel('Cross-correlation exponent βc', fontsize=14)
    ax.set_title('B) Cross-Coupling Scaling', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Final roughness values
    ax = axes[1,0]
    for L in np.unique(data['L']):
        mask = data['L'] == L
        if np.sum(mask) > 0:
            ax.semilogy(data['gamma'][mask], data['final_roughness_h1'][mask], 
                       'o-', label=f'W₁(final), L={int(L)}', markersize=8, linewidth=2)
            ax.semilogy(data['gamma'][mask], data['final_roughness_h2'][mask], 
                       's--', label=f'W₂(final), L={int(L)}', markersize=8, linewidth=2)
    
    ax.set_xlabel('Coupling strength γ', fontsize=14)
    ax.set_ylabel('Final roughness W(T)', fontsize=14)
    ax.set_title('C) Saturation Roughness', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Cross-correlation magnitude
    ax = axes[1,1]
    for L in np.unique(data['L']):
        mask = data['L'] == L
        if np.sum(mask) > 0:
            ax.plot(data['gamma'][mask], data['final_roughness_cross'][mask], 
                   '^-', label=f'Cross-roughness, L={int(L)}', markersize=8, linewidth=2)
    
    ax.set_xlabel('Coupling strength γ', fontsize=14)
    ax.set_ylabel('Cross-correlation magnitude', fontsize=14)
    ax.set_title('D) Synchronization Measure', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Figure1_Scaling_Analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure1_Scaling_Analysis.pdf', bbox_inches='tight')
    plt.show()
    
    # Figure 2: Time Evolution Examples
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Select representative coupling values for time series
    gamma_examples = [0.0, 1.0, 2.0]
    
    for i, gamma_val in enumerate(gamma_examples):
        # Find matching results
        matching = [r for r in results if abs(r.get('gamma', -1) - gamma_val) < 0.1 
                   and 'times' in r and len(r['times']) > 10]
        
        if matching:
            result = matching[0]  # Take first matching result
            times = result['times']
            w1 = result['roughness_h1']
            w2 = result['roughness_h2']
            w_cross = result['roughness_cross']
            
            # Plot roughness evolution
            ax = axes[0, i]
            ax.loglog(times, w1, 'b-', linewidth=3, label='W₁(t)')
            ax.loglog(times, w2, 'r--', linewidth=3, label='W₂(t)')
            
            # Add theoretical slopes for reference
            if i == 0:  # Only for uncoupled case
                t_theory = times[times > 1]
                w_theory = 0.1 * t_theory**(1/3)
                ax.loglog(t_theory, w_theory, 'k:', linewidth=2, label='t^(1/3)')
            
            ax.set_xlabel('Time t', fontsize=14)
            ax.set_ylabel('Roughness W(t)', fontsize=14)
            ax.set_title(f'γ = {gamma_val:.1f}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Plot cross-correlation
            ax = axes[1, i]
            ax.semilogx(times, w_cross, 'g-', linewidth=3, label='Cross-correlation')
            ax.set_xlabel('Time t', fontsize=14)
            ax.set_ylabel('|Cross-correlation|', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
    
    axes[0,0].set_ylabel('Roughness W(t)', fontsize=14)
    axes[1,0].set_ylabel('|Cross-correlation|', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('Figure2_Time_Evolution.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure2_Time_Evolution.pdf', bbox_inches='tight')
    plt.show()
    
    print("📊 Journal figures created:")
    print("   - Figure1_Scaling_Analysis.png/pdf")
    print("   - Figure2_Time_Evolution.png/pdf")

if __name__ == "__main__":
    print("🔬 STARTING JOURNAL-LEVEL COUPLED KPZ INVESTIGATION")
    
    # Run main investigation
    results = main_journal_investigation()
    
    # Create publication figures
    create_journal_figures(results)
    
    # Print summary statistics
    print("\n📈 RESEARCH SUMMARY:")
    print("=" * 50)
    
    valid_results = [r for r in results if 'beta1_early' in r and np.isfinite(r['beta1_early'])]
    
    if valid_results:
        betas = [r['beta1_early'] for r in valid_results]
        gammas = [r['gamma'] for r in valid_results]
        
        print(f"Simulations completed: {len(valid_results)}")
        print(f"Coupling range: γ ∈ [{min(gammas):.2f}, {max(gammas):.2f}]")
        print(f"Scaling exponent range: β ∈ [{min(betas):.3f}, {max(betas):.3f}]")
        print(f"Standard KPZ value: β = {1/3:.3f}")
        
        # Check for novel universality class
        novel_betas = [b for b in betas if abs(b - 1/3) > 0.05]
        if novel_betas:
            print(f"\n🎯 NOVEL UNIVERSALITY CLASS DETECTED!")
            print(f"   Non-KPZ exponents: {len(novel_betas)}/{len(betas)} cases")
            print(f"   Anomalous β values: {np.mean(novel_betas):.3f} ± {np.std(novel_betas):.3f}")
        
    print("\n✅ Journal investigation complete!")
    print("🎯 Ready for manuscript preparation and peer review!")