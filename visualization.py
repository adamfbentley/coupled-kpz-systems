"""
Visualization utilities for coupled KPZ simulations
===================================================

Publication-quality plotting functions for coupled interface growth analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Set professional plotting style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'lines.linewidth': 2,
    'figure.figsize': (10, 8),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def plot_interface_evolution(times, widths_h1, widths_h2, title="Interface Evolution", 
                             save_path=None):
    """
    Plot temporal evolution of interface widths
    
    Parameters:
    -----------
    times : array_like
        Time points
    widths_h1, widths_h2 : array_like
        Interface widths for both surfaces
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(times, widths_h1, 'b-', label='Interface 1', linewidth=2, alpha=0.8)
    ax.plot(times, widths_h2, 'r-', label='Interface 2', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Time', fontsize=13)
    ax.set_ylabel('Interface Width W(t)', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_scaling_analysis(times, widths, scaling_result, interface_name="h1",
                          save_path=None):
    """
    Plot scaling analysis with power-law fit
    
    Parameters:
    -----------
    times : array_like
        Time points
    widths : array_like
        Interface widths
    scaling_result : dict
        Results from analysis.scaling_analysis()
    interface_name : str
        Name for plot label
    save_path : str, optional
        Path to save figure
    """
    if scaling_result is None:
        print("Error: No scaling result provided")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale plot
    ax1.plot(times, widths, 'ko', markersize=4, alpha=0.6, label='Data')
    
    # Plot fit
    t_fit = scaling_result['fit_times']
    w_fit_theory = scaling_result['A'] * t_fit**scaling_result['beta']
    ax1.plot(t_fit, w_fit_theory, 'r-', linewidth=2.5, 
             label=f"Fit: W ∝ t^{scaling_result['beta']:.3f}")
    
    # KPZ theory line
    if len(times) > 10:
        t_theory = np.linspace(times[5], times[-1], 100)
        w_kpz = widths[5] * (t_theory / times[5])**(1/3)
        ax1.plot(t_theory, w_kpz, 'g--', linewidth=2, alpha=0.6, label='KPZ (β=1/3)')
    
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('Interface Width W(t)', fontsize=12)
    ax1.set_title(f'Scaling Analysis: {interface_name}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Log-log plot
    valid_mask = (times > 0) & (widths > 0)
    ax2.loglog(times[valid_mask], widths[valid_mask], 'ko', markersize=4, alpha=0.6, label='Data')
    ax2.loglog(t_fit, w_fit_theory, 'r-', linewidth=2.5, 
               label=f"β = {scaling_result['beta']:.3f} ± {scaling_result['beta_error']:.3f}")
    
    # Reference slopes
    t_ref = np.array([t_fit[0], t_fit[-1]])
    w_ref_kpz = widths[5] * (t_ref / times[5])**(1/3)
    ax2.loglog(t_ref, w_ref_kpz, 'g--', linewidth=2, alpha=0.6, label='KPZ slope (1/3)')
    
    ax2.set_xlabel('Time t (log scale)', fontsize=12)
    ax2.set_ylabel('Interface Width W(t) (log scale)', fontsize=12)
    ax2.set_title(f'Log-Log Analysis (R² = {scaling_result["r_squared"]:.4f})', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_cross_correlation_evolution(times, cross_correlations, coupling_type="symmetric",
                                     save_path=None):
    """
    Plot evolution of cross-correlations over time
    
    Parameters:
    -----------
    times : array_like
        Time points
    cross_correlations : array_like
        Cross-correlation values
    coupling_type : str
        Type of coupling for title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(times, cross_correlations, 'b-', linewidth=2, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add mean line
    mean_corr = np.mean(cross_correlations)
    ax.axhline(y=mean_corr, color='r', linestyle='--', linewidth=2, 
               label=f'Mean = {mean_corr:.4f}')
    
    ax.set_xlabel('Time', fontsize=13)
    ax.set_ylabel('Cross-Correlation C₁₂(t)', fontsize=13)
    ax.set_title(f'Cross-Correlation Evolution ({coupling_type} coupling)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_interface_snapshots(h1, h2, time=None, coupling_params=None, save_path=None):
    """
    Plot 2D interface snapshots side by side
    
    Parameters:
    -----------
    h1, h2 : ndarray
        Interface height fields (2D arrays)
    time : float, optional
        Time point for title
    coupling_params : dict, optional
        Coupling parameters for title
    save_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot interface 1
    im1 = ax1.imshow(h1, cmap='viridis', aspect='auto', origin='lower')
    ax1.set_title('Interface 1', fontsize=13, fontweight='bold')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Height h₁')
    
    # Plot interface 2
    im2 = ax2.imshow(h2, cmap='plasma', aspect='auto', origin='lower')
    ax2.set_title('Interface 2', fontsize=13, fontweight='bold')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='Height h₂')
    
    # Add overall title
    title = 'Interface Snapshots'
    if time is not None:
        title += f' at t = {time:.1f}'
    if coupling_params is not None:
        title += f" (γ₁₂={coupling_params.get('gamma_12', 0):.2f}, γ₂₁={coupling_params.get('gamma_21', 0):.2f})"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_phase_diagram(coupling_strengths, correlation_data, save_path=None):
    """
    Plot phase diagram of correlations vs coupling parameters
    
    Parameters:
    -----------
    coupling_strengths : array_like
        Array of coupling parameter values
    correlation_data : array_like
        Corresponding correlation values
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(coupling_strengths, correlation_data, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Coupling Strength γ', fontsize=13)
    ax.set_ylabel('Mean Cross-Correlation', fontsize=13)
    ax.set_title('Phase Diagram: Correlation vs Coupling', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add shaded regions
    positive_mask = correlation_data > 0.05
    negative_mask = correlation_data < -0.05
    
    if np.any(positive_mask):
        ax.axhspan(0.05, ax.get_ylim()[1], alpha=0.1, color='blue', label='Positive correlation')
    if np.any(negative_mask):
        ax.axhspan(ax.get_ylim()[0], -0.05, alpha=0.1, color='red', label='Negative correlation')
    
    ax.legend(fontsize=11)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comprehensive_analysis(times, widths_h1, widths_h2, cross_corr, 
                                scaling_h1, scaling_h2, coupling_info, save_path=None):
    """
    Create comprehensive multi-panel figure with all key results
    
    Parameters:
    -----------
    times : array_like
        Time points
    widths_h1, widths_h2 : array_like
        Interface widths
    cross_corr : array_like
        Cross-correlation time series
    scaling_h1, scaling_h2 : dict
        Scaling analysis results
    coupling_info : dict
        Coupling parameters
    save_path : str, optional
        Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Interface evolution
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, widths_h1, 'b-', label='Interface 1', linewidth=2)
    ax1.plot(times, widths_h2, 'r-', label='Interface 2', linewidth=2)
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Interface Width W(t)', fontsize=11)
    ax1.set_title('Temporal Evolution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scaling analysis h1 (log-log)
    ax2 = fig.add_subplot(gs[1, 0])
    valid_mask = (times > 0) & (widths_h1 > 0)
    ax2.loglog(times[valid_mask], widths_h1[valid_mask], 'ko', markersize=3, alpha=0.5)
    if scaling_h1:
        t_fit = scaling_h1['fit_times']
        w_fit = scaling_h1['A'] * t_fit**scaling_h1['beta']
        ax2.loglog(t_fit, w_fit, 'r-', linewidth=2, 
                   label=f"β={scaling_h1['beta']:.3f}±{scaling_h1['beta_error']:.3f}")
    ax2.set_xlabel('Time (log)', fontsize=10)
    ax2.set_ylabel('W(t) (log)', fontsize=10)
    ax2.set_title('Scaling: Interface 1', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    
    # 3. Scaling analysis h2 (log-log)
    ax3 = fig.add_subplot(gs[1, 1])
    valid_mask = (times > 0) & (widths_h2 > 0)
    ax3.loglog(times[valid_mask], widths_h2[valid_mask], 'ko', markersize=3, alpha=0.5)
    if scaling_h2:
        t_fit = scaling_h2['fit_times']
        w_fit = scaling_h2['A'] * t_fit**scaling_h2['beta']
        ax3.loglog(t_fit, w_fit, 'r-', linewidth=2,
                   label=f"β={scaling_h2['beta']:.3f}±{scaling_h2['beta_error']:.3f}")
    ax3.set_xlabel('Time (log)', fontsize=10)
    ax3.set_ylabel('W(t) (log)', fontsize=10)
    ax3.set_title('Scaling: Interface 2', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    
    # 4. Cross-correlation evolution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(times, cross_corr, 'g-', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    mean_corr = np.mean(cross_corr)
    ax4.axhline(y=mean_corr, color='r', linestyle='--', linewidth=2,
                label=f'Mean={mean_corr:.4f}')
    ax4.set_xlabel('Time', fontsize=10)
    ax4.set_ylabel('Cross-Correlation', fontsize=10)
    ax4.set_title('Cross-Correlation Evolution', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Statistical summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    summary_text = "COUPLING PARAMETERS:\n"
    summary_text += f"γ₁₂ = {coupling_info.get('gamma_12', 0):.3f}\n"
    summary_text += f"γ₂₁ = {coupling_info.get('gamma_21', 0):.3f}\n\n"
    
    summary_text += "SCALING RESULTS:\n"
    if scaling_h1:
        summary_text += f"Interface 1: β = {scaling_h1['beta']:.3f} ± {scaling_h1['beta_error']:.3f}\n"
        summary_text += f"             {scaling_h1['classification']}\n"
    if scaling_h2:
        summary_text += f"Interface 2: β = {scaling_h2['beta']:.3f} ± {scaling_h2['beta_error']:.3f}\n"
        summary_text += f"             {scaling_h2['classification']}\n\n"
    
    summary_text += "CORRELATION:\n"
    summary_text += f"Mean: {mean_corr:.4f} ± {np.std(cross_corr):.4f}\n"
    if mean_corr > 0.05:
        summary_text += "Positive correlation\n"
    elif mean_corr < -0.05:
        summary_text += "Negative correlation\n"
    else:
        summary_text += "Weak/no correlation\n"
    
    ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle('Comprehensive Coupled KPZ Analysis', fontsize=15, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()
