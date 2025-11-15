"""
Analysis utilities for coupled KPZ simulations
==============================================

This module provides statistical analysis tools for examining coupled
interface growth, including scaling analysis, cross-correlation measurements,
and regime identification.
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


def compute_interface_width(h):
    """
    Compute interface width W = sqrt(<[h - <h>]²>)
    
    Parameters:
    -----------
    h : ndarray
        Height field (can be 1D or 2D)
    
    Returns:
    --------
    float : Interface width
    """
    h_mean = np.mean(h)
    h_centered = h - h_mean
    return np.sqrt(np.mean(h_centered**2))


def scaling_analysis(widths, times, fit_range=None, verbose=True):
    """
    Perform power-law scaling analysis: W(t) ~ t^β
    
    Parameters:
    -----------
    widths : array_like
        Interface width time series
    times : array_like
        Corresponding time points
    fit_range : tuple, optional
        (start_idx, end_idx) for fitting. If None, automatically determined
    verbose : bool
        Print detailed results
    
    Returns:
    --------
    dict : Analysis results containing:
        - beta : Growth exponent
        - beta_error : Standard error
        - r_squared : Coefficient of determination
        - p_value : Statistical significance
        - classification : Comparison to KPZ theory
    """
    times = np.array(times)
    widths = np.array(widths)
    
    # Automatic fit range selection if not provided
    if fit_range is None:
        # Identify growth regime
        dw_dt = np.gradient(widths, times)
        initial_growth = np.mean(dw_dt[1:6])
        
        # Find where growth rate drops significantly
        transition_mask = dw_dt > initial_growth * 0.3
        if np.any(transition_mask):
            growth_end = np.where(transition_mask)[0][-1]
            growth_end = min(growth_end, len(times) - 10)
        else:
            growth_end = len(times) // 2
        
        fit_start = max(5, len(times) // 8)  # Avoid early transients
        fit_end = min(growth_end, 3 * len(times) // 4)
    else:
        fit_start, fit_end = fit_range
    
    # Extract fitting data
    t_fit = times[fit_start:fit_end]
    w_fit = widths[fit_start:fit_end]
    
    # Ensure positive values for log transform
    valid_mask = (t_fit > 0) & (w_fit > 0)
    t_fit = t_fit[valid_mask]
    w_fit = w_fit[valid_mask]
    
    if len(t_fit) < 5:
        if verbose:
            print("Error: Insufficient valid data points for scaling analysis")
        return None
    
    # Power law fit in log space: log(W) = log(A) + β*log(t)
    log_t = np.log(t_fit)
    log_w = np.log(w_fit)
    
    beta, log_A, r_value, p_value, std_err = stats.linregress(log_t, log_w)
    A = np.exp(log_A)
    r_squared = r_value**2
    
    # Theoretical comparison
    kpz_beta = 1/3
    deviation_kpz = abs(beta - kpz_beta)
    significance = deviation_kpz / std_err if std_err > 0 else 0
    
    # Classification
    if significance < 2:
        classification = "Consistent with KPZ (β=1/3)"
    elif significance < 3:
        classification = "Possible deviation from KPZ"
    else:
        classification = "Significant deviation from KPZ"
    
    if verbose:
        print(f"=== SCALING ANALYSIS ===")
        print(f"Power law fit: W(t) = {A:.4e} × t^{beta:.4f}")
        print(f"Growth exponent: β = {beta:.4f} ± {std_err:.4f}")
        print(f"R² = {r_squared:.4f}, p-value = {p_value:.2e}")
        print(f"Deviation from KPZ: {deviation_kpz:.4f} ({significance:.1f}σ)")
        print(f"Assessment: {classification}")
    
    return {
        'beta': beta,
        'beta_error': std_err,
        'A': A,
        'r_squared': r_squared,
        'p_value': p_value,
        'fit_times': t_fit,
        'fit_widths': w_fit,
        'deviation_from_kpz': deviation_kpz,
        'significance': significance,
        'classification': classification
    }


def cross_correlation(h1, h2):
    """
    Compute cross-correlation between two interfaces
    
    Parameters:
    -----------
    h1, h2 : ndarray
        Height fields
    
    Returns:
    --------
    float : Cross-correlation coefficient
    """
    h1_centered = h1 - np.mean(h1)
    h2_centered = h2 - np.mean(h2)
    
    numerator = np.mean(h1_centered * h2_centered)
    denominator = np.sqrt(np.mean(h1_centered**2) * np.mean(h2_centered**2))
    
    if denominator > 0:
        return numerator / denominator
    return 0.0


def analyze_correlation_evolution(h1_series, h2_series, times, verbose=True):
    """
    Analyze evolution of cross-correlations over time
    
    Parameters:
    -----------
    h1_series : list of ndarrays
        Time series of h1 snapshots
    h2_series : list of ndarrays
        Time series of h2 snapshots
    times : array_like
        Corresponding time points
    verbose : bool
        Print summary statistics
    
    Returns:
    --------
    dict : Correlation analysis results
    """
    cross_corrs = []
    
    for h1, h2 in zip(h1_series, h2_series):
        corr = cross_correlation(h1, h2)
        cross_corrs.append(corr)
    
    cross_corrs = np.array(cross_corrs)
    
    mean_corr = np.mean(cross_corrs)
    std_corr = np.std(cross_corrs)
    
    if verbose:
        print(f"=== CROSS-CORRELATION ANALYSIS ===")
        print(f"Mean: {mean_corr:.4f} ± {std_corr:.4f}")
        print(f"Range: [{np.min(cross_corrs):.4f}, {np.max(cross_corrs):.4f}]")
        
        if mean_corr > 0.05:
            print("Interpretation: Positive correlation (interfaces grow together)")
        elif mean_corr < -0.05:
            print("Interpretation: Negative correlation (interfaces grow oppositely)")
        else:
            print("Interpretation: Weak or no correlation")
    
    return {
        'cross_correlations': cross_corrs,
        'times': times,
        'mean': mean_corr,
        'std': std_corr,
        'min': np.min(cross_corrs),
        'max': np.max(cross_corrs)
    }


def identify_regime(widths, times, threshold=0.1):
    """
    Identify if system is in growth or saturated regime
    
    Parameters:
    -----------
    widths : array_like
        Interface width time series
    times : array_like
        Corresponding time points
    threshold : float
        Relative change threshold for saturation detection
    
    Returns:
    --------
    str : 'growth', 'saturated', or 'transitional'
    """
    widths = np.array(widths)
    
    # Check last 25% of evolution
    quarter_point = 3 * len(widths) // 4
    late_widths = widths[quarter_point:]
    
    if len(late_widths) < 2:
        return 'insufficient_data'
    
    # Calculate relative change
    mean_width = np.mean(late_widths)
    relative_variation = np.std(late_widths) / mean_width if mean_width > 0 else 0
    
    # Check trend
    late_times = times[quarter_point:]
    if len(late_times) >= 10:
        slope, _, _, _, _ = stats.linregress(late_times, late_widths)
        growth_rate = slope / mean_width if mean_width > 0 else 0
    else:
        growth_rate = (late_widths[-1] - late_widths[0]) / (times[-1] - times[quarter_point]) / mean_width
    
    if relative_variation < threshold and abs(growth_rate) < threshold:
        return 'saturated'
    elif growth_rate > threshold:
        return 'growth'
    else:
        return 'transitional'


def compute_scaling_exponents_multiple_sizes(simulations_by_size, verbose=True):
    """
    Compute scaling exponents from multiple system sizes for finite-size scaling analysis
    
    Parameters:
    -----------
    simulations_by_size : dict
        Dictionary mapping system sizes (L) to simulation results
        Each result should have 'times' and 'widths' arrays
    verbose : bool
        Print detailed analysis
    
    Returns:
    --------
    dict : Finite-size scaling results
    """
    results = {}
    
    for L, sim_data in simulations_by_size.items():
        times = sim_data['times']
        widths = sim_data['widths']
        
        scaling_result = scaling_analysis(widths, times, verbose=False)
        
        if scaling_result:
            results[L] = scaling_result
            
            if verbose:
                print(f"L = {L}: β = {scaling_result['beta']:.4f} ± {scaling_result['beta_error']:.4f}")
    
    # Check for finite-size effects
    if len(results) >= 2 and verbose:
        betas = [r['beta'] for r in results.values()]
        beta_std = np.std(betas)
        print(f"\nFinite-size scaling: β varies by {beta_std:.4f} across system sizes")
        
        if beta_std < 0.05:
            print("Assessment: Weak finite-size effects, exponents converged")
        else:
            print("Assessment: Significant finite-size effects present")
    
    return results
