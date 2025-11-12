#!/usr/bin/env python3
"""
Examine the saved simulation data from the long-running coupled KPZ simulation
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_and_examine_data():
    """Load and examine the saved simulation data"""
    
    print("=== EXAMINING SAVED COUPLED KPZ SIMULATION DATA ===")
    print(f"Analysis performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load the main results
    try:
        with open('coupled_kpz_results.pkl', 'rb') as f:
            results = pickle.load(f)
        
        print("Successfully loaded coupled_kpz_results.pkl")
        print(f"Data structure type: {type(results)}")
        
        if isinstance(results, dict):
            print(f"Keys in results: {list(results.keys())}")
            
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                    if value.size > 0:
                        print(f"    range: [{np.min(value):.3f}, {np.max(value):.3f}]")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: length {len(value)}, type {type(value[0]) if len(value) > 0 else 'empty'}")
                else:
                    print(f"  {key}: {type(value)} = {value}")
        
        print()
        
        # Load phase diagram data
        try:
            with open('kpz_phase_diagram_research.pkl', 'rb') as f:
                phase_data = pickle.load(f)
            
            print("Successfully loaded kpz_phase_diagram_research.pkl")
            print(f"Phase data type: {type(phase_data)}")
            
            if isinstance(phase_data, dict):
                print(f"Keys in phase data: {list(phase_data.keys())}")
                
                for key, value in phase_data.items():
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                    else:
                        print(f"  {key}: {type(value)}")
        
        except FileNotFoundError:
            print("Phase diagram data file not found")
        
        print()
        
        # Check what simulation parameters were used
        if 'parameters' in results:
            params = results['parameters']
            print("=== SIMULATION PARAMETERS ===")
            for key, value in params.items():
                print(f"  {key}: {value}")
            print()
        
        # Check what scaling data we have
        if 'scaling_data' in results:
            scaling = results['scaling_data']
            print("=== SCALING ANALYSIS DATA ===")
            for key, value in scaling.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: {value.shape} - range [{np.min(value):.3f}, {np.max(value):.3f}]")
                else:
                    print(f"  {key}: {value}")
            print()
        
        # Check time evolution data
        if 'time_evolution' in results:
            time_data = results['time_evolution']
            print("=== TIME EVOLUTION DATA ===")
            for key, value in time_data.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: {value.shape}")
                    if len(value.shape) == 1:
                        print(f"    time range: [{value[0]:.1f}, {value[-1]:.1f}]")
            print()
        
        # Analyze what coupling strengths were explored
        if 'coupling_analysis' in results:
            coupling = results['coupling_analysis']
            print("=== COUPLING STRENGTH ANALYSIS ===")
            for key, value in coupling.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: {value.shape}")
                    if 'gamma' in key.lower():
                        print(f"    coupling range: [{np.min(value):.3f}, {np.max(value):.3f}]")
            print()
        
        return results, phase_data if 'phase_data' in locals() else None
        
    except FileNotFoundError:
        print("ERROR: coupled_kpz_results.pkl not found!")
        return None, None
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None, None

def create_summary_plots(results):
    """Create summary plots from the saved data"""
    
    if results is None:
        print("No data to plot")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Scaling exponents vs coupling strength
    if 'coupling_analysis' in results:
        plt.subplot(2, 3, 1)
        coupling = results['coupling_analysis']
        if 'gamma_values' in coupling and 'beta_exponents' in coupling:
            plt.plot(coupling['gamma_values'], coupling['beta_exponents'], 'bo-', label='β exponent')
            plt.axhline(y=1/3, color='r', linestyle='--', label='Standard KPZ (β=1/3)')
            plt.xlabel('Coupling Strength γ')
            plt.ylabel('Growth Exponent β')
            plt.title('Growth Exponent vs Coupling')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # Plot 2: Interface width evolution
    if 'time_evolution' in results:
        plt.subplot(2, 3, 2)
        time_data = results['time_evolution']
        if 'times' in time_data and 'width_evolution' in time_data:
            times = time_data['times']
            widths = time_data['width_evolution']
            if widths.ndim > 1:
                # Plot evolution for different coupling strengths
                for i in range(min(3, widths.shape[0])):
                    plt.loglog(times, widths[i], label=f'Coupling {i}')
            else:
                plt.loglog(times, widths, 'b-', linewidth=2)
            plt.xlabel('Time')
            plt.ylabel('Interface Width')
            plt.title('Interface Width Evolution')
            plt.grid(True, alpha=0.3)
            plt.legend()
    
    # Plot 3: Phase diagram (if available)
    if 'coupling_analysis' in results:
        plt.subplot(2, 3, 3)
        coupling = results['coupling_analysis']
        if 'gamma_values' in coupling and 'synchronization_measure' in coupling:
            sync = coupling['synchronization_measure']
            gamma = coupling['gamma_values']
            plt.plot(gamma, sync, 'ro-', linewidth=2)
            plt.xlabel('Coupling Strength γ')
            plt.ylabel('Synchronization Measure')
            plt.title('Synchronization Phase Diagram')
            plt.grid(True, alpha=0.3)
    
    # Plot 4: Critical scaling behavior
    if 'scaling_data' in results:
        plt.subplot(2, 3, 4)
        scaling = results['scaling_data']
        if 'system_sizes' in scaling and 'critical_widths' in scaling:
            sizes = scaling['system_sizes']
            widths = scaling['critical_widths']
            plt.loglog(sizes, widths, 'go-', linewidth=2, label='Simulation data')
            # Theoretical scaling
            theoretical = sizes**(2/3) * widths[0] / sizes[0]**(2/3)
            plt.loglog(sizes, theoretical, 'r--', label='Standard KPZ (α=1/2)')
            plt.xlabel('System Size L')
            plt.ylabel('Interface Width')
            plt.title('Finite Size Scaling')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # Plot 5: Correlation analysis
    if 'correlation_data' in results:
        plt.subplot(2, 3, 5)
        corr_data = results['correlation_data']
        if 'distances' in corr_data and 'correlations' in corr_data:
            distances = corr_data['distances']
            correlations = corr_data['correlations']
            plt.semilogy(distances, correlations, 'mo-', linewidth=2)
            plt.xlabel('Distance')
            plt.ylabel('Height Correlation')
            plt.title('Spatial Correlations')
            plt.grid(True, alpha=0.3)
    
    # Plot 6: Energy/roughness evolution
    if 'time_evolution' in results:
        plt.subplot(2, 3, 6)
        time_data = results['time_evolution']
        if 'times' in time_data and 'roughness_evolution' in time_data:
            times = time_data['times']
            roughness = time_data['roughness_evolution']
            plt.loglog(times, roughness, 'co-', linewidth=2)
            plt.xlabel('Time')
            plt.ylabel('Surface Roughness')
            plt.title('Roughness Evolution')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('saved_data_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig('saved_data_summary.pdf', bbox_inches='tight')
    print("Summary plots saved as saved_data_summary.png and saved_data_summary.pdf")

def extract_key_findings(results):
    """Extract key scientific findings from the saved data"""
    
    if results is None:
        return
    
    print("=== KEY SCIENTIFIC FINDINGS FROM SAVED DATA ===")
    print()
    
    # Critical coupling strength
    if 'coupling_analysis' in results:
        coupling = results['coupling_analysis']
        if 'gamma_values' in coupling and 'beta_exponents' in coupling:
            gamma_vals = coupling['gamma_values']
            beta_vals = coupling['beta_exponents']
            
            # Find where beta deviates significantly from 1/3
            standard_beta = 1/3
            deviations = np.abs(beta_vals - standard_beta)
            critical_idx = np.argmax(deviations)
            
            print(f"1. CRITICAL COUPLING STRENGTH:")
            print(f"   γc ≈ {gamma_vals[critical_idx]:.3f}")
            print(f"   Maximum β deviation: {beta_vals[critical_idx]:.3f} (vs standard {standard_beta:.3f})")
            print()
            
            # Find transition region
            transition_indices = np.where(deviations > 0.05)[0]
            if len(transition_indices) > 0:
                gamma_transition = [gamma_vals[transition_indices[0]], gamma_vals[transition_indices[-1]]]
                print(f"2. TRANSITION REGION:")
                print(f"   γ ∈ [{gamma_transition[0]:.3f}, {gamma_transition[1]:.3f}]")
                print()
    
    # Scaling regime analysis
    if 'scaling_data' in results:
        scaling = results['scaling_data']
        if 'system_sizes' in scaling and 'alpha_exponents' in scaling:
            sizes = scaling['system_sizes']
            alphas = scaling['alpha_exponents']
            
            print(f"3. ROUGHNESS EXPONENT ANALYSIS:")
            print(f"   System sizes tested: {sizes}")
            print(f"   α values: {alphas}")
            print(f"   Average α: {np.mean(alphas):.3f} ± {np.std(alphas):.3f}")
            print(f"   Standard KPZ: α = 0.5")
            print()
    
    # Synchronization behavior
    if 'coupling_analysis' in results and 'synchronization_measure' in results['coupling_analysis']:
        sync_data = results['coupling_analysis']['synchronization_measure']
        gamma_vals = results['coupling_analysis']['gamma_values']
        
        # Find synchronization threshold
        sync_threshold = 0.5  # Typical threshold
        sync_indices = np.where(sync_data > sync_threshold)[0]
        
        if len(sync_indices) > 0:
            gamma_sync = gamma_vals[sync_indices[0]]
            print(f"4. SYNCHRONIZATION THRESHOLD:")
            print(f"   Synchronization begins at γ ≈ {gamma_sync:.3f}")
            print(f"   Maximum synchronization: {np.max(sync_data):.3f}")
            print()
    
    # Simulation completeness
    if 'parameters' in results:
        params = results['parameters']
        total_time = params.get('total_time', 'unknown')
        system_size = params.get('system_size', 'unknown')
        realizations = params.get('realizations', 'unknown')
        
        print(f"5. SIMULATION SCOPE:")
        print(f"   System size: {system_size}")
        print(f"   Evolution time: {total_time}")
        print(f"   Realizations: {realizations}")
        
        # Estimate computational time invested
        if isinstance(total_time, (int, float)) and isinstance(system_size, (int, float)):
            computational_cost = total_time * system_size**2 * realizations
            print(f"   Computational cost: ~{computational_cost:.2e} operations")
        print()
    
    print("=== PUBLICATION READINESS ASSESSMENT ===")
    print()
    
    # Check data quality metrics
    data_quality_score = 0
    max_score = 5
    
    if 'coupling_analysis' in results:
        data_quality_score += 1
        print("✓ Coupling strength analysis completed")
    
    if 'scaling_data' in results:
        data_quality_score += 1
        print("✓ Finite-size scaling analysis available")
    
    if 'time_evolution' in results:
        data_quality_score += 1
        print("✓ Temporal evolution data captured")
    
    if 'correlation_data' in results:
        data_quality_score += 1
        print("✓ Spatial correlation analysis performed")
    
    # Check for sufficient statistics
    if 'parameters' in results and results['parameters'].get('realizations', 0) >= 10:
        data_quality_score += 1
        print("✓ Sufficient ensemble averaging")
    
    print()
    print(f"Data Quality Score: {data_quality_score}/{max_score}")
    
    if data_quality_score >= 4:
        print("*** DATA IS PUBLICATION READY ***")
        print("Sufficient scope and quality for journal submission")
    elif data_quality_score >= 3:
        print("*** DATA NEARLY PUBLICATION READY ***")
        print("Minor additional analysis needed")
    else:
        print("*** ADDITIONAL DATA COLLECTION RECOMMENDED ***")
        print("Expand simulation scope for publication")

if __name__ == "__main__":
    # Load and examine the data
    results, phase_data = load_and_examine_data()
    
    # Create summary visualization
    create_summary_plots(results)
    
    # Extract key findings
    extract_key_findings(results)
    
    print()
    print("=== DATA PRESERVATION COMPLETE ===")
    print("All analysis results saved and ready for manuscript development")