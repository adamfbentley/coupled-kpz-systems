#!/usr/bin/env python3
"""
Check what data actually exists in the phase diagram files
"""

import pickle
import numpy as np

# Check the phase diagram research data
print("=== PHASE DIAGRAM RESEARCH DATA ===")
with open('kpz_phase_diagram_research.pkl', 'rb') as f:
    phase_data = pickle.load(f)

print("Keys:", list(phase_data.keys()))

if 'phase_diagram' in phase_data:
    print("Phase diagram keys:", list(phase_data['phase_diagram'].keys()))
    
if 'individual_results' in phase_data:
    print(f"Number of individual simulations: {len(phase_data['individual_results'])}")
    print("\nFirst 10 simulation parameters:")
    for i in range(min(10, len(phase_data['individual_results']))):
        result = phase_data['individual_results'][i]
        print(f"  {i}: γ12={result['gamma_12']:.2f}, γ21={result['gamma_21']:.2f}")
    
    # Check parameter ranges
    gamma_12_values = [r['gamma_12'] for r in phase_data['individual_results']]
    gamma_21_values = [r['gamma_21'] for r in phase_data['individual_results']]
    
    print(f"\nγ12 range: [{min(gamma_12_values):.2f}, {max(gamma_12_values):.2f}]")
    print(f"γ21 range: [{min(gamma_21_values):.2f}, {max(gamma_21_values):.2f}]")
    print(f"Unique γ12 values: {len(set(gamma_12_values))}")
    print(f"Unique γ21 values: {len(set(gamma_21_values))}")

if 'research_metadata' in phase_data:
    print("\nResearch metadata:")
    for key, value in phase_data['research_metadata'].items():
        print(f"  {key}: {value}")

print("\n" + "="*50)

# Check the coupled KPZ results
print("=== COUPLED KPZ RESULTS DATA ===")
with open('coupled_kpz_results.pkl', 'rb') as f:
    coupled_data = pickle.load(f)

print("Keys:", list(coupled_data.keys()))

if 'parameters' in coupled_data:
    print("Parameter sets:")
    for key, params in coupled_data['parameters'].items():
        print(f"  {key}: γ12={params['gamma_12']}, γ21={params['gamma_21']}")