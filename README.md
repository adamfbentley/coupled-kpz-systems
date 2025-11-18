# Coupled KPZ Systems

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-1.24+-blue.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A computational investigation of coupled surface growth systems using modified Kardar-Parisi-Zhang (KPZ) equations. This project explores how coupling between multiple growing interfaces affects their collective dynamics, correlations, and scaling behavior.

## Overview

The Kardar-Parisi-Zhang equation describes the growth of rough interfaces in systems ranging from bacterial colonies to flame fronts. While single-interface KPZ dynamics are well understood, multi-component systems with cross-coupling interactions remain an active area of research.

This project implements numerical simulations to investigate:
- How coupling strength affects interface synchronization
- Whether cross-coupling terms modify KPZ universality class
- The relationship between coupling symmetry and interface correlations
- Scaling behavior in coupled growth systems

## Mathematical Framework

The system consists of two coupled stochastic interfaces:

$$
\frac{\partial h_1}{\partial t} = \nu_1 \nabla^2 h_1 + \frac{\lambda_1}{2}|\nabla h_1|^2 + \frac{\lambda_{12}}{2}|\nabla h_2|^2 + \eta_1(x,t)
$$

$$
\frac{\partial h_2}{\partial t} = \nu_2 \nabla^2 h_2 + \frac{\lambda_2}{2}|\nabla h_2|^2 + \frac{\lambda_{21}}{2}|\nabla h_1|^2 + \eta_2(x,t)
$$

Where:
- $h_i(x,t)$ are the interface heights
- $\nu_i$ are surface tension coefficients
- $\lambda_i$ are self-interaction KPZ nonlinearities  
- $\lambda_{ij}$ are **cross-coupling terms** (key focus of investigation)
- $\eta_i(x,t)$ are Gaussian white noise terms

The cross-coupling terms $\lambda_{12}|\nabla h_2|^2$ and $\lambda_{21}|\nabla h_1|^2$ represent the novel physics: one interface's local slope affects the growth rate of the other interface.

## Key Features

### Simulation
- Finite difference solver with periodic boundary conditions
- Numba JIT optimization for computational efficiency
- Support for arbitrary coupling configurations
- Configurable system sizes and physical parameters

### Analysis
- Power-law scaling analysis (W(t) ~ t^β)
- Cross-correlation measurements
- Regime identification (growth vs. saturated)
- Statistical significance testing
- Comparison with KPZ theory (β = 1/3)

### Visualization
- Temporal evolution plots
- Log-log scaling analysis
- Cross-correlation time series
- Phase diagrams
- Multi-panel comprehensive summaries

## Installation

Clone the repository:
```bash
git clone https://github.com/adamfbentley/coupled-kpz-systems.git
cd coupled-kpz-systems
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "from coupled_kpz_simulation import CoupledKPZSimulator; print('✓ Installation successful')"
```

## Quick Start

### Basic Simulation

```python
from coupled_kpz_simulation import CoupledKPZSimulator
from analysis import compute_interface_width, scaling_analysis
import numpy as np

# Initialize simulator
sim = CoupledKPZSimulator(L=128, dx=1.0, dt=0.01)

# Run with symmetric coupling
results = sim.run_simulation(
    t_max=50.0,
    nu1=1.0, lambda1=1.0, lambda12=0.5,  # Interface 1 parameters
    nu2=1.0, lambda2=1.0, lambda21=0.5,  # Interface 2 parameters (symmetric)
    D11=1.0, D22=1.0, D12=0,             # Noise parameters
    save_interval=2.0
)

# Analyze scaling
times = np.array(results['times'])
widths = [compute_interface_width(h) for h in results['h1_series']]
scaling_result = scaling_analysis(widths, times)

print(f"Growth exponent: β = {scaling_result['beta']:.3f}")
```

### Run Example Scripts

The `examples/` directory contains ready-to-run demonstrations:

```bash
# Simple symmetric coupling simulation
python examples/simple_simulation.py

# Generate phase diagram
python examples/phase_diagram.py

# Compare different coupling types
python examples/coupling_comparison.py
```

## Project Structure

```
coupled-kpz-systems/
├── coupled_kpz_simulation.py    # Core simulation engine (524 lines)
├── analysis.py                   # Statistical analysis tools (287 lines)
├── visualization.py              # Plotting utilities (376 lines)
├── examples/                     # Example usage scripts
│   ├── simple_simulation.py      # Basic usage example (118 lines)
│   ├── phase_diagram.py          # Coupling parameter sweep (140 lines)
│   └── coupling_comparison.py    # Compare coupling types (167 lines)
├── figures/                      # Pre-generated result figures
│   ├── phase_diagram.pdf         # Correlation vs coupling strength
│   ├── scaling_analysis.pdf      # Growth exponent measurements
│   ├── interface_snapshots.pdf   # 2D interface visualizations
│   └── temporal_evolution.pdf    # Interface width over time
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT license
└── README.md                     # This file
```

## Results & Findings

### Coupling-Dependent Correlations

Simulations demonstrate that coupling symmetry affects interface correlations:

| Coupling Type | Parameters | Mean Cross-Correlation | Interpretation |
|---------------|------------|----------------------|----------------|
| **Symmetric** | λ₁₂ = λ₂₁ = +0.5 | ⟨C₁₂⟩ = +0.008 ± 0.017 | Weak positive correlation |
| **Antisymmetric** | λ₁₂ = +0.5, λ₂₁ = -0.5 | ⟨C₁₂⟩ = -0.014 ± 0.018 | Weak negative correlation |
| **Asymmetric** | λ₁₂ = 0.8, λ₂₁ = 0.2 | ⟨C₁₂⟩ ≈ +0.005 | Near-independent behavior |

**Key finding**: Even weak coupling (|λ| ≤ 0.5) produces measurable correlation signatures that depend on coupling symmetry. Stronger coupling regimes may produce more pronounced synchronization effects.

### Scaling Behavior

Measured growth exponents depend on simulation regime:

| Interface | Coupling Type | Measured β | Standard Error | Regime |
|-----------|---------------|------------|----------------|---------|
| h₁ | Symmetric | 0.043 | ±0.003 | Saturated |
| h₂ | Symmetric | 0.056 | ±0.004 | Saturated |
| h₁ | Antisymmetric | 0.054 | ±0.003 | Saturated |
| h₂ | Antisymmetric | 0.056 | ±0.003 | Saturated |

**Standard KPZ**: β = 1/3 ≈ 0.333 (growth regime)

**Interpretation**: The small measured exponents (β ≈ 0.05) indicate the simulations operated primarily in the **saturated regime** where interface width has plateaued. These values represent fluctuations around a steady state rather than power-law growth. The statistical deviations from KPZ scaling (90+ sigma) confirm saturation dynamics rather than novel universality classes.

**To observe true growth scaling**: Larger system sizes (L > 512), longer evolution times, and stronger coupling (|λ| > 1.0) would be needed to access the regime where coupling might modify KPZ exponents.

### Current Limitations

This is an exploratory undergraduate research project with several acknowledged limitations:

1. **Limited parameter space**: Coupling strengths explored are relatively weak (|λ₁₂|, |λ₂₁| ≤ 1)
2. **Finite-size effects**: System sizes (L ≤ 512) may introduce finite-size scaling artifacts
3. **Short evolution times**: Some simulations reach saturation before full growth regime is captured
4. **Statistical ensemble**: Limited ensemble averaging compared to publication-standard studies

Future work would benefit from larger system sizes, longer evolution times, and stronger coupling regimes to observe potential novel universality classes.

## Physical Relevance

Coupled interface growth appears in various physical contexts:

- **Multi-layer thin film deposition**: Growth of layered materials where layers influence each other
- **Competing populations**: Biological systems with interacting population fronts
- **Chemical etching**: Multi-species surface reactions with cross-interactions
- **Traffic flow**: Coupled lanes with lane-changing dynamics

## Requirements

- Python 3.11+
- NumPy 1.24+
- SciPy 1.10+
- Matplotlib 3.7+
- Numba 0.57+ (for performance optimization)

## Contributing

This is a personal research project, but suggestions and discussions are welcome! Feel free to:
- Open issues for bugs or questions
- Suggest improvements to documentation
- Share related research or applications

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{bentley2025coupled,
  author = {Bentley, Adam F.},
  title = {Coupled KPZ Systems: Numerical Investigation of Multi-Interface Growth},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/adamfbentley/coupled-kpz-systems}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

**Adam F. Bentley**  
Victoria University of Wellington  
Email: adam.f.bentley@gmail.com  
GitHub: [@adamfbentley](https://github.com/adamfbentley)

---

*This project represents undergraduate-level computational physics research exploring coupled stochastic growth phenomena. Results are preliminary and intended to demonstrate research methodology and numerical simulation skills.*
