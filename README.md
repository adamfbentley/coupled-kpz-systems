# Coupled KPZ Systems

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-1.24+-blue.svg)](https://numpy.org/)

An exploratory project investigating coupled surface growth systems using the Kardar-Parisi-Zhang (KPZ) equation. This work examines how coupling between multiple interfaces affects their growth dynamics and correlations.

## Overview

This project explores what happens when two growing surfaces influence each other's evolution. Using numerical simulations, I'm investigating different coupling scenarios (symmetric, antisymmetric, and asymmetric) to see how they affect surface roughness and cross-correlations.

## Mathematical Background

The project studies two coupled interfaces described by modified KPZ equations:

$$
\frac{\partial h_i}{\partial t} = \nu \nabla^2 h_i + \frac{\lambda}{2}|\nabla h_i|^2 + \sum_{j \neq i} \gamma_{ij} h_j|\nabla h_j|^2 + \eta_i(x,t)
$$

The coupling term $\gamma_{ij} h_j|\nabla h_j|^2$ allows one interface to influence the growth of the other.

## Features

- Numerical simulation of coupled KPZ systems
- Analysis of scaling behavior and roughness exponents
- Visualization tools for phase diagrams and temporal evolution
- Support for symmetric, antisymmetric, and asymmetric coupling

## Requirements

```
numpy
matplotlib
scipy
```

## Usage

Run the main simulation:
```python
python coupled_kpz_simulation.py
```

## Project Structure

```
coupled-kpz-systems/
├── coupled_kpz_simulation.py    # Main simulation code
├── figures/                      # Generated plots
│   ├── phase_diagram.pdf
│   ├── scaling_analysis.pdf
│   ├── interface_snapshots.pdf
│   └── temporal_evolution.pdf
└── README.md
```

## Results

The simulations explore how different coupling parameters affect:
The simulations explore how different coupling parameters affect:
- Interface roughness over time
- Cross-correlations between surfaces
- Scaling exponents

Early results suggest that coupling strength significantly influences the growth dynamics, though more analysis is needed to draw firm conclusions.

## Notes

This is an early-stage exploration project. The code is functional but results are preliminary and require further validation.

## License

MIT
