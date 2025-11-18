# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Installation verification command in README
- Quantitative results table with actual correlation values
- Detailed scaling behavior table with measured exponents
- Figure descriptions in project structure
- Line counts for all modules

### Changed
- Updated project structure to show correct line counts (524 lines for main simulator)
- Enhanced Results section with statistical measurements
- Improved interpretation of saturated regime dynamics

## [1.0.0] - 2025-11-15

### Added
- Core simulation engine with Numba optimization (524 lines)
- Analysis module with scaling analysis and cross-correlation tools (287 lines)
- Visualization module with publication-quality plotting (376 lines)
- Three example scripts demonstrating usage
- Comprehensive README with mathematical framework
- Requirements.txt for reproducible environment
- MIT license
- Pre-generated result figures (phase diagram, scaling analysis, interface snapshots, temporal evolution)

### Features
- Finite difference solver for coupled KPZ equations
- Periodic boundary conditions
- Configurable coupling strengths and system parameters
- Power-law scaling analysis (W(t) ~ t^β)
- Cross-correlation measurements between interfaces
- Regime identification (growth vs saturated)
- Statistical significance testing
- Multiple visualization options

### Documentation
- Mathematical framework with LaTeX equations
- Installation instructions
- Quick start guide with code examples
- Project structure overview
- Physical relevance section
- Citation format (BibTeX)
- Honest assessment of current limitations

## [0.1.0] - 2025-11-12

### Initial Release
- Basic coupled KPZ simulation implementation
- Initial README and documentation
- First commit to GitHub repository
