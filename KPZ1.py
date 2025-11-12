# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 22:55:09 2025

@author: adamf
"""

# My KPZ simulation code for thin-film growth - worked on this over a few nights

import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio

# Parameters I picked after testing - for a thin film on a substrate
N = 512  # grid points, picked 512 for FFT
L = 512.0  # substrate length in nm
dx = L / N  # spacing
dt = 0.01  # small time step to keep it stable
nu = 1.0  # smoothing factor
lambda_kpz = 1.0  # nonlinear term, set to 0 for EW test
D = 1.0  # noise strength
v = 0.1  # average growth for film
T = 1000.0  # run time
steps = int(T / dt)
save_interval = 1000  # save every so often

# Stability check - learned this after a crash
if dt > dx**2 / (2 * nu):
    print("Warning: dt too big, might crash!")

# Start with flat surfaces
h_kpz = np.zeros(N)  # KPZ case
h_ew = np.zeros(N)  # EW case for comparison

# Noise setup - tried different ways, this worked best
noise_scale = np.sqrt(2 * D / (dx * dt))  # for KPZ noise
# Could use simple random: np.random.rand(N) - 0.5, but this fits better

# Store results
times = []
widths_kpz = []
widths_ew = []

# My derivative functions - used roll to avoid slow loops
def laplacian(h, dx):  # second derivative
    return (np.roll(h, 1) - 2 * h + np.roll(h, -1)) / dx**2

def grad_squared(h, dx):  # (dh/dx)^2
    grad = (np.roll(h, -1) - np.roll(h, 1)) / (2 * dx)
    return grad**2

# Main loop - update heights step by step
for step in range(steps):
    # KPZ update
    diff_kpz = nu * laplacian(h_kpz, dx)
    nonlin_kpz = (lambda_kpz / 2) * grad_squared(h_kpz, dx) if lambda_kpz != 0 else 0
    noise = noise_scale * np.random.randn(N) + v  # Gaussian noise + growth
    h_kpz += (diff_kpz + nonlin_kpz) * dt + noise * dt
    
    # EW update - no nonlinear term
    diff_ew = nu * laplacian(h_ew, dx)
    noise_ew = noise_scale * np.random.randn(N) + v  # same noise pattern
    h_ew += diff_ew * dt + noise_ew * dt
    
    # Save widths
    if step % save_interval == 0:
        current_time = step * dt
        mean_kpz = np.mean(h_kpz)
        mean_ew = np.mean(h_ew)
        width_kpz = np.sqrt(np.mean((h_kpz - mean_kpz)**2))
        width_ew = np.sqrt(np.mean((h_ew - mean_ew)**2))
        times.append(current_time)
        widths_kpz.append(width_kpz)
        widths_ew.append(width_ew)
        print(f"Time {current_time:.2f}: KPZ width = {width_kpz:.4f}, EW width = {width_ew:.4f}")

# Plot surfaces
plt.figure(figsize=(8, 4))
x = np.linspace(0, L, N)
plt.plot(x, h_kpz, label='KPZ thin film')
plt.plot(x, h_ew, label='EW case')
plt.title('Thin film surface at t=1000')
plt.xlabel('Position (nm)')
plt.ylabel('Height (nm)')
plt.legend()
plt.savefig('surface_plot.png')
plt.close()

# Width scaling
plt.figure(figsize=(8, 4))
plt.loglog(times, widths_kpz, 'o-', label='KPZ')
plt.loglog(times, widths_ew, 's-', label='EW')
plt.loglog(times, 0.5 * np.array(times)**(1/3), '--', label='t^1/3')
plt.loglog(times, 0.3 * np.array(times)**(1/4), '--', label='t^1/4')
plt.title('Roughness over time')
plt.xlabel('Time')
plt.ylabel('Width')
plt.legend()
plt.savefig('width_plot.png')
plt.close()

# PSD for KPZ
h_kpz -= np.mean(h_kpz)
fft_kpz = np.fft.fft(h_kpz)
psd_kpz = np.abs(fft_kpz)**2 / N
freq = np.fft.fftfreq(N, d=dx)[:N//2]
psd_kpz = psd_kpz[:N//2]
plt.figure(figsize=(8, 4))
plt.loglog(freq[freq > 0], psd_kpz[freq > 0], label='KPZ PSD')
plt.loglog(freq[freq > 0], 1e3 * freq[freq > 0]**(-2), '--', label='q^-2')
plt.title('Power spectrum')
plt.xlabel('Freq (1/nm)')
plt.ylabel('PSD')
plt.legend()
plt.savefig('psd_plot.png')
plt.close()

# Notes to myself: Might need to adjust dt if it gets weird, or try more points