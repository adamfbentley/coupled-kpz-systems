# My code for simulating thin-film growth with KPZ - put this together from notes
# Took some time to get the noise and plots right, based on PHYS345 project guide
import numpy as np
import matplotlib.pyplot as plt

def solve_stochastic_growth_1d(
    L=512.0, T=1000.0, dt=0.005, dx=1.0, v=1.0, lam=1.0, beta=0.0, D=1.0, growth_rate=0.1
):
    """
    Solves the 1D KPZ equation for thin-film growth using finite differences.

    Args:
        L (float): Length of substrate (nm).
        T (float): Total deposition time (s).
        dt (float): Time step (s).
        dx (float): Spatial step (nm).
        v (float): Diffusion coefficient.
        lam (float): Nonlinear growth coefficient.
        beta (float): Higher-order diffusion coefficient.
        D (float): Noise strength.
        growth_rate (float): Average growth rate (nm/s).

    Returns:
        tuple: Final height profile and height evolution.
    """
    nx = int(L / dx)
    nt = int(T / dt)

    # Initialize flat surface with tiny random noise
    h = 0.01 * np.random.randn(nx)
    h_evolution = np.zeros((nt, nx))

    # Stability check - from project notes, dt should be small
    if dt > (dx**2) / (2 * v):
        print("Warning: dt too large, might be unstable!")

    # Noise scaling to match white noise correlation
    noise_amplitude = np.sqrt(2 * D / dx) * np.sqrt(dt)

    for i in range(nt):
        # Periodic boundaries with roll
        h_plus = np.roll(h, -1)
        h_minus = np.roll(h, 1)
        h_plus2 = np.roll(h, -2)
        h_minus2 = np.roll(h, 2)

        # Finite differences - followed the guide closely
        grad_h = (h_plus - h_minus) / (2 * dx)
        laplacian_h = (h_plus - 2 * h + h_minus) / (dx**2)
        grad_h_sq = grad_h**2

        # Higher-order term if beta > 0
        if beta != 0:
            laplacian_of_laplacian_h = (h_plus2 - 4 * h_plus + 6 * h - 4 * h_minus + h_minus2) / (dx**4)
        else:
            laplacian_of_laplacian_h = 0

        # Stochastic noise with mean growth
        eta = noise_amplitude * np.random.randn(nx) + growth_rate

        # Update height - Euler method from the notes
        h += dt * (v * laplacian_h + (lam / 2) * grad_h_sq - beta * laplacian_of_laplacian_h + eta)
        h_evolution[i, :] = h

    return h, h_evolution

if __name__ == "__main__":
    # Parameters for thin-film growth with KPZ
    h_final, h_evo = solve_stochastic_growth_1d(
        L=512.0, T=1000.0, dt=0.005, dx=1.0, v=1.0, lam=1.0, beta=0.0, D=1.0, growth_rate=0.1
    )

    # EW case for comparison
    h_final_ew, h_evo_ew = solve_stochastic_growth_1d(
        L=512.0, T=1000.0, dt=0.005, dx=1.0, v=1.0, lam=0.0, beta=0.0, D=1.0, growth_rate=0.1
    )

    # Plotting - following project guide for surface and PSD
    plt.figure(figsize=(12, 6))

    # Surface profile - final state as suggested
    plt.subplot(1, 2, 1)
    x = np.arange(len(h_final)) * dx
    plt.plot(x, h_final, label='KPZ Thin Film')
    plt.plot(x, h_final_ew, label='EW Case')
    plt.title('Final Thin-Film Surface at t=1000s')
    plt.xlabel('Position (nm)')
    plt.ylabel('Height (nm)')
    plt.legend()

    # PSD analysis - replacing evolution plot with spectral info
    h_final -= np.mean(h_final)  # Remove mean for PSD
    fft_h = np.fft.fft(h_final)
    psd = np.abs(fft_h)**2 / len(h_final)
    freq = np.fft.fftfreq(len(h_final), d=dx)[:len(h_final)//2]
    psd = psd[:len(h_final)//2]

    plt.subplot(1, 2, 2)
    plt.loglog(freq[freq > 0], psd[freq > 0], label='KPZ PSD')
    plt.loglog(freq[freq > 0], 1e3 * freq[freq > 0]**(-2), '--', label='~q^-2')
    plt.title('Power Spectral Density')
    plt.xlabel('Spatial Frequency (1/nm)')
    plt.ylabel('PSD')
    plt.legend()

    plt.tight_layout()
    plt.savefig('thin_film_plots.png')  # Save for presentation
    plt.show()  # Show in Spyder
    # plt.close()  # Optional, keep open for inspection

    # Notes to myself: Check stability with smaller dt if needed, PSD slope should be ~2