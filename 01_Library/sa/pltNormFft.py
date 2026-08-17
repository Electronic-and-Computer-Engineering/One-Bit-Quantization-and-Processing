import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import sys
sys.path.append('../../01_Library')
# individual packages
import sa

def pltNormFft(vbM, sNbins, sFs, ax=None, figure_size_cm=(12.5, 4), fft_color='black', fft_linewidth=1.5):
    """
    Plots or updates the normalized FFT spectrum of a signal.

    Parameters:
    -----------
    vbM : np.ndarray
        Input signal (1D array).
    sNbins : int
        Number of bins (FFT length).
    sFs : float
        Sampling frequency in Hz.
    ax : matplotlib.axes.Axes or None
        Axis to plot into. If None, a new figure and axis are created.
    figure_size_cm : tuple
        Figure size in centimeters if a new figure is created.
    fft_color : str
        Color of the FFT line.
    fft_linewidth : float
        Width of the FFT line.
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axis object used for plotting.
    """

    # Define x-tick positions and labels
    xticks = np.linspace(0, np.pi, 11)
    xtick_labels = [
        r'$0$', r'$\frac{\pi}{10}$', r'$\frac{2\pi}{10}$', r'$\frac{3\pi}{10}$',
        r'$\frac{4\pi}{10}$', r'$\frac{5\pi}{10}$', r'$\frac{6\pi}{10}$',
        r'$\frac{7\pi}{10}$', r'$\frac{8\pi}{10}$', r'$\frac{9\pi}{10}$', r'$\pi$'
    ]

    # Frequency axis and normalized frequency
    sT = 1 / sFs
    vFreq = np.fft.fftfreq(sNbins, sT)
    vNormFrequ = (vFreq / sFs) * 2 * np.pi

    # FFT magnitude
    vBfft = np.fft.fft(vbM, sNbins)
    max_val = np.max(np.abs(vBfft))
    if max_val == 0:
        raise ValueError("FFT magnitude is zero everywhere — normalization invalid.")
    vBfftMag = 20 * sa.safelog10(np.abs(vBfft) / max_val)

    # If no axis is provided, create one
    if ax is None:
        width_inch = figure_size_cm[0] / 2.54
        height_inch = figure_size_cm[1] / 2.54
        fig, ax = plt.subplots(figsize=(width_inch, height_inch))

    else:
        # If an axis is provided, clear it
        ax.clear()

    # Plot the FFT
    ax.plot(vNormFrequ[:sNbins // 2], vBfftMag[:sNbins // 2],
            color=fft_color, linewidth=fft_linewidth)

    ax.set_ylabel(r'Magnitude (dB)', fontsize=13)
    ax.set_xlim([0, np.pi])
    ax.set_ylim([-60, 5])
    ax.minorticks_on()
    ax.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=13)

    # Optional: update plot if in interactive mode
    ax.figure.canvas.draw_idle()

    return ax
