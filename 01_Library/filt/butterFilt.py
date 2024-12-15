# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:19:55 2024

@author: mayerflo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, dimpulse

def butterFiltImp(order, digital_cutoff, sNFilt, plot_impulse=False):
    """
    Computes the impulse response of a Butterworth filter.

    Parameters:
    - order: int, the order of the Butterworth filter
    - digital_cutoff: float, the normalized cutoff frequency (between 0 and 1)
    - n_impulse: int, the number of samples for the impulse response (default is 50)
    - plot_impulse: bool, flag to plot the impulse response (default is False)

    Returns:
    - impulse_response: array, the impulse response of the Butterworth filter
    - b, a: filter coefficients (numerator and denominator)
    """
    # Design the Butterworth filter
    b, a = butter(order, digital_cutoff, btype='low', analog=False)

    # Compute the impulse response
    _, h = dimpulse((b, a, 1), n=sNFilt)
    impulse_response = np.squeeze(h)

    # Plot the impulse response if the flag is set to True
    if plot_impulse:
        plt.figure(figsize=(8, 4))
        plt.stem(np.arange(len(impulse_response)), impulse_response)
        plt.title(f'Impulse Response of Butterworth Filter (Order {order}, Cutoff {digital_cutoff})')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    return impulse_response, b, a
