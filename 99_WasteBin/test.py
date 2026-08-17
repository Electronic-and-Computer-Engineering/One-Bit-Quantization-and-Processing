import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Filter specifications
numtaps = 47   # Number of coefficients (filter order + 1)
fs = 1000.0    # Sampling frequency (Hz)
# Bands: [0, pass_start, pass_stop, nyquist]
bands = [0, 125, 200, 300, 350, fs/2]
# Desired amplitudes at band edges
desired = [0, 0, 1, 1, 0, 0]

# Calculate FIR coefficients
# firls normalizes frequency, so divide bands by (fs/2)
h = signal.firls(numtaps, bands, desired, fs=fs)

# Frequency response of the designed filter
w, H = signal.freqz(h, worN=8000)
freq = w * fs / (2 * np.pi)

# Plotting
plt.figure()
plt.plot(freq, 20 * np.log10(np.abs(H)))
plt.title('FIR Filter Frequency Response (Least Squares)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.show()