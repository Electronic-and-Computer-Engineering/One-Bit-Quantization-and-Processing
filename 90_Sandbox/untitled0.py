# Test script to compare two implementations of DFT matrix-based transformation

import numpy as np
import matplotlib.pyplot as plt

# Define the original implementation
def dftTrans_original(vx, sK):
    """
    Computes the DFT of a signal using an extended DFT transformation matrix of size K x N.
    
    Parameters:
    vx (array-like): The input signal of length N for which to compute the DFT.
    sK (int): The number of frequency bins (rows of the DFT matrix, K).
    
    Returns:
    numpy.ndarray: The DFT of the input signal with extended frequency resolution (length K).
    """
    sN = len(vx)  # Length of the input signal
    vn = np.arange(sN)  # [0, 1, 2, ..., N-1] (time indices)
    vk = np.arange(sK).reshape((sK, 1))  # Column vector [0, 1, 2, ..., K-1]
    mTrans = np.exp(-2j * np.pi * vk * vn / sK)  # DFT matrix
    vX = mTrans @ vx  # Compute the extended DFT
    return vX

# Define the corrected implementation
def dftTrans_corrected(vx, sK):
    """
    Computes the DFT of a signal using an extended DFT transformation matrix of size K x N.
    
    Parameters:
    vx (array-like): The input signal of length N for which to compute the DFT.
    sK (int): The number of frequency bins (rows of the DFT matrix, K).
    
    Returns:
    numpy.ndarray: The DFT of the input signal with extended frequency resolution (length K).
    """
    sN = len(vx)  # Length of the input signal
    vn = np.arange(sN)  # [0, 1, 2, ..., N-1] (time indices)
    vk = np.arange(sK).reshape((sK, 1))  # Column vector [0, 1, 2, ..., K-1]
    mTrans = np.exp(-2j * np.pi * vk * vn / sK)  # DFT matrix
    vX = mTrans @ vx  # Compute the extended DFT
    return vX

# Test parameters
N = 64  # Signal length
K = 12*N  # Desired number of frequency bins (K = 2N)
vx = np.sin(2 * np.pi * np.linspace(0, 1, N)) + np.sin(5 * np.pi * np.linspace(0, 1, N)) + np.sin(7 * np.pi * np.linspace(0, 1, N)) + np.sin(11 * np.pi * np.linspace(0, 1, N))  # Example sinusoidal signal

# Compute the DFT using both implementations
vX_original = dftTrans_original(vx, K)
vX_corrected = dftTrans_corrected(vx, K)

# Comparison metrics
difference = np.abs(vX_original - vX_corrected)
max_difference = np.max(difference)

# Plotting results
plt.figure(figsize=(12, 6))

# Original function
plt.subplot(1, 2, 1)
plt.plot(np.linspace(0, 1, K), np.abs(vX_original), label="Original")
plt.title("Original Implementation")
plt.xlabel("Frequency (normalized)")
plt.ylabel("Magnitude")
plt.grid()

# Corrected function
plt.subplot(1, 2, 2)
plt.plot(np.linspace(0, 1, K), np.abs(vX_corrected), label="Corrected", color='orange')
plt.title("Corrected Implementation")
plt.xlabel("Frequency (normalized)")
plt.ylabel("Magnitude")
plt.grid()

plt.tight_layout()
plt.show()

# Print comparison result
print(f"Maximum difference between implementations: {max_difference:.5e}")
