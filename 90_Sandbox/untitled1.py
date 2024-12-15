import numpy as np
from gurobipy import Model, GRB, quicksum

def compute_reduced_coeffs(N):
    """
    Compute reduced cosine and sine coefficient matrices for FFT.
    Only stores the first half of coefficients.
    
    Parameters:
    - N: Signal length.
    
    Returns:
    - cos_coeffs: Reduced cosine coefficients (N/2 x N).
    - sin_coeffs: Reduced sine coefficients (N/2 x N).
    """
    frequencies = np.arange(N // 2 + 1)  # Only compute for first half
    cos_coeffs = np.cos(2 * np.pi * np.outer(frequencies, np.arange(N)) / N)
    sin_coeffs = np.sin(2 * np.pi * np.outer(frequencies, np.arange(N)) / N)
    return cos_coeffs, sin_coeffs

def gurobi_fft_reduced(signal, N):
    """
    Computes FFT using reduced cosine and sine coefficient matrices in Gurobi.
    
    Parameters:
    - signal: Input binary signal (list or NumPy array).
    - N: Length of the signal (zero-padded if necessary).
    
    Returns:
    - real_part: Real part of FFT components (list).
    - imag_part: Imaginary part of FFT components (list).
    """
    # Zero-pad the signal if necessary
    signal = np.pad(signal, (0, max(0, N - len(signal))), mode='constant')

    # Compute reduced cosine and sine coefficients
    cos_coeffs, sin_coeffs = compute_reduced_coeffs(N)

    # Create Gurobi model
    model = Model("FFT_Reduced")
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output

    # Create variables for FFT components
    real_part = model.addVars(N // 2 + 1, lb=-GRB.INFINITY, name="Re")
    imag_part = model.addVars(N // 2 + 1, lb=-GRB.INFINITY, name="Im")

    # Define constraints for the first half of FFT components
    for k in range(N // 2 + 1):
        model.addConstr(
            real_part[k] == quicksum(signal[n] * cos_coeffs[k, n] for n in range(N)),
            name=f"Re_{k}",
        )
        model.addConstr(
            imag_part[k] == quicksum(signal[n] * sin_coeffs[k, n] for n in range(N)),
            name=f"Im_{k}",
        )

    # Add dummy objective
    model.setObjective(0, GRB.MINIMIZE)

    # Solve the model
    model.optimize()

    # Extract results for the first half
    real_values = [real_part[k].x for k in range(N // 2 + 1)]
    imag_values = [imag_part[k].x for k in range(N // 2 + 1)]

    # Use symmetry to compute the full FFT
    full_real = real_values + real_values[1:-1][::-1]
    full_imag = imag_values + [-x for x in imag_values[1:-1][::-1]]

    return np.array(full_real), np.array(full_imag)

# Example usage
binary_signal = [1, -1, 1, -1]
desired_length = 4

# Compute FFT using reduced coefficients
gurobi_real, gurobi_imag = gurobi_fft_reduced(binary_signal, desired_length)

# Compute FFT using NumPy
np_signal = np.pad(binary_signal, (0, max(0, desired_length - len(binary_signal))), mode='constant')
np_fft = np.fft.fft(np_signal)
np_real = np.real(np_fft)
np_imag = np.imag(np_fft)

# Compare results
print("NumPy FFT Real:", np_real)
print("NumPy FFT Imag:", np_imag)
print("Gurobi FFT Real:", gurobi_real)
print("Gurobi FFT Imag:", gurobi_imag)
