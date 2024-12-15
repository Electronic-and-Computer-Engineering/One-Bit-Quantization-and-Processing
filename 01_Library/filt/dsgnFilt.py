import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, tf2zpk
import scipy.signal as sigP
import cvxpy as cp

def dsgnFilt(desired_impulse_response, filter_order=None, design_method='LS', min_phase=False, plot_results=False, include_delay=False, delay=None):
    """
    Designs a FIR filter to approximate a desired impulse response using convex optimization.
    
    Parameters:
    desired_impulse_response : numpy array
        The desired impulse response to approximate.
    filter_order : int, optional
        The order (length-1) of the designed filter. If None, defaults to 80% of the length of desired_impulse_response.
    design_method : str, 'LS' or 'MinMax'
        The design method to use: 'LS' for least-squares, 'MinMax' for minimax.
    min_phase : bool
        Whether to convert the designed filter to minimum phase.
    plot_results : bool
        Whether to plot the results (magnitude, phase, zero-pole diagram).
    include_delay : bool
        Whether to include a delay in the desired frequency response.
    delay : float, optional
        The delay to include. If None, defaults to half the filter order.
        
    Returns:
    v_h_value : numpy array
        The designed filter coefficients.
    """
    # Frequency range
    v_w = np.arange(0, np.pi, step=0.01).reshape(-1, 1)
    
    # Load the desired impulse response (provided as input)
    v_h_ls_input = desired_impulse_response
    
    # Set filter order
    if filter_order is None:
        s_M = int(len(v_h_ls_input)*0.8)
    else:
        s_M = filter_order
    
    # Adjust delay parameter if needed
    if delay is None:
        s_Delay = s_M / 2
    else:
        s_Delay = delay
    
    # Compute desired frequency response
    v_H_ls_input = freqz(v_h_ls_input, 1, v_w.flatten())[1]
    if include_delay:
        v_Dd = v_H_ls_input.flatten() * np.exp(-1j * v_w.flatten() * s_Delay)
    else:
        v_Dd = v_H_ls_input.flatten()
    
    if plot_results:
        # Visualize the desired magnitude and phase
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(v_w / np.pi, 20 * np.log10(np.abs(v_Dd)))
        plt.title('Desired Magnitude (User-Provided IR)')
        plt.xlabel('Normalized Frequency (×π rad/sample)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(v_w / np.pi, np.unwrap(np.angle(v_Dd)))
        plt.title('Desired Phase (User-Provided IR)')
        plt.xlabel('Normalized Frequency (×π rad/sample)')
        plt.ylabel('Phase (radians)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    # DTFT Matrix for the FIR Filter
    v_n = np.arange(0, s_M + 1)
    v_wd = np.arange(0, np.pi, step=0.01).reshape(-1, 1)
    m_F = np.exp(-1j * v_wd * v_n)
    
    # Recompute v_Dd at v_wd frequencies
    v_H_ls_input = freqz(v_h_ls_input, 1, v_wd.flatten())[1]
    if include_delay:
        v_Dd = v_H_ls_input.flatten() * np.exp(-1j * v_wd.flatten() * s_Delay)
    else:
        v_Dd = v_H_ls_input.flatten()
    
    # Set up the design method
    if design_method == 'LS':
        # Least-squares design
        v_h = cp.Variable(s_M + 1)
        objective = cp.Minimize(cp.norm(m_F @ v_h - v_Dd, 2))
    elif design_method == 'MinMax':
        # Minimax design
        v_h = cp.Variable(s_M + 1)
        objective = cp.Minimize(cp.max(cp.abs(m_F @ v_h - v_Dd)))
    else:
        raise ValueError("design_method must be 'LS' or 'MinMax'")
    
    problem = cp.Problem(objective)
    problem.solve()
    
    v_h_value = v_h.value
    
    # Convert to minimum phase if requested
    if min_phase:
        v_h_value = sigP.minimum_phase(v_h_value, n_fft = len(v_h_value)*2, method='homomorphic', half = False)
    
    # Compute frequency response
    v_H = freqz(v_h_value, 1, v_wd.flatten())[1]
    
    if plot_results:
        # Plot the frequency response and zero-pole diagram
        plt.figure(figsize=(12, 10))
        
        # Magnitude Response
        plt.subplot(3, 1, 1)
        plt.plot(v_wd / np.pi, 20 * np.log10(np.abs(v_Dd)), '--', label='Desired')
        plt.plot(v_wd / np.pi, 20 * np.log10(np.abs(v_H)), label=f'{design_method} approximation')
        plt.title(f'Magnitude Response ({design_method} Design)')
        plt.xlabel('Normalized Frequency (×π rad/sample)')
        plt.ylabel('Magnitude (dB)')
        plt.legend()
        plt.grid(True)
        
        # Phase Response
        plt.subplot(3, 1, 2)
        plt.plot(v_wd / np.pi, np.unwrap(np.angle(v_Dd)), '--', label='Desired')
        plt.plot(v_wd / np.pi, np.unwrap(np.angle(v_H)), label=f'{design_method} approximation')
        plt.title(f'Phase Response ({design_method} Design)')
        plt.xlabel('Normalized Frequency (×π rad/sample)')
        plt.ylabel('Phase (radians)')
        plt.legend()
        plt.grid(True)
        
        # Zero-Pole Diagram
        plt.subplot(3, 1, 3)
        z, p, _ = tf2zpk(v_h_value, [1])
        plt.plot(np.real(z), np.imag(z), 'o', label='Zeros')
        plt.plot(np.real(p), np.imag(p), 'x', label='Poles')
        # Unit circle for reference
        theta = np.linspace(0, 2 * np.pi, 512)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle')
        plt.title(f'Zero-Pole Diagram ({design_method} Design)')
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    # Return the designed filter coefficients
    return v_h_value

