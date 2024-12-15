import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sigP
from matplotlib.ticker import FuncFormatter, MultipleLocator

import filt

def dB20(array):
    with np.errstate(divide='ignore'):
        return 20 * np.log10(array)

def anAndPlotK(vHFilt, sFs, sFpb, sFsb, sFpb2, sFsb2, strType):
    (vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.anFiltKaiser(vHFilt, sFs, strType, sFpb, sFsb, sFpb2, sFsb2)
    filt.plotFrequResp(vw, vH, sFs, sFpb, sFsb, sHpbMin, sHpbMax, sHsbMax, sFpb2, sFsb2, strType)

def plotFrequResp(vw, vH, sFs, sFpb1, sFsb1, sHpbMin, sHpbMax, sHsbMax, sFpb2=None, sFsb2=None, strType='lowpass'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # Create a figure with 2 subplots
    fig.suptitle("Frequency and Phase Response")
    
    # First subplot: Frequency Response
    ax1.set_title("Frequency Response")
    ax1.grid(True)
    
    # Normalize frequencies to the range 0 to π
    sFpb1_normalized = sFpb1 / (sFs / 2) * np.pi
    sFsb1_normalized = sFsb1 / (sFs / 2) * np.pi
    
    if strType == 'lowpass':
        ax1.plot(vw, dB20(np.abs(vH)), "r", label='Frequency Response')
        ax1.plot([0, sFpb1_normalized], [dB20(sHpbMax), dB20(sHpbMax)], "b--", linewidth=1.0, label='Passband Max')
        ax1.plot([0, sFpb1_normalized], [dB20(sHpbMin), dB20(sHpbMin)], "b--", linewidth=1.0, label='Passband Min')
        ax1.plot([sFsb1_normalized, np.pi], [dB20(sHsbMax), dB20(sHsbMax)], "b--", linewidth=1.0, label='Stopband Max')
    
    elif strType == 'highpass':
        ax1.plot(vw, dB20(np.abs(vH)), "r", label='Frequency Response')
        ax1.plot([sFpb1_normalized, np.pi], [dB20(sHpbMax), dB20(sHpbMax)], "b--", linewidth=1.0, label='Passband Max')
        ax1.plot([sFpb1_normalized, np.pi], [dB20(sHpbMin), dB20(sHpbMin)], "b--", linewidth=1.0, label='Passband Min')
        ax1.plot([0, sFsb1_normalized], [dB20(sHsbMax), dB20(sHsbMax)], "b--", linewidth=1.0, label='Stopband Max')
    
    elif strType == 'bandpass':
        if sFpb2 is None or sFsb2 is None:
            raise ValueError("sFpb2 and sFsb2 must be provided for bandpass filter.")
        
        sFpb2_normalized = sFpb2 / (sFs / 2) * np.pi
        sFsb2_normalized = sFsb2 / (sFs / 2) * np.pi

        ax1.plot(vw, dB20(np.abs(vH)), "r", label='Frequency Response')
        ax1.plot([sFpb1_normalized, sFpb2_normalized], [dB20(sHpbMax), dB20(sHpbMax)], "b--", linewidth=1.0, label='Passband Max')
        ax1.plot([sFpb1_normalized, sFpb2_normalized], [dB20(sHpbMin), dB20(sHpbMin)], "b--", linewidth=1.0, label='Passband Min')
        ax1.plot([0, sFsb1_normalized], [dB20(sHsbMax), dB20(sHsbMax)], "b--", linewidth=1.0, label='Stopband Max Lower')
        ax1.plot([sFsb2_normalized, np.pi], [dB20(sHsbMax), dB20(sHsbMax)], "b--", linewidth=1.0, label='Stopband Max Upper')
    
    ax1.set_xlim(0, np.pi)
    ax1.set_ylim(-120, 3)
    
    ax3 = ax1.twiny()
    ax3.set_xlim(0, sFs / 2)
    
    update_ticks(ax1, ax3, sFs)
    
    ax1.set_xlabel('Normalized Frequency (π radians/sample)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.legend()
    
    ax3.set_xlabel('Frequency (Hz)')
    
    # Second subplot: Phase Response
    phase_response = np.unwrap(np.angle(vH))
    ax2.plot(vw, phase_response, 'g', label='Phase Response (Linear)')
    ax2.set_title('Phase Response')
    ax2.set_xlabel('Normalized Frequency (π radians/sample)')
    ax2.set_ylabel('Phase (radians)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust layout to accommodate the overall title
    plt.show()
    
def update_ticks(ax, ax2, sFs):
    def pi_formatter(x, pos):
        frac = x / np.pi
        if frac == 0:
            return '0'
        elif frac == 1:
            return r'$\pi$'
        elif frac == 0.5:
            return r'$\frac{\pi}{2}$'
        elif frac == 0.25:
            return r'$\frac{\pi}{4}$'
        elif frac % 1 == 0:
            return r'${}\pi$'.format(int(frac))
        else:
            return r'${:.2g}\pi$'.format(frac)
        
    def hz_formatter(x, pos, sFs):
            return f'{x:.2f} Hz'
        
    ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    xlim = ax.get_xlim()
    if (xlim[1] - xlim[0]) < (np.pi / 4):  # Adjust finer scale when zoomed in
        ax.xaxis.set_major_locator(MultipleLocator(np.pi / 16))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(np.pi / 8))
    
    hz_ticks = np.linspace(0, sFs / 2, 9)
    ax2.set_xlim(0, sFs / 2)
    ax2.set_xticks(hz_ticks)
    ax2.set_xticklabels([hz_formatter(tick, pos, sFs) for pos, tick in enumerate(hz_ticks)])
    ax.figure.canvas.draw()