# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:13:57 2024

@author: mayerflo
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sigP
from matplotlib.ticker import FuncFormatter, MultipleLocator

def anFiltEqu(vHFilt, sFs, sFpb, sFsb, strType):
    
    (vw, vH) = sigP.freqz(vHFilt)

    # Calculate passband ripple
    # Find pass band ripple
    sHpbMin = min(np.abs(vH[0:int(sFpb/sFs*2 * len(vH))]))
    sHpbMax = max(np.abs(vH[0:int(sFpb/sFs*2 * len(vH))]))
    sRpb = 1 - (sHpbMax - sHpbMin)#20 * np.log10(sHpbMax / sHpbMin)  # Ripple in dB

    # Calculate stopband attenuations
    # Find stop band attenuation
    sHsbMax = max(np.abs(vH[int(sFsb/sFs*2 * len(vH)+1):len(vH)]))
    sRsb = sHsbMax  # Attenuation in dB

    return vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax

def anFiltKaiser(vHFilt, sFs, strType='lowpass', sFpb1=None, sFsb1=None, sFpb2=None, sFsb2=None):
    """
    Analyze the frequency response of a FIR filter using the Kaiser window method.

    Parameters:
    vHFilt : Filter coefficients
    sFs    : Sampling frequency
    sFpb1  : Passband lower edge frequency (for bandpass/highpass) or single passband edge (for lowpass)
    sFpb2  : Passband upper edge frequency (only for bandpass)
    sFsb1  : Stopband lower edge frequency (for bandpass/highpass) or single stopband edge (for lowpass)
    sFsb2  : Stopband upper edge frequency (only for bandpass)
    strType: Type of filter ('lowpass', 'bandpass', 'highpass')

    Returns:
    vw      : Frequency array
    vH      : Frequency response
    sRpb    : Passband ripple (dB)
    sRsb    : Stopband attenuation (dB)
    sHpbMin : Minimum passband gain
    sHpbMax : Maximum passband gain
    sHsbMax : Maximum stopband gain
    """

    (vw, vH) = sigP.freqz(vHFilt)

    # Normalize frequencies
    freqs = vw * sFs / (2 * np.pi)

    if strType == 'lowpass':
        # Calculate passband ripple
        vPassInd = np.where((freqs >= 0) & (freqs <= sFpb1))[0]
        sPbResp = np.abs(vH[vPassInd])
        sHpbMin = np.min(sPbResp)
        sHpbMax = np.max(sPbResp)
        sRpb = 20 * np.log10(sHpbMax / sHpbMin)  # Ripple in dB

        # Calculate stopband attenuation
        vStopInd = np.where((freqs >= sFsb1) & (freqs <= sFs / 2))[0]
        sStpResp = np.abs(vH[vStopInd])
        sHsbMax = np.max(sStpResp)
        sRsb = -20 * np.log10(sHsbMax)  # Attenuation in dB

    elif strType == 'bandpass':
        # Calculate passband ripple for bandpass filter
        vPassInd = np.where((freqs >= sFpb1) & (freqs <= sFpb2))[0]
        sPbResp = np.abs(vH[vPassInd])
        sHpbMin = np.min(sPbResp)
        sHpbMax = np.max(sPbResp)
        sRpb = 20 * np.log10(sHpbMax / sHpbMin)  # Ripple in dB

        # Calculate stopband attenuation for bandpass filter
        vStopInd1 = np.where((freqs >= 0) & (freqs <= sFsb1))[0]
        vStopInd2 = np.where((freqs >= sFsb2) & (freqs <= sFs / 2))[0]
        sStpResp1 = np.abs(vH[vStopInd1])
        sStpResp2 = np.abs(vH[vStopInd2])
        sHsbMax1 = np.max(sStpResp1)
        sHsbMax2 = np.max(sStpResp2)
        sHsbMax = max(sHsbMax1, sHsbMax2)
        sRsb = -20 * np.log10(sHsbMax)  # Attenuation in dB

    elif strType == 'highpass':
        # Calculate passband ripple for highpass filter
        vPassInd = np.where((freqs >= sFpb1) & (freqs <= sFs / 2))[0]
        sPbResp = np.abs(vH[vPassInd])
        sHpbMin = np.min(sPbResp)
        sHpbMax = np.max(sPbResp)
        sRpb = 20 * np.log10(sHpbMax / sHpbMin)  # Ripple in dB

        # Calculate stopband attenuation for highpass filter
        vStopInd = np.where((freqs >= 0) & (freqs <= sFsb1))[0]
        sStpResp = np.abs(vH[vStopInd])
        sHsbMax = np.max(sStpResp)
        sRsb = -20 * np.log10(sHsbMax)  # Attenuation in dB

    else:
        raise ValueError("Unsupported filter type. Use 'lowpass', 'bandpass', or 'highpass'.")

    return vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax