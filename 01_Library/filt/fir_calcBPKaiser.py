# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:02:31 2024

@author: mayerflo
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sigP
from matplotlib.ticker import FuncFormatter, MultipleLocator

import filt

def fir_calcBPKaiser(sFs, sFpb1, sFpb2, sFsb1, sFsb2, sApb, sAsb, sMinPhase = False):
    """
    Bandpass filter design using the Kaiser window method.
    
    Parameters:
    sFs   : Sampling frequency
    sFpb1 : Passband lower cutoff frequency
    sFpb2 : Passband upper cutoff frequency
    sFsb1 : Stopband lower cutoff frequency
    sFsb2 : Stopband upper cutoff frequency
    sApb  : Passband ripple (dB)
    sAsb  : Stopband attenuation (dB)
    
    Returns:
    vHFilt  : Filter coefficients
    vw      : Frequency array
    vH      : Frequency response
    sRpb    : Passband ripple
    sRsb    : Stopband attenuation
    sHpbMin : Minimum passband gain
    sHpbMax : Maximum passband gain
    sHsbMax : Maximum stopband gain
    """

    # Digital filter specifications
    sWidthDig1 = (sFpb1 - sFsb1) / (sFs / 2)
    sWidthDig2 = (sFsb2 - sFpb2) / (sFs / 2)
    sWidthDig = min(sWidthDig1, sWidthDig2)  # Narrowest transition width

    # Cutoff frequencies for the bandpass filter
    sCutDig = [sFpb1 / (sFs / 2), sFpb2 / (sFs / 2)]

    # Kaiser window parameters
    sFiltordK, sBeta = sigP.kaiserord(sAsb - sApb, sWidthDig)
    
    # Bandpass filter design
    vHFilt = sigP.firwin(
            sFiltordK, 
            sCutDig, 
            window=('kaiser', sBeta),
            pass_zero = False  # Bandpass filter
            )           
    # Convert to minimum-phase filter if flag is set
    if sMinPhase:
        vHFilt = sigP.minimum_phase(vHFilt, n_fft = len(vHFilt)*64, method='homomorphic')
        
    # Analyze the filter's frequency response using the new anFiltKaiser call
    (vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.anFiltKaiser(vHFilt, sFs, strType='bandpass', sFpb1=sFpb1, sFpb2=sFpb2, sFsb1=sFsb1, sFsb2=sFsb2)

    return (vHFilt, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax)