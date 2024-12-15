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

def fir_calcHPKaiser(sFs, sFpb, sFsb, sApb, sAsb):
    """
    Highpass filter design using the Kaiser window method.
    
    Parameters:
    sFs   : Sampling frequency
    sFpb  : Passband cutoff frequency
    sFsb  : Stopband cutoff frequency
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
    sWidthDig = (sFpb - sFsb) / (sFs / 2)  # Transition width

    # Cutoff frequency for the highpass filter
    sCutDig = sFpb / (sFs / 2)

    # Kaiser window parameters
    sFiltordK, sBeta = sigP.kaiserord(sAsb - sApb, sWidthDig)
    
    # Highpass filter design
    vHFilt = sigP.firwin(
            sFiltordK, 
            sCutDig, 
            window=('kaiser', sBeta),
            pass_zero = False,  # Highpass filter (use False for bandpass/highpass, True for lowpass)
            scale=False
            )           
    
    # Analyze the filter's frequency response using the new anFiltKaiser call
    (vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.anFiltKaiser(vHFilt, sFs, strType='highpass', sFpb1=sFpb, sFsb1=sFsb)

    return (vHFilt, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax)