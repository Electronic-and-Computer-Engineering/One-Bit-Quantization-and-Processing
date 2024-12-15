# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:02:31 2024

@author: mayerflo
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sigP
from scipy.fftpack import hilbert

import filt

def fir_calcLPKaiser(sFs, sFpb, sFsb, sApb, sAsb, sTaps = None, sMinPhase = False):
    """
    Lowpass filter design using the Kaiser window method.
    
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
    sWidthDig = (sFsb - sFpb) / (sFs / 2)  # Transition width

    # Cutoff frequency for the lowpass filter
    sCutDig = sFpb / (sFs / 2)

    # Kaiser window parameters
    if sTaps == None:
        sFiltordK, sBeta = sigP.kaiserord(sAsb - sApb, sWidthDig)
        if sFiltordK % 2 == 0:
            sFiltordK += 1
            print("Found even sFiltordK. Incrementing to odd sFiltord=%d" % sFiltordK)
        # Lowpass filter design
        vHFilt = sigP.firwin(
                sFiltordK, 
                sCutDig, 
                window=('kaiser', sBeta),
                pass_zero='lowpass'  # Lowpass filter
                )  
    else:
        sFiltordK = sTaps 
        # Lowpass filter design
        vHFilt = sigP.firwin(
                sFiltordK, 
                sCutDig, 
                window= 'blackman',
                pass_zero='lowpass'  # Lowpass filter
                )    
        
    # Convert to minimum-phase filter if flag is set
    if sMinPhase:
        vHFilt = sigP.minimum_phase(vHFilt, method='homomorphic')
    
    # Analyze the filter's frequency response using the new anFiltKaiser call
    (vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.anFiltKaiser(vHFilt, sFs, strType='lowpass', sFpb1=sFpb, sFsb1=sFsb)

    return (vHFilt, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax)