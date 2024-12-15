# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:25:27 2024

@author: mayerflo
"""

import numpy as np

import sys
sys.path.append('../01_Library')
# individual packages
import sg, sa, sp, obq, filt

def cRadix2FFT(vx):
    # Find the next power of 2 for zero-padding
    sP = int(np.ceil(np.log2(len(vx))))
    sNumSamples = 2 ** sP
    vx = np.concatenate((vx, np.zeros(sNumSamples - len(vx), dtype=complex)))
    
    # Bit-reverse order the input array
    vxRev = sp.bitRevOrder(vx)
    
    vx_real = np.real(vxRev).copy()
    vx_imag = np.imag(vxRev).copy()
    
    sO = int(np.log2(sNumSamples))
    sHalf = 1
    
    for sStage in range(1, sO + 1):
        sG = sNumSamples // (2 ** sStage)
        for idx in range(0, sNumSamples, 2 ** sStage):
            for sn in range(sHalf):
                sPos = sn + idx
                sk = (2 ** (sO - sStage)) * sn
                w_real = np.cos((2 * np.pi * sk) / sNumSamples)
                w_imag = -np.sin((2 * np.pi * sk) / sNumSamples)
                
                realPathTemp = vx_real[sPos + sHalf] * w_real - vx_imag[sPos + sHalf] * w_imag
                imagPathTemp = vx_imag[sPos + sHalf] * w_real + vx_real[sPos + sHalf] * w_imag
                
                vx_real[sPos + sHalf] = vx_real[sPos] - realPathTemp
                vx_imag[sPos + sHalf] = vx_imag[sPos] - imagPathTemp
                
                vx_real[sPos] = vx_real[sPos] + realPathTemp
                vx_imag[sPos] = vx_imag[sPos] + imagPathTemp
        
        sHalf *= 2
    
    XReal = vx_real
    XImag = vx_imag
    
    return XReal, XImag