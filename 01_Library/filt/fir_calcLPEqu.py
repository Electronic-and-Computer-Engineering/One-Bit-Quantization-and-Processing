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

def fir_calcLPEqu(sFs, sFpb, sFsb, sApb, sAsb, sN):

    vBands = np.array([0., sFpb/sFs, sFsb/sFs, .5])

    # Remez weight calculation:
    # https://www.dsprelated.com/showcode/209.php

    sErr_pb = (1 - 10**(-sApb/20))/2      # /2 is not part of the article above, but makes it work much better.
    sErr_sb = 10**(-sAsb/20)

    sW_pb = 1/sErr_pb
    sW_sb = 1/sErr_sb
    
    vHFilt = sigP.remez(
            sN,                          # Desired number of taps
            vBands,                      # All the band inflection points
            [1,0]                       # Desired gain for each of the bands: 1 in the pass band, 0 in the stop band
            )               
    
    (vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.anFiltEqu(vHFilt, sFs, sFpb, sFsb, 'lowpass')

    return (vHFilt, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax)

