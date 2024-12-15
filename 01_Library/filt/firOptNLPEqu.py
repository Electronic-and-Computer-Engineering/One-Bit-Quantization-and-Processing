# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:10:29 2024

@author: mayerflo
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sigP
from matplotlib.ticker import FuncFormatter, MultipleLocator

import filt

def firOptNLPEqu(sFs, sFpb, sFsb, sApb, sAsb, sNmin = 2, sNmax = 1000):
    for sN in range(sNmin, sNmax):
        (vHFilt, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.fir_calcLPEqu(sFs, sFpb, sFsb, sApb, sAsb, sN)
        if -dB20(sRpb) <= sApb and -dB20(sRsb) >= sAsb:
            print("Trying up to N=%d" % sN)
            print("Rpb: %fdB" % (-dB20(sRpb)))
            print("Rsb: %fdB" % -dB20(sRsb))
            if sN % 2 == 0:
                sN += 1
                print("Found even sN. Incrementing to odd sN=%d" % sN)
            return sN
    return None

def dB20(array):
    with np.errstate(divide='ignore'):
        return 20 * np.log10(array)
