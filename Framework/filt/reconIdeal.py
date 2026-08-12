#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:47:27 2026

@author: mayerflo
"""
import numpy as np

def reconIdeal(vSig, vFiltFFT):
    """Ideal reconstruction: circular convolution == bin mask in freq domain."""
    return np.fft.ifft(np.fft.fft(vSig) * vFiltFFT).real