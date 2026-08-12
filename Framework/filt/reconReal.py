#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:48:03 2026

@author: mayerflo
"""

import numpy as np

def reconReal(vSig, vFilt):
    """Non-ideal reconstruction: finite length FIR."""
    return np.convolve(vFilt, vSig, 'same')