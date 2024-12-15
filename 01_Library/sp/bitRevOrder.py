# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:03:45 2024

@author: mayerflo
"""

import numpy as np

def bitRevOrder(vec):
    """Generate bit-reversed indices for an array of length n (must be a power of 2)."""
    """Reorder a vector according to bit-reversed indices."""
    sN = len(vec)
    if not (sN != 0 and ((sN & (sN - 1)) == 0)):
        raise ValueError("Length of the input vector must be a power of 2.")
    
    sP = int(np.log2(sN))
    bitRevIdx = [int(bin(i)[2:].zfill(sP)[::-1], 2) for i in range(sN)]
    return vec[bitRevIdx]