# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:50:54 2024

@author: mayerflo
"""

import numpy as np

def dftTrans(vx, sK):
    """
    Computes the DFT of a signal using an extended DFT transformation matrix of size M x N.
    
    Parameters:
    signal (array-like): The input signal for which to compute the DFT.
    K (int): The number of frequency bins (rows of the DFT matrix).
    
    Returns:
    numpy.ndarray: The DFT of the input signal with extended frequency resolution.
    """
    sN = len(vx)
    # Create the index arrays for matrix construction
    vn = np.arange(sN)        # [0, 1, 2, ..., N-1]
    vk = np.arange(sK).reshape((sK, 1))  # Column vector [0, 1, 2, ..., M-1].T

    # Create the extended DFT matrix
    mTrans = np.exp(-2j * np.pi * vk * vn / sK)
    
    # Perform the matrix multiplication to compute the extended DFT
    vX = mTrans @ vx
    
    return vX
