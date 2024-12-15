# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:28:52 2024

@author: mayerflo
"""

import numpy as np


def dftMat(sN, sK):
    """
    Computes the full DFT matrix of size N x N.

    Parameters:
    - N: Size of the DFT matrix (signal length).

    Returns:
    - Real and imaginary parts of the DFT matrix.
    """
    vn = np.arange(sN)        # [0, 1, 2, ..., N-1]
    vk = np.arange(sK).reshape((sK, 1))  # Column vector [0, 1, 2, ..., M-1].T

    # Create the extended DFT matrix
    mOmega = np.exp(-2j * np.pi * vk * vn / sK)
    return mOmega