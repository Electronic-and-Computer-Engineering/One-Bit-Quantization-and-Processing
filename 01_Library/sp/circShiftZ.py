# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:39:49 2024

@author: mayerflo
"""
import numpy as np

def circShiftZ(mMtx, sShift, sAxis=None):
    # Handle 1D array (vector)
    if mMtx.ndim == 1:
        # Roll the array
        mRollMtx = np.roll(mMtx, sShift)
        
        # Create a mask for setting overflow elements to zero
        mMask = np.zeros_like(mMtx, dtype=bool)
        
        if sShift > 0:
            mMask[:sShift] = True
        elif sShift < 0:
            mMask[sShift:] = True
        
        # Apply the mask
        mRollMtx[mMask] = 0
        return mRollMtx
    # Handle 2D array (matrix)
    elif mMtx.ndim == 2:
        # Roll the array
        mRollMtx = np.roll(mMtx, sShift, axis=sAxis)
        
        # Create a mask for setting overflow elements to zero
        mMask = np.zeros_like(mMtx, dtype=bool)
        
        if sShift > 0:
            if sAxis == 1:
                mMask[:, :sShift] = True
            elif sAxis == 0:
                mMask[:sShift, :] = True
        elif sShift < 0:
            if sAxis == 1:
                mMask[:, sShift:] = True
            elif sAxis == 0:
                mMask[sShift:, :] = True
        
        # Apply the mask
        mRollMtx[mMask] = 0
        return mRollMtx
    else:
        raise ValueError("Only 1D and 2D arrays are supported.")