# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:29:28 2024

@author: mayerflo
"""
import numpy as np

def linWeightVec(sN, sMaxWeight):
    if sN < 1:
        raise ValueError("sN must be at least 1")
    
    vWeights = np.zeros((sN,1)).flatten()  
    vWeights[0] = sMaxWeight
    
    if sN == 1:
        return vWeights
    
    # Generate decreasing values for the remaining T-1 coefficients
    sRemSum = 1 - sMaxWeight
    vDecWeights = np.linspace(sRemSum / (sN-1), 0, sN-1)
    vWeights[1:] = vDecWeights * (sRemSum / np.sum(vDecWeights))
    
    return vWeights