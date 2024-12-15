# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:47:14 2024

@author: mayerflo
"""

import numpy as np

def MtxIsSing(mMtx):
    # Berechnung der Determinante
    det = np.linalg.det(mMtx)
    return np.isclose(det, 0)
