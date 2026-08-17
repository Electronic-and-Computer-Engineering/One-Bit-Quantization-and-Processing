# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 11:14:31 2025

@author: mayerflo
"""
from numpy.linalg import solve, lstsq, pinv, matrix_rank
import numpy as np

def linsolve(A, b):
    """
    MATLAB-like backslash operator using NumPy.
    Returns minimum-norm solution if singular or rank-deficient.
    """
    A = np.asarray(A)
    b = np.asarray(b)

    if matrix_rank(A) < min(A.shape):
        bFeas = False
        #print("⚠ Warning: Matrix is rank-deficient.")
    else: 
        bFeas = True    

    try:
        return solve(A, b), bFeas
    except np.linalg.LinAlgError:
        if A.shape[0] >= A.shape[1]:
            x, _, _, _ = lstsq(A, b, rcond=None)
            return x, bFeas
        else:
            return pinv(A) @ b, bFeas