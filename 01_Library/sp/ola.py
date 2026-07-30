#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 14:18:13 2025

@author: mayerflo
"""
import numpy as np

def ola(d_blocks, g, H, N=None):
    """
    Simple Overlap-Add (OLA).

    Parameters
    ----------
    d_blocks : list or 2D array, shape (P, M)
        Zeitblöcke d^(q)
    g : 1D array, shape (M,)
        Fenster (Synthese-Fenster)
    H : int
        Hop-Size
    N : int, optional
        Gesamtlänge des Ziels; falls None → automatisch aus Blöcken berechnet

    Returns
    -------
    e : 1D ndarray
        Überlappt-addierte Zeitreihe e[n]
    """
    d_blocks = np.asarray(d_blocks)
    if d_blocks.ndim == 1:
        d_blocks = d_blocks[None, :]  # single block

    P, M = d_blocks.shape
    if N is None:
        N = (P - 1) * H + M

    e = np.zeros(N)
    norm = np.zeros(N)

    for q in range(P):
        start = q * H
        end = min(start + M, N)
        g_seg = g[:end - start]
        e[start:end] += g_seg * d_blocks[q, :end - start]
        norm[start:end] += g_seg

    # Normalisieren, falls Fenster keine COLA-Bedingung erfüllt
    e /= (norm + 1e-12)
    return e
