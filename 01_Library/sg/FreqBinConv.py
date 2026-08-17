import numpy as np

def rad2bin(omega, N):
    omega = np.asarray(omega)
    bins = np.round(omega * N / (2 * np.pi)).astype(int)
    return bins.item() if bins.ndim == 0 else bins

def bin2rad(k, N):
    k = np.asarray(k)
    omega = 2 * np.pi * k / N
    return omega.item() if omega.ndim == 0 else omega

def freq2rad(f_Hz, fs):
    f_Hz = np.asarray(f_Hz)
    omega = 2 * np.pi * f_Hz / fs
    return omega.item() if omega.ndim == 0 else omega

def rad2freq(omega, fs):
    omega = np.asarray(omega)
    f_Hz = omega * fs / (2 * np.pi)
    return f_Hz.item() if f_Hz.ndim == 0 else f_Hz

def freq2bin(f_Hz, N, fs):
    f_Hz = np.asarray(f_Hz)
    bins = np.round(f_Hz * N / fs).astype(int)
    return bins.item() if bins.ndim == 0 else bins

def bin2freq(k, N, fs):
    k = np.asarray(k)
    f_Hz = k * fs / N
    return f_Hz.item() if f_Hz.ndim == 0 else f_Hz