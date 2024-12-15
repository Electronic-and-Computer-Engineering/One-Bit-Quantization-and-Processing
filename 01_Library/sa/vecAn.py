# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:42:29 2024

@author: mayerflo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

def safelog10(x):
    return np.log10(np.maximum(x, 1e-10))

def vecAn(sErrorVec, vX, vBlockIdx, fs, sNbins=None, sThr=None):
    # Calculate delta values (difference between consecutive elements)
    vDeltaValues = np.diff(sErrorVec)
    
    # Set threshold to mean of delta values if not provided
    if sThr is None:
        sThr = np.mean(vDeltaValues)
    
    # Set sNbins to fs if not provided
    if sNbins is None:
        sNbins = fs
    
    # Identify delta values that exceed the threshold
    vExcIdxs = np.where(vDeltaValues > sThr)[0]
    vExcIVals = vDeltaValues[vExcIdxs]
    
    # Select the rows in vBlockIdx corresponding to the exceeding indices
    vBlocksAboveThr = vBlockIdx[vExcIdxs]
    
    # Plot the original error vector and delta values
    plt.figure(figsize=(14, 6))
    
    # Plot the original error vector
    plt.subplot(2, 1, 1)
    plt.plot(sErrorVec, marker='o', linestyle='-', color='b')
    plt.title('Error Vector')
    plt.xlabel('Index')
    plt.ylabel('Error Value')
    
    # Plot the delta values
    plt.subplot(2, 1, 2)
    plt.stem(vDeltaValues)
    plt.axhline(y=sThr, color='g', linestyle='--', label=f'Threshold = {sThr:.5f}')
    plt.title('Delta Error Values')
    plt.xlabel('Index')
    plt.ylabel('Delta Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Create the main window
    root = tk.Tk()
    root.title("Error Analysis")

    # Create a notebook (tabs container)
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)
    
    # Create tabs for each block
    for i, (start_idx, end_idx) in enumerate(vBlocksAboveThr):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=f'Block {i + 1}')

        # Create a figure for the time-domain plot and FFT
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        snippet = vX[int(start_idx):int(end_idx)]
        
        # Plot the discrete time snippet of vX
        ax1.plot(snippet, marker='o', linestyle='-', color='b')
        ax1.set_title(f'Delta Index: {vExcIdxs[i]}, Delta Value: {vExcIVals[i]:.5f}')
        ax1.set_xlabel(f'Index (Start: {int(start_idx)}, End: {int(end_idx)})')
        ax1.set_ylabel('Signal Value')
        
        # Compute and plot the FFT magnitude
        vBlock = np.fft.fft(snippet, sNbins)
        vBlockMag = 20 * safelog10(np.abs(vBlock) / np.max(np.abs(vBlock)))
        freqs = np.fft.fftfreq(sNbins, d=1/fs)
        ax2.plot(freqs[:len(freqs)//2], vBlockMag[:len(vBlockMag)//2], linestyle='-', color='r')
        ax2.set_title('FFT of the Snippet')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude (dB)')
        
        # Embed the matplotlib figure into the Tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig)  # Close the figure to prevent it from displaying separately

    # Run the Tkinter event loop
    root.mainloop()
    
    # Output the exceeding values and their indices
    output = {
        'exceeding_indices': vExcIdxs,
        'exceeding_values': vExcIVals,
        'threshold': sThr,
        'blocks_above_threshold': vBlocksAboveThr
    }
    
    return output