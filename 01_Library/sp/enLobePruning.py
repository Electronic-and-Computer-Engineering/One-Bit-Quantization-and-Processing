import numpy as np
import matplotlib.pyplot as plt

def enLobePruning(vCoeffs, tLobes, sETh=0.1, sBSize=2, bStartAtGrad=False):
    """
    Perform energy-based pruning on FIR filter coefficients and retain structure.
    The starting point can either be at the beginning of the earliest lobe or at the maximum gradient within the earliest lobe.
    
    Args:
    vCoeffs (numpy array): Array of FIR filter coefficients.
    tLobes (list of tuples): List of lobes where each lobe is a tuple (start_idx, end_idx, width).
    sETh (float): Threshold relative to main lobe energy for retaining lobes (default is 0.1, i.e., 10%).
    sBSize (int): Block size used to search for the maximum gradient (default is 1, i.e., consecutive elements).
    start_at_gradient (bool): If True, starts at the maximum gradient within the earliest lobe. Otherwise, starts at the beginning of the earliest lobe.
    
    Returns:
    tuple:
        - vPrunCoeffs (numpy array): Pruned coefficients where insignificant values are set to zero but retain original positions.
        - sPrunIdx (int): The index where pruning starts (either beginning of the earliest lobe or the index of the maximum gradient).
    """
    sNumLobes = len(tLobes)
    
    # Calculate the energy for each lobe
    tLobeEn = []
    for lobeIdx in range(sNumLobes):
        start, end, _ = tLobes[lobeIdx]
        sLobeEn = np.sum(vCoeffs[start:end+1] ** 2)  # +1 to include the end index
        tLobeEn.append((start, end, sLobeEn))
    
    # Sort lobes by their energy contribution in descending order
    tLobeEnSorted = sorted(tLobeEn, key=lambda x: x[2], reverse=True)
    
    # The main lobe is the lobe with the highest energy
    sMainLobeEn = tLobeEnSorted[0][2]
    sEnThr = sMainLobeEn * sETh
    vSortedUsedLobes = np.zeros((sNumLobes,1)).flatten()
    # Initialize an array for pruned coefficients with zeros (same length as the original)
    vPrunCoeffs = np.zeros_like(vCoeffs)
    
    # Keep only lobes that have energy greater than the threshold relative to the main lobe
    for EnLobeIdx in range(sNumLobes):
        start, end, sLobeEn = tLobeEnSorted[EnLobeIdx]
        if tLobeEnSorted[EnLobeIdx][2] > sEnThr:
            startIdx = tLobeEnSorted[EnLobeIdx][0]
            endIdx = tLobeEnSorted[EnLobeIdx][1]
            vSortedUsedLobes[EnLobeIdx] = 1
            vPrunCoeffs[startIdx:endIdx+1] = vCoeffs[startIdx:endIdx+1]  # Keep significant coefficients
    
    # **Find the earliest lobe** (the lobe with the smallest start index)
    sPrunOffset = np.argmax(vPrunCoeffs != 0)
    for lobeIdx in range(sNumLobes):
        if tLobes[lobeIdx][0] == sPrunOffset:
            sEarliestLobeEnd = sPrunOffset + tLobes[lobeIdx][2] // 2   
        
    # Flag to decide where to start pruning
    if bStartAtGrad:
        # Search for the maximum gradient in the earliest lobe using block size sBSize
        sMaxGradient        = -np.inf
        sMaxGradientIdx      = sPrunOffset

        for i in range(sPrunOffset, sEarliestLobeEnd - sBSize + 1):
            sGradient = abs(vCoeffs[i + sBSize] - vCoeffs[i])  # Calculate gradient over block size sBSize
            if sGradient > sMaxGradient:
                sMaxGradient = sGradient
                sMaxGradientIdx = i
        
        sPrunIdx = sMaxGradientIdx
    else:
        # Start at the beginning of the earliest lobe
        vPrunCoeffs = vPrunCoeffs[vPrunCoeffs != 0]
        sPrunIdx    = sPrunOffset
        
    plotLobePruning(vCoeffs, vPrunCoeffs, tLobes, tLobeEnSorted, vSortedUsedLobes, sPrunIdx, bPlot=True)
    # Return the pruned coefficients and the starting index (sPrunIdx)
    return vPrunCoeffs, sPrunIdx

def plotLobePruning(vCoeffs, vPrunCoeffs, tLobes, tLobeEnSorted, vSortedUsedLobes, sPrunIdx, bPlot=True):
    """
    Plot FIR filter coefficients, pruning point, and pruned zones based on used or unused lobes.
    
    Args:
    vCoeffs (numpy array): The original filter coefficients.
    vPrunCoeffs (numpy array): The pruned coefficients.
    tLobes (list of tuples): List of lobes where each lobe is a tuple (start_idx, end_idx, width).
    tLobeEnSorted (list of tuples): Sorted lobes with energy and a flag indicating if used (start_idx, end_idx, width, energy, used_flag).
    sPrunIdx (int): The index where pruning starts.
    bPlot (bool): Flag to decide whether to plot or not.
    """
    
    if not bPlot:
        return  # If plotting is disabled, do nothing
    
    # Set up the plot
    plt.figure(figsize=(12, 6))
    
    # Plot the original coefficients
    plt.plot(vCoeffs, label='Original Coefficients', color='gray', linewidth=1)
    
    # Determine the full range of the coefficient vector
    total_length = len(vCoeffs)
    
    # Flags to ensure we label 'Used' and 'Unused' only once
    used_label_added = False
    unused_label_added = False
    
    # Highlight zones (green for used, red for unused) based on the 'used' flag in tLobeEnSorted
    for lobeIdx in range(len(tLobeEnSorted)):
        start, end, _ = tLobeEnSorted[lobeIdx]  # The 5th element is the 'used' flag
        if vSortedUsedLobes[lobeIdx]:
            # Only add 'Used Zones' label once
            if not used_label_added:
                plt.axvspan(start, end, color='green', alpha=0.3, label='Used Zones')
                used_label_added = True
            else:
                plt.axvspan(start, end, color='green', alpha=0.3)
        else:
            # Only add 'Unused Zones' label once
            if not unused_label_added:
                plt.axvspan(start, end, color='red', alpha=0.3, label='Unused Zones')
                unused_label_added = True
            else:
                plt.axvspan(start, end, color='red', alpha=0.3)
    
    # Plot the pruned coefficients with a thicker line, starting from the pruning index
    plt.plot(np.arange(sPrunIdx, total_length), vCoeffs[sPrunIdx:], label='Pruned Coefficients', color='blue', linewidth=2)
    
    # Highlight the pruning point
    plt.scatter(sPrunIdx, vCoeffs[sPrunIdx], color='blue', marker='o', s=100, label='Pruning Point')
    
    # Titles and labels
    plt.title('FIR Filter Coefficients with Pruned and Unused Zones')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Coefficient Value')
    
    # Add grid and legend
    plt.grid(True)
    plt.legend(loc='best')
    
    # Show plot
    plt.show()
