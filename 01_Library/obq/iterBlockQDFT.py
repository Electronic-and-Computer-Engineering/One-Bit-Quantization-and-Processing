import numpy as np
import sg, sp, globalTools
import obq
import os

os.makedirs("analysis", exist_ok=True)


def iterBlockQDFT(vx, vW, sK, sM, mRange,
                   sHop=None, sType='grb', verbose=True, sSweeps=0):
    """
    Iterative block-based one-bit DFT-domain quantization with EXACT,
    unbounded global error accumulation.

    Core idea: instead of a bounded memory window (sL) with OLA tapering,
    the error of the ENTIRE already-finalized past is accumulated exactly
    (vE_cum). Algebraically this is equivalent to optimizing each block
    against the global target spectrum of the full signal length N -- the
    still-undecided "future" part cancels out exactly, because it is
    approximated in an unbiased way by x itself (see derivation in chat).

    For overlapping blocks (sHop < sM), only the sHop-exclusive part of a
    block -- the part that is truly final -- is committed to vE_cum. The
    remainder (beyond sHop) is still provisional, since it will be
    overwritten by the next block, and stays implicitly accounted for via
    the x placeholder in Fw_m @ vx_m. This replaces the previous OLA
    tapering with exact bookkeeping -- no sL, no sWinLen, no Hamming window
    needed anymore.

    Args:
        vx:      Input signal (1D array), length N
        vW:      Filter frequency response (one-sided, [0, pi)), length K_orig
        sK:      Number of frequency bins for optimization (resolution hyperparameter)
        sM:      Length of optimizable segment per block
        mRange:  Frequency range(s) [StopStart, PassStart, PassEnd, StopEnd]
                 in radians/sample. Shape (4,) for single band, (B, 4) for B bands.
        sHop:    Hop size between blocks (default = sM, i.e. non-overlapping)
        sType:   Optimization type ('grb', 'other')
        verbose: If True, print block-wise info
        sSweeps: Number of additional refinement passes AFTER the causal
                 Pass 1. 0 = Pass 1 only (default). From Pass 2 onward, vb is
                 already fully known everywhere -- so for each block the
                 REAL remainder of the signal (past AND future) is removed
                 from the global spectral balance B_full, instead of
                 estimating the future via the x placeholder as in Pass 1.
                 Classic block-coordinate-descent / Gauss-Seidel on
                 || Fw_full @ (x - b) ||^2, with Pass 1 as the warm start.
    Returns:
        vb:        Quantized one-bit signal, length N
        vBlockErr: Per-block error from Pass 1
        vSweepErr: Global spectral error after each sweep (empty if sSweeps == 0)
    """

    sxLen = len(vx)
    swLen = len(vW)

    if sHop is None:
        sHop = sM

    mRange_arr = np.asarray(mRange)
    if mRange_arr.ndim == 1:
        freq_range = [mRange_arr[0], mRange_arr[3]]
    else:
        freq_range = [mRange_arr[:, 0], mRange_arr[:, 3]]

    # Full signal DFT matrix -- computed once, sliced per block.
    F_full, _ = sg.dftMat(sxLen, sK, freq_range=freq_range)   # shape: (sK x N)
    F_w, _    = sg.dftMat(swLen, sK, freq_range=freq_range)
    vW_D      = F_w @ vW
    Fw_full   = vW_D[:, None] * F_full                        # shape: (sK x N)

    sNumBlocks = (sxLen - sM) // sHop + 1
    vBlockErr  = np.zeros(sNumBlocks)
    vb         = np.zeros(sxLen)

    # Exact, unbounded error accumulation of the already FINALIZED past --
    # replaces vE_l (previously: bounded sL window + OLA taper).
    vE_cum = np.zeros(Fw_full.shape[0], dtype=complex)

    for m in range(sNumBlocks):
        if m == 0:
            progressDFTBlock = globalTools.SimpleProgressBar(
                sNumBlocks, width=40,
                prefix="BlockDFT",
                fill="█", empty=" ", end=" ✓"
            )

        sStIdx  = m * sHop
        sEndIdx = sStIdx + sM

        if sEndIdx > sxLen:
            if verbose:
                print(f"Skipping block {m}: exceeds signal length.")
            continue

        vx_m = vx[sStIdx : sEndIdx]                       # shape: (sM,)
        Fw_m = Fw_full[:, sStIdx : sEndIdx]                # shape: (sK x sM), NO taper anymore

        # Target spectrum: exact global past + current block (x placeholder)
        vX_t_L_c = vE_cum + Fw_m @ vx_m

        mRIFw_m  = np.vstack([Fw_m.real, Fw_m.imag])       # shape: (2K x sM)
        vRIX_t_L = np.hstack([vX_t_L_c.real, vX_t_L_c.imag])

        if sType == 'grb':
            vbm, sBlockErr = obq.OptDFT(vx_m, vRIX_t_L, mRIFw_m, sM)
        else:
            vbm = obq.combOptBlock(vx_m, vW_D, np.zeros(sM))
            sBlockErr = np.nan

        vb[sStIdx : sEndIdx] = vbm
        vBlockErr[m] = sBlockErr

        # Only commit the final (sHop-exclusive) part of the block --
        # the rest will still be overwritten by the next block and stays
        # accounted for via the x placeholder in vX_t_L_c until then.
        sCommitEnd = min(sStIdx + sHop, sxLen)
        Fw_commit  = Fw_full[:, sStIdx : sCommitEnd]
        vE_cum    += Fw_commit @ (vx[sStIdx : sCommitEnd] - vb[sStIdx : sCommitEnd])

        progressDFTBlock.update(m)

    # ------------------------------------------------------------------
    # Optional further passes (Sweep 2, 3, ...): vb is now known
    # everywhere, so the REAL remainder of the signal (instead of an x
    # placeholder for the future) can be used. Block-coordinate-descent
    # on the exact global target, Pass 1 serves as the warm start.
    # ------------------------------------------------------------------
    vSweepErr = np.zeros(sSweeps)
    if sSweeps > 0:
        X_tilde = Fw_full @ vx           # fixed global target spectrum
        B_full  = Fw_full @ vb           # spectrum of the current (Pass-1) result

        for s in range(sSweeps):
            if verbose:
                progressSweep = globalTools.SimpleProgressBar(
                    sNumBlocks, width=40,
                    prefix=f"Sweep {s+1}/{sSweeps}",
                    fill="█", empty=" ", end=" ✓"
                )
                
            vb_old = vb.copy()
            
            for m in range(sNumBlocks):
                sStIdx  = m * sHop
                sEndIdx = sStIdx + sM
                if sEndIdx > sxLen:
                    continue
                
                vx_m     = vx[sStIdx:sEndIdx]
                Fw_m     = Fw_full[:, sStIdx:sEndIdx]
                vb_m_old = vb[sStIdx:sEndIdx]

                # Remove this block's contribution entirely from the global
                # balance (not just the previously "committed" sHop part --
                # by now the whole block is already final and may be revised)
                B_rest   = B_full - Fw_m @ vb_m_old
                target_m = X_tilde - B_rest

                mRIFw_m   = np.vstack([Fw_m.real, Fw_m.imag])
                vRItarget = np.hstack([target_m.real, target_m.imag])

                vbm_new, _ = obq.OptDFT(vx_m, vRItarget, mRIFw_m, sM)

                B_full += Fw_m @ (vbm_new - vb_m_old)
                vb[sStIdx:sEndIdx] = vbm_new

                if verbose:
                    progressSweep.update(m)

            vSweepErr[s] = float(np.linalg.norm(X_tilde - B_full) ** 2)
            
            if s > 0 and vSweepErr[s] > vSweepErr[s - 1]:
                vb = vb_old.copy()
                vSweepErr = vSweepErr[:s]
                break
            
            if verbose:
                print(f"\n  Sweep {s+1}/{sSweeps}: global spectral error = {vSweepErr[s]:.6g}")

    return vb, vBlockErr, vSweepErr


# import numpy as np
# import sg, sp, globalTools
# import obq
# import os

# os.makedirs("analysis", exist_ok=True)

# def iterBlockQDFT(vx, vW, sK,
#                   sM, sL, mRange, sWinLen,
#                   sHop=None, sType='grb', verbose=True):
#     """
#     Iterative block-based one-bit DFT-domain quantization.
#     Args:
#         vx:      Input signal (1D array), length N
#         vW:      Filter frequency response (one-sided, [0, pi)), length K_orig
#         sK:      Number of frequency bins for optimization (resolution hyperparameter)
#         sM:      Length of optimizable segment per block
#         sL:      Max memory length in samples (Gedächtnistiefe)
#         mRange:  Frequency range(s) [StopStart, PassStart, PassEnd, StopEnd]
#                  in radians/sample. Shape (4,) for single band, (B, 4) for B bands.
#         sHop:    Hop size between blocks (default = sM)
#         sType:   Optimization type ('grb', 'other')
#         verbose: If True, print block-wise info
#     Returns:
#         vb:      Quantized one-bit signal, length N
#     """
    
#     bTune = False
#     #vx = np.pad(vx, (sL, 0), mode='constant')
#     sxLen   = len(vx)
#     swLen   = len(vW)
#     sOlaLen = sL + sWinLen
    

#     if sHop is None:
#         sHop = sM

#     # Total number of blocks covering the signal
#     sNumBlocks = (sxLen - sM) // sHop + 1

#     vBlockErr  = np.zeros(sNumBlocks)
#     vb         = np.zeros(sxLen)

#     # ------------------------------------------------------------------
#     # Precomputation -- all quantities that are identical for every block
#     # ------------------------------------------------------------------

#     # Interpolate filter weights vW to sK points within the passband.
#     # vW_D carries the perceptual weighting; sK controls frequency resolution.
#     #vW_D       = sp.getFiltWeights(vW, sK, mRange)           # shape: (sK,)

#     # Extract the full frequency range [StopStart, StopEnd] from mRange.
#     mRange_arr = np.asarray(mRange)
#     if mRange_arr.ndim == 1:
#         freq_range = [mRange_arr[0], mRange_arr[3]]
#     else:
#         freq_range = [mRange_arr[:, 0], mRange_arr[:, 3]]

#     # Full signal DFT matrix -- computed once, sliced per block.
#     F_full, _  = sg.dftMat(sxLen, sK, freq_range=freq_range)  # shape: (sK x N)
#     F_w, _     = sg.dftMat(swLen, sK, freq_range=freq_range)  # shape: (sK x N)
#     vW_D       = F_w @ vW
#     Fw_full    = vW_D[:, None] * F_full                       # shape: (sK x N)
#     #vX_tilde   = Fw_full @ vx
    
#     # ------------------------------------------------------------------
#     # Main loop -- process one block of sM samples per iteration
#     # ------------------------------------------------------------------
    
#     # Coherent M-block projection matrix
#     #k = 0 
#     vOlaSum          = np.zeros(sOlaLen)     
#     #vWin             = np.hanning(sWinLen-1)
#     vWinOla          = np.hamming(sWinLen)
#     #vWinCut     = vWin[len(vWin)//2 - sM//2 + k:len(vWin)//2 + sM//2 + k]
    
#     for wIdx in range(sL//sHop):
#         sStIdx  = wIdx * sHop
#         if sStIdx <= sL:
#             sOlaWStart = sStIdx
#             sOlaWEnd = sOlaWStart + sM
#             vOlaSum[sStIdx : sStIdx + sWinLen] += vWinOla
#             sOlaMax  = vOlaSum.max() if vOlaSum.max() > 0 else 1.0
#             vOlaNorm = vOlaSum / sOlaMax           
#         else:
#             break
        
#     sOlaWStart  = sOlaLen - sWinLen
#     sOlaWEnd    = sOlaWStart + sM
            
#     for m in range(sNumBlocks):
#         if m == 0:
#             progressDFTBlock = globalTools.SimpleProgressBar(
#                 sNumBlocks, width=40,
#                 prefix="BlockDFT",
#                 fill="█", empty=" ", end=" ✓"
#             )
        
#         sStIdx  = m * sHop
#         sEndIdx = sStIdx + sM

#         if sEndIdx > sxLen:
#             if verbose:
#                 print(f"Skipping block {m}: exceeds signal length.")
#             continue

#         # Signal samples for the current optimizable block
#         vx_m = vx[sStIdx : sEndIdx]                      # shape: (sM,)

#         # Memory segment: grows from 0 to sL during the first sL/sHop blocks,
#         # then slides forward with the hop size.
#         if sStIdx <= sL:
#             sl_cur = sStIdx           

#         else:
#             sl_cur = sL
            
#         #vOla, vOla_norm = olaWeights(sStIdx, vWinOla, sHop, sL)
            
#         vb_l        = vb[sStIdx - sl_cur : sStIdx]
#         vx_l        = vx[sStIdx - sl_cur : sStIdx]
#         Fw_l        = Fw_full[:,sStIdx - sl_cur : sStIdx] * vOlaNorm[sOlaWStart - sl_cur : sOlaWStart] 
#         vd_l        = vx_l - vb_l
#         vE_l        = Fw_l @ vd_l
        
#         Fw_m        = Fw_full[:, sStIdx : sEndIdx] * vOlaNorm[sOlaWStart : sOlaWEnd]       # shape: (sK x sM)
#         mRIFw_m     = np.vstack([Fw_m.real, Fw_m.imag])         # shape: (2K x sM)
        
#         ############ BQP #######################
#         #mFg         = mRIFw_m.T @ mRIFw_m       
#         #vE_hat      = vE_l + Fw_m @ vx_m
#         #vB_l        = Fw_l @ vb_l
#         #vE_tilde    = vX_tilde - vB_l
#         #vE_R        = vE_hat + vE_tilde
        
#         #vRIEr       = np.hstack([vE_R.real, vE_R.imag])
#         ############ BQP #######################
#         vX_t_L_c = vE_l + Fw_m @ vx_m
        
#         # Real/imag stacking after projection
#         vRIX_t_L = np.hstack([vX_t_L_c.real, vX_t_L_c.imag])           # shape: (2K,)
        
        
#         if sType == 'grb':
            
#             if ((bTune == True) & (m == 90)):
#                vbm, sBlockErr, tune_info = obq.OptDFT(vx_m, vRIX_t_L, mRIFw_m, sM, bTune)
#             else:
#                vbm, sBlockErr, = obq.OptDFT(vx_m, vRIX_t_L, mRIFw_m, sM)
#             #vbm, sBlockErr = obq.OptDFT(mRIFw_m, mFg, vRIEr, sM)    
#         else:
#             vbm = obq.combOptBlock(vx_m, vW_D, np.zeros_like(vb_l))

#         # Write the optimized binary block into the output signal
#         vb[sStIdx : sEndIdx] = vbm                           # shape: (sM,)
#         vBlockErr[m] = sBlockErr

#         progressDFTBlock.update(m)


#     return vb, vBlockErr
