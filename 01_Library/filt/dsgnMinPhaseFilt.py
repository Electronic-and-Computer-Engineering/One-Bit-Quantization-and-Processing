import numpy as np

def prunOptimal(vWlin, sW0Rel=0.2, sMetric='L2', sMaxPrune=None, sNfft=8192,
                bRequireMinPhase=True, bTable=False):
    """
    Prunes the leading taps of a linear-phase FIR such that, subject to

        (1) the result is minimum phase   (all |z| < 1)        [optional]
        (2) w0 >= sW0Rel * max|w|         (controllability of the block problem)

    the magnitude error with respect to vWlin is MINIMAL.

    All cut indices are evaluated and the best one is returned: the error is
    NOT monotone in the cut index, so the smallest feasible cut is not
    automatically the best one.

    Each candidate is LS-rescaled before it is judged (g = <Hcut,Hlin>/||Hcut||^2)
    so that only the shape of the response is compared, not its level.

    Args:
        vWlin:            Linear-phase FIR coefficients.
        sW0Rel:           Required ratio w0/max|w| of the pruned filter.
        sMetric:          'L2'    relative ||g*|Hcut| - |Hlin|||_2 over the full band
                          'maxdB' largest dB deviation where |Hlin| > 1e-3 * max
                          'pass'  largest dB deviation inside the passband (|Hlin| >= 0.5)
        sMaxPrune:        Upper bound for the cut index; None = up to the peak.
        bRequireMinPhase: Enforce constraint (1). Costly - see notes.
        bTable:           Also return the full candidate table.

    Returns:
        vWcut:    Pruned and rescaled filter.
        sPrunIdx: Selected cut index.
        dInfo:    Diagnostics; dInfo['table'] holds all candidates if bTable.

    Note:
        w0 is not monotone in the cut index - it collapses at every zero
        crossing of the impulse response. Restricting the candidates to lobe
        boundaries avoids those dips.
    """
    vw = np.asarray(vWlin, dtype=float).ravel()
    vHl = np.abs(np.fft.rfft(vw, sNfft))
    sMax = vHl.max()
    vPass = vHl >= 0.5 * sMax                      # passband mask
    vDef = vHl > 1e-3 * sMax                       # where the target is defined
    vStop = vHl < 1e-2 * sMax                      # stopband mask

    sLim = int(np.argmax(np.abs(vw))) if sMaxPrune is None else int(sMaxPrune)
    lRows = []
    for sIdx in range(sLim + 1):
        vC = vw[sIdx:]
        vHc = np.abs(np.fft.rfft(vC, sNfft))
        sG = float(np.dot(vHc, vHl) / max(np.dot(vHc, vHc), 1e-300))   # LS level
        vHs = sG * vHc
        sMaxZ = float(np.max(np.abs(np.roots(vC))))
        lRows.append({
            'prunIdx': sIdx, 'len': len(vC), 'gain': sG,
            'w0_rel': float(abs(vC[0]) / np.max(np.abs(vC))),
            'max_abs_z': sMaxZ, 'min_phase': bool(sMaxZ < 1.0),
            'err_L2': float(np.linalg.norm(vHs - vHl) / np.linalg.norm(vHl)),
            'err_maxdB': float(np.max(np.abs(20 * np.log10(vHs[vDef] / vHl[vDef])))),
            'err_pass': float(np.max(np.abs(20 * np.log10(vHs[vPass] / vHl[vPass])))),
            'stop_dB': float(20 * np.log10(max(vHs[vStop].max(), 1e-30) / sMax))})

    sKey = {'L2': 'err_L2', 'maxdB': 'err_maxdB', 'pass': 'err_pass'}[sMetric]
    lFeas = [r for r in lRows
             if r['w0_rel'] >= sW0Rel and (r['min_phase'] or not bRequireMinPhase)]
    if lFeas:
        dBest = min(lFeas, key=lambda r: r[sKey])
    else:                                          # no candidate satisfies both
        lAlt = [r for r in lRows if r['min_phase']] if bRequireMinPhase else lRows
        dBest = min(lAlt, key=lambda r: r[sKey]) if lAlt else lRows[-1]

    sPrunIdx = dBest['prunIdx']
    vWcut = vw[sPrunIdx:] * dBest['gain']
    dInfo = dict(dBest)
    dInfo.update({'w0_rel_req': float(sW0Rel), 'metric': sMetric,
                  'n_feasible': len(lFeas),
                  'constraints_met': bool(lFeas and dBest in lFeas),
                  'stop_dB_linphase': float(20 * np.log10(vHl[vStop].max() / sMax)),
                  'peak_idx': int(np.argmax(np.abs(vw)))})
    if bTable:
        dInfo['table'] = lRows
    return vWcut, sPrunIdx, dInfo