import numpy as np
from tabu import TabuSampler

# Reuse one sampler across all blocks: constructing it per call costs more
# than the search itself for M in the tens.
_SAMPLER = TabuSampler()

def OptBlockTabu(vx, mW, vCe, sNumReads = 3, sTimeout = 5, sTenure = None):
    """
    Drop-in replacement for obq.OptBlock using the D-Wave MST2 tabu search
    (pip install dwave-tabu).

    Minimizes  || mW @ (vx - vb) + vCe ||_2^2  over vb in {-1,+1}^M.

    Args:
        vx:         Input vector (length M).
        mW:         Filter matrix mW_ext, ((nu+1)*M x M) or (M x M).
        vCe:        Accumulated error vector vCe_ext (length (nu+1)*M).
        sNumReads:  Independent restarts of the tabu search.
        sTimeout:   Milliseconds per read.
        sTenure:    Tabu tenure; None lets the solver pick.

    Returns:
        vb_out:     Quantized one-bit vector.
        ve:         Error vector  mW @ (vx - vb) + vCe.
        outTxt:     Status string (same slot as the Gurobi version).
    """
    sM = mW.shape[1]

    # ---- Map the row-wise cost onto an Ising model -----------------------
    # sample_ising accepts ONLY per-variable fields h_i and pairwise
    # couplings J_ij. The row form sum_i ( sum_j W_ij d_j + c_i )^2 has no
    # explicit pairwise terms, so the square must be expanded first. That
    # expansion is exactly the Gram matrix:
    #
    #   sum_i ( sum_j W_ij d_j + c_i )^2
    #     = d^T G d + 2 g^T d + const,     G = W^T W,  g = W^T c
    #
    # Substituting d = vx - vb and dropping vb-independent terms:
    #
    #   f(vb) = vb^T G vb - 2 (G vx + g)^T vb + const'
    #
    # G is symmetric, so the off-diagonal pair (i,j) appears twice -> J_ij = 2 G_ij.
    # The diagonal contributes G_ii * vb_i^2 = G_ii, i.e. a constant -> dropped.
    mG  = mW.T @ mW          # M x M, Gram matrix
    vg  = mW.T @ vCe         # M,     projected accumulated error
    vGx = mG @ vx            # M,     reused in the linear term

    # Linear fields: h_i = -2*(G vx)_i - 2*g_i
    dH = {i: float(-2.0 * vGx[i] - 2.0 * vg[i]) for i in range(sM)}

    # Couplings: J_ij = 2*G_ij for i < j (upper triangle only)
    dJ = {(i, j): float(2.0 * mG[i, j])
          for i in range(sM) for j in range(i + 1, sM)}

    # ---- Run the tabu search --------------------------------------------
    # Variables are already spin-valued {-1,+1}, so sample_ising is the
    # native interface; sample_qubo would require a 0/1 remapping.
    dKwargs = dict(num_reads = sNumReads, timeout = sTimeout)
    if sTenure is not None:
        dKwargs['tenure'] = sTenure

    res = _SAMPLER.sample_ising(dH, dJ, **dKwargs)

    # res.first is the lowest-energy sample found; .sample is {index: +-1}
    dBest  = res.first.sample
    vb_out = np.array([dBest[i] for i in range(sM)], dtype=float)

    # ---- Error vector ----------------------------------------------------
    # Recomputed from the original row form, so the caller's error
    # bookkeeping (veBlock[:sM] = current-block error) stays valid.
    # Note: res.first.energy omits the constant terms dropped above and is
    # therefore NOT the true cost -- it is only useful for status output.
    ve = mW @ (vx - vb_out) + vCe

    outTxt = "tabu (E=%.4g)" % float(res.first.energy)

    return vb_out, ve, outTxt