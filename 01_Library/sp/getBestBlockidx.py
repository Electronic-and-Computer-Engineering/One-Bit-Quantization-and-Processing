import numpy as np

def getBestBlockidx(vX, sM=32, sH=8, sW=6, sEps=1e-12):
    """
    Return the start index sStart in {0,...,sH-1} that minimizes the
    mean edge-energy across all blocks of length sM with hop sH.

    Parameters
    ----------
    vX : 1D numpy array
        Input signal of length sN.
    sM : int
        Block length (default 32).
    sH : int
        Hop size (default 8).
    sW : int
        Edge width (samples at each block edge, default 6).
    sEps : float
        Small constant to avoid division by zero (default 1e-12).

    Returns
    -------
    sStart : int
        Optimal first-block start index (0..sH-1).
    vJ : 1D numpy array
        Mean edge-energy value for each start candidate (length sH).
    """
    sN = len(vX)
    vStartCandidates = np.arange(0, sH, dtype=int)
    vJ = np.zeros(sH, dtype=float)

    def EdgeEnergy(vBlk):
        """Edge-energy ratio for one block."""
        sNum = np.sum(vBlk[:sW]**2) + np.sum(vBlk[-sW:]**2)
        sDen = np.sum(vBlk**2) + sEps
        return sNum / sDen

    for s in vStartCandidates:
        vBlockStarts = np.arange(s, sN - sM + 1, sH, dtype=int)
        if vBlockStarts.size == 0:
            vJ[s] = np.inf
            continue
        vE = [EdgeEnergy(vX[sT:sT + sM]) for sT in vBlockStarts]
        vJ[s] = float(np.mean(vE))

    sStart = int(np.argmin(vJ))
    return sStart, vJ