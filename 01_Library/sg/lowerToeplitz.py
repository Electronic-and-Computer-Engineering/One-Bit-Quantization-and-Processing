import numpy as np
import scipy.linalg as scLinAlg
import scipy.sparse as scSparse
 
 
def lowerToeplitz(vw, sN, bSparse=False):
    """
    Lower triangular Toeplitz (causal FIR convolution) matrix.
 
    Builds the sN x sN matrix mT with
 
        mT[i, j] = vw[i - j]   for  0 <= i - j < sL
                 = 0           otherwise
 
    i.e. the first column is vw zero-padded from sL up to sN, and every
    further column is that column shifted down by one. Multiplying gives the
    truncated causal convolution:  mT @ vx == np.convolve(vx, vw)[:sN].
 
    Input-Arguments:
        vw:       Filter impulse response, length sL.
        sN:       Matrix dimension. Must satisfy sN > sL; the gap between sL
                  and sN is filled with zeros.
        bSparse:  Return scipy.sparse CSR instead of a dense array. The matrix
                  has only sL non-zeros per column, so for large sN the dense
                  form wastes most of its memory (sN = 4096 dense float64 is
                  128 MB, sparse with sL = 101 is about 3 MB).
 
    Returning:
        mT:       (sN x sN) lower triangular Toeplitz matrix.
    """
    vw = np.asarray(vw).ravel()
    sL = vw.size
 
    if sN <= sL:
        raise ValueError(
            "sN must be strictly greater than the filter length sL "
            "(got sN=%d, sL=%d). Truncating the filter is almost never what "
            "you want here, so this is an error rather than a silent clip."
            % (sN, sL))
 
    if bSparse:
        # dia_matrix takes one diagonal per tap: tap k sits on offset -k.
        # Data rows are read with the *column* index, so no shifting is
        # needed for negative offsets beyond broadcasting the value.
        mData = np.repeat(vw[:, None], sN, axis=1)
        return scSparse.dia_matrix((mData, -np.arange(sL)),
                                   shape=(sN, sN)).tocsr()
 
    # Dense: give toeplitz an explicit first row of zeros (except the corner)
    # so it emits the lower triangular form directly. Building the symmetric
    # toeplitz and calling np.tril afterwards would allocate the same matrix
    # twice and discard half of it.
    vCol = np.zeros(sN, dtype=vw.dtype)
    vCol[:sL] = vw
    vRow = np.zeros(sN, dtype=vw.dtype)
    vRow[0] = vw[0]
    return scLinAlg.toeplitz(vCol, vRow)