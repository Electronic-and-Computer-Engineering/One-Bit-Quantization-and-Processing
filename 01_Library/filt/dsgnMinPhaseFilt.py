import numpy as np
import cvxpy as cp

from scipy.signal import freqz


def dsgnMinPhaseFilt(desired_impulse_response, filter_order=None, design_method='LS',
                     w0_target=None, weight='uniform', num_freq=400, n_fft=8192,
                     verbose=False):
    """
    Designs a MINIMUM-PHASE FIR filter approximating a desired magnitude response,
    with an explicit lower bound on the leading coefficient w0.

    Instead of designing h and converting to minimum phase afterwards (which
    destroys the LS optimality and fails numerically for deep stopbands), the
    optimization runs on the AUTOCORRELATION r of the filter:

        P(w) = |H(e^jw)|^2 = r_0 + 2 * sum_k r_k cos(k w)      (linear in r)

    Every non-negative P corresponds to exactly one minimum-phase FIR of the
    same length (spectral factorization), so minimum phase is structural here,
    not an afterthought.  The leading coefficient obeys

        w0 = exp( 1/(2*pi) * integral ln|H| dw ),

    and ln P is concave in r, so "w0 >= w0_target" is a convex constraint.

    Parameters
    ----------
    desired_impulse_response : ndarray   target IR (only its MAGNITUDE is used)
    filter_order             : int       order (length-1); default 80 % of input
    design_method            : 'LS' | 'MinMax'
    w0_target                : float     lower bound for w0 (None = unconstrained)
    weight                   : 'uniform' | 'relative'  (relative ~ dB-like fit)
    num_freq                 : int       frequency grid points on [0, pi)
    n_fft                    : int       FFT length for the spectral factorization

    Returns
    -------
    v_h_value : ndarray   minimum-phase filter, length filter_order+1
    d_info    : dict      diagnostics (w0, max|z|, tail energy, fit error)
    """
    v_h_ls_input = np.asarray(desired_impulse_response, dtype=float).ravel()
    s_M = int(len(v_h_ls_input) * 0.8) if filter_order is None else filter_order
    s_L = s_M + 1

    # ---- desired POWER spectrum on the design grid --------------------------
    v_wd = np.linspace(0, np.pi, num_freq, endpoint=False)
    v_Dd = freqz(v_h_ls_input, 1, v_wd)[1]
    v_Pd = np.abs(v_Dd) ** 2
    s_Scale = v_Pd.max()
    v_Pd = v_Pd / s_Scale                                    # max |H| = 1

    # ---- cosine matrix: P = m_A @ r ----------------------------------------
    v_k = np.arange(s_L)
    m_A = 2.0 * np.cos(np.outer(v_wd, v_k))
    m_A[:, 0] = 1.0

    # ---- convex program -----------------------------------------------------
    v_r = cp.Variable(s_L)
    v_P = m_A @ v_r
    v_wgt = 1.0 / (v_Pd + 1e-6) if weight == 'relative' else np.ones(num_freq)
    v_res = cp.multiply(v_wgt, v_P - v_Pd)

    if design_method == 'LS':
        objective = cp.Minimize(cp.norm(v_res, 2))
    elif design_method == 'MinMax':
        objective = cp.Minimize(cp.norm(v_res, 'inf'))
    else:
        raise ValueError("design_method must be 'LS' or 'MinMax'")

    constraints = [v_P >= 1e-9]                              # spectral factorability
    if w0_target is not None:
        # (1/pi) * int_0^pi ln P dw  >=  2 ln w0_target      (concave >= const)
        constraints.append(cp.sum(cp.log(v_P)) / num_freq >= 2.0 * np.log(w0_target))

    cp.Problem(objective, constraints).solve(verbose=verbose)
    if v_r.value is None:
        raise RuntimeError("solver failed (w0_target probably infeasible)")

    # ---- spectral factorization: P -> minimum-phase h ----------------------
    v_rFull = np.zeros(n_fft)
    v_rFull[:s_L] = v_r.value
    v_rFull[-(s_L - 1):] = v_r.value[:0:-1]
    v_Pfit = np.real(np.fft.fft(v_rFull))
    v_Pfit = np.maximum(v_Pfit, 1e-14 * v_Pfit.max())
    v_cep = np.real(np.fft.ifft(0.5 * np.log(v_Pfit)))
    v_fold = np.zeros(n_fft)
    v_fold[0] = v_cep[0]
    v_fold[1:n_fft // 2] = 2.0 * v_cep[1:n_fft // 2]
    v_fold[n_fft // 2] = v_cep[n_fft // 2]
    v_hFull = np.real(np.fft.ifft(np.exp(np.fft.fft(v_fold))))
    v_h_value = v_hFull[:s_L] * np.sqrt(s_Scale)

    v_z = np.roots(v_h_value)
    d_info = {'w0': float(v_h_value[0] / np.sqrt(s_Scale)),
              'w0_rel': float(abs(v_h_value[0]) / np.max(np.abs(v_h_value))),
              'max_abs_z': float(np.max(np.abs(v_z))),
              'tail_energy': float(np.sum(v_hFull[s_L:] ** 2) / np.sum(v_hFull ** 2)),
              'fit_rms': float(np.sqrt(np.mean((m_A @ v_r.value - v_Pd) ** 2)))}
    return v_h_value, d_info