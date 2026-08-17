import numpy as np

def iterSequQ(vx, mW, se_init):
    sxLen = len(vx)
    ve     = np.zeros(sxLen)
    vb     = np.zeros(sxLen)
    ve_hat = np.zeros(sxLen)

    vWx = mW @ vx   # (sxLen,) — einmal vorausberechnen, ändert sich nicht

    for sn in range(sxLen):
        # se_hat: nur untere Dreiecks-Anteil (k < sn)
        ve_hat[sn] = se_init + mW[sn, :sn] @ (vx[:sn] - vb[:sn])

        # Zeile sn von mW @ vb, mit vb[sn] = +1 bzw -1
        # mW[sn,:] @ vb = mW[sn,:sn] @ vb[:sn] + mW[sn,sn]*vb[sn] + mW[sn,sn+1:] @ vb[sn+1:]
        # vb[sn+1:] = 0 noch, also:
        sWb_base = mW[sn, :sn] @ vb[:sn]   # gemeinsamer Teil

        se_p = vWx[sn] - (sWb_base + mW[sn, sn] *  1)
        se_n = vWx[sn] - (sWb_base + mW[sn, sn] * -1)

        if se_p**2 <= se_n**2:
            vb[sn] =  1
            ve[sn] = se_p
        else:
            vb[sn] = -1
            ve[sn] = se_n

    return vb, ve, ve_hat