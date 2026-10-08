import numpy as np
import sg


def swapCloseBits(vx, mW, vC_hat, vb_hat, ve, sSwapBitLen=4):
    """
    Sucht die Bits mit knappster Einzelentscheidung und loest ueber genau
    diese exakt.

    Schritt 1: fuer jedes Bit beide Vorzeichen auswerten, nichts aendern.
               Die Differenz |E(+1) - E(-1)| sagt, wie fest das Bit sitzt.
    Schritt 2: die sSwapBitLen knappsten Indizes nehmen und alle 2^n
               Vorzeichenmuster ueber diese Stellen durchrechnen.

    Die Indizes liegen verstreut ueber den Block - eine Nachbarschaft, die
    zusammenhaengende Fenster nicht abdecken.

    Der Fehler wird durchgehend ueber alle Zeilen von mW bewertet, also
    ve = mW (vx - vb) + vC_hat. Der Ist-Zustand ist unter den Kandidaten,
    E kann damit nicht steigen.

    Rueckgabe: vb_hat, ve, sE, vIdx
    """
    sBSize = len(vx)
    sN     = min(sSwapBitLen, sBSize)

    # ---- 1) Einzelbit-Knappheit ----------------------------------------
    vGap = np.empty(sBSize)
    for i in range(sBSize):
        vCol     = mW[:, i]
        veRest   = ve + vCol*vb_hat[i]
        vEp, vEm = veRest - vCol, veRest + vCol
        sEp, sEm = vEp @ vEp, vEm @ vEm
        vGap[i]  = abs(sEp - sEm)

    # ---- 2) exakt ueber die sN knappsten Bits ---------------------------
    vIdx = np.sort(np.argsort(vGap)[:sN])
    mC   = sg.binaryComb(sN).T * 2.0 - 1.0
    mWs  = mW[:, vIdx]

    veRest = ve + mWs @ vb_hat[vIdx]
    mERest = veRest[:, None] - mWs @ mC
    vMins  = np.sum(mERest**2, axis=0)

    j = int(np.argmin(vMins))
    vb_hat[vIdx] = mC[:, j]
    ve[:]        = mERest[:, j]

    return vb_hat, ve, float(vMins[j]), vIdx

