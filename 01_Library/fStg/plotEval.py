import numpy as np

from PyQt5.QtCore    import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                             QVBoxLayout, QLabel)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

import sa

# --- x axis: 0 .. pi in steps of pi/10
vXTicks  = np.linspace(0, np.pi, 11)
vXLabels = ([r'$0$'] +
            [rf'$\frac{{{k}\pi}}{{10}}$' for k in range(1, 10)] +
            [r'$\pi$'])

dictBox = dict(boxstyle='round', facecolor='lightyellow',
               edgecolor='black', linewidth=1.2, alpha=0.95)

# --- publication style: no titles, label boxes inside the axes
sFsTick  = 15
sFsLabel = 17
sFsBox   = 16


def magDb(vSig, sN, sNorm):
    """Magnitude spectrum in dB, normalized to sNorm, positive half only."""
    return 20 * sa.safelog10(np.abs(np.fft.fft(vSig, sN))[:sN // 2] / sNorm)


# =============================================================================
# OVERVIEW
# =============================================================================
def barRecon(ax, dictRes, vMethods, strRecon, strTitle):
    """One bar plot: best method at delta 0, all others below."""

    vMean  = np.array([dictRes[f"{strRecon}_{m}"][:, 1].mean() for m in vMethods])

    # best method first, rest by descending mean SER
    vOrder   = np.argsort(vMean)[::-1]
    vLabels  = [vMethods[i] for i in vOrder]
    vMean    = vMean[vOrder]
    vDelta   = vMean - vMean[0]

    vIdx = np.arange(len(vLabels))
    ax.bar(vIdx, vDelta, 0.5, color='0.4')
    ax.axhline(0, color='black', linewidth=1.0)

    for i in vIdx:
        strTxt = f"{vMean[i]:.2f} dB"
        if i > 0:
            strTxt += f"\n$\\Delta$ {vDelta[i]:.2f} dB"
        ax.text(i, 0, strTxt, ha='center', va='bottom', fontsize=11, bbox=dictBox)

    sDown = min(vDelta.min(), -1.0)
    ax.set_xticks(vIdx)
    ax.set_xticklabels(vLabels, fontsize=12)
    ax.set_ylabel(r'$\Delta$ mean SER (dB)', fontsize=12)
    ax.set_title(strTitle, fontsize=12)
    ax.set_ylim([sDown * 1.3, abs(sDown) * 0.55])
    ax.minorticks_on()
    ax.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')


def figOverview(dictRes, sCaseFile):
    """Mean SER per method for both reconstructions -- REF excluded."""

    vMethods = [k[len("ideal_"):] for k in dictRes
                if k.startswith("ideal_") and not k.endswith("_REF")]

    fig = Figure(figsize=(9, 8))
    vAx = fig.subplots(2, 1)

    barRecon(vAx[0], dictRes, vMethods, "ideal", "ideal reconstruction")
    barRecon(vAx[1], dictRes, vMethods, "real",  "non-ideal reconstruction")

    fig.suptitle(sCaseFile)
    fig.tight_layout()
    return fig


# =============================================================================
# METHOD TAB
# =============================================================================
class MethodTab(QWidget):
    """Signal and noise spectrum of one method, one realization at a time.

    The figure carries no titles -- the panel labels sit inside the axes, so
    the figure can be saved straight from the toolbar and used as is. The
    realization counter lives in a QLabel below the canvas and therefore does
    not end up in the saved file.
    """

    def __init__(self, strMethod, mx, mb, dictRes, vw, sYLim):
        super().__init__()
        self.strMethod = strMethod
        self.mx        = mx
        self.mb        = mb
        self.dictRes   = dictRes
        self.sN, self.sBatchSize = mx.shape
        self.sYLim     = sYLim
        self.idx       = 0

        # shaping filter response, normalized to its own maximum
        vWfft     = np.abs(np.fft.fft(vw, self.sN))[:self.sN // 2]
        self.vWdB = 20 * sa.safelog10(vWfft / np.max(vWfft))

        self.vOmega = np.linspace(0, np.pi, self.sN // 2, endpoint=False)

        self.fig    = Figure(figsize=(9, 7))
        self.vAx    = self.fig.subplots(2, 1)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(Qt.NoFocus)       # keys go to the window
        self.status = QLabel()                       # outside the figure

        lay = QVBoxLayout(self)
        lay.addWidget(NavToolbar(self.canvas, self))  # save / zoom / pan
        lay.addWidget(self.canvas)
        lay.addWidget(self.status)

        self.plot()

    def step(self, sStep):
        self.goTo(self.idx + sStep)

    def goTo(self, sIdx):
        self.idx = int(np.clip(sIdx, 0, self.sBatchSize - 1))
        self.plot()

    def plot(self):
        vx = self.mx[:, self.idx]
        vb = self.mb[:, self.idx]

        sNorm = np.max(np.abs(np.fft.fft(vx, self.sN)))

        sSerIdeal = self.dictRes[f"ideal_{self.strMethod}"][self.idx, 1]
        sSerReal  = self.dictRes[f"real_{self.strMethod}"][self.idx, 1]

        for ax in self.vAx:
            ax.clear()

        # --- a) signal spectrum
        self.vAx[0].plot(self.vOmega, magDb(vb, self.sN, sNorm),
                         color='black', linewidth=1.0)

        # --- b) noise spectrum plus shaping filter
        self.vAx[1].plot(self.vOmega, magDb(vx - vb, self.sN, sNorm),
                         color='black', linewidth=1.0)
        self.vAx[1].plot(self.vOmega, self.vWdB,
                         color='cornflowerblue', linestyle='--', linewidth=3.0)

        self.vAx[1].set_xlabel('Normalized Frequency (radians/sample)',
                               fontsize=sFsLabel)

        # --- panel labels inside the axes, lower right
        for ax, strPanel, sSer in ((self.vAx[0], 'a', sSerIdeal),
                                   (self.vAx[1], 'b', sSerReal)):
            ax.text(0.98, 0.05,
                    f"{strPanel}) {self.strMethod}, SER: {sSer:.2f} dB",
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=sFsBox, bbox=dictBox)

            ax.set_ylabel('Magnitude (dB)', fontsize=sFsLabel)
            ax.set_xlim([0, np.pi])
            ax.set_ylim(self.sYLim)
            ax.set_xticks(vXTicks)
            ax.set_xticklabels(vXLabels, fontsize=sFsTick)
            ax.tick_params(axis='y', labelsize=sFsTick)
            ax.minorticks_on()
            ax.grid(True, which='both', linestyle='--', linewidth=0.3,
                    color='gray')

        self.status.setText(
            f"realization {self.idx + 1} / {self.sBatchSize}     "
            f"SER ideal {sSerIdeal:.2f} dB     non-ideal {sSerReal:.2f} dB")

        self.fig.tight_layout()
        self.canvas.draw()


# =============================================================================
# WINDOW
# =============================================================================
class EvalWindow(QMainWindow):
    """Tab container.

    Navigation through the realizations, identical on Windows and macOS:

        right / down    +1        left / up      -1
        .               +10       ,              -10
        end             last      home           first

    Arrow keys and the two punctuation keys are plain Qt key codes, so they
    need no modifier on either platform. Home and End require fn + arrow on
    Mac laptops, hence they are the secondary binding, not the primary one.
    """

    strKeys = ("navigation:   \u2190 \u2192 \u2191 \u2193  =  \u00b11        "
               ",  .  =  \u00b110        home / end  =  first / last")

    def __init__(self, mx, dictMb, dictRes, vw, sCaseFile, sYLim):
        super().__init__()
        self.setWindowTitle(sCaseFile)

        self.tabs = QTabWidget()

        tabOv = QWidget()
        layOv = QVBoxLayout(tabOv)
        canOv = FigureCanvas(figOverview(dictRes, sCaseFile))
        canOv.setFocusPolicy(Qt.NoFocus)
        layOv.addWidget(NavToolbar(canOv, tabOv))
        layOv.addWidget(canOv)
        self.tabs.addTab(tabOv, "Overview")

        self.vTabs = []
        for strMethod in sorted(dictMb):
            tab = MethodTab(strMethod, mx, dictMb[strMethod], dictRes, vw, sYLim)
            self.vTabs.append(tab)
            self.tabs.addTab(tab, strMethod)

        self.setCentralWidget(self.tabs)

        # QTabWidget consumes left/right to switch tabs -- take the focus away
        # from it and from the tab bar so every key reaches keyPressEvent.
        self.tabs.setFocusPolicy(Qt.NoFocus)
        self.tabs.tabBar().setFocusPolicy(Qt.NoFocus)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        self.statusBar().showMessage(self.strKeys)
        self.resize(1000, 900)

    def keyPressEvent(self, event):
        dictStep = {Qt.Key_Right: +1, Qt.Key_Up:    +1,
                    Qt.Key_Left:  -1, Qt.Key_Down:  -1,
                    Qt.Key_Period: +10, Qt.Key_Comma: -10}
        if event.key() in dictStep:
            for tab in self.vTabs:
                tab.step(dictStep[event.key()])
        elif event.key() in (Qt.Key_Home, Qt.Key_End):
            for tab in self.vTabs:
                tab.goTo(0 if event.key() == Qt.Key_Home else tab.sBatchSize - 1)


# =============================================================================
# ENTRY POINT
# =============================================================================
def plotEval(mx, dictMb, dictRes, vw, sCaseFile, sYLim=(-60, 5)):
    """Open the evaluation window. Blocks until it is closed."""
    app = QApplication.instance() or QApplication([])
    win = EvalWindow(mx, dictMb, dictRes, vw, sCaseFile, sYLim)
    win.show()
    app.exec_()