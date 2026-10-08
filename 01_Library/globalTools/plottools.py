import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# MATLAB-like Plot Utilities (plottools)
# ----------------------------

def subplot(nrows, ncols, index):
    """MATLAB-like subplot selection."""
    plt.subplot(nrows, ncols, index)

def plot(x, y=None, *args, **kwargs):
    """Plot x or (x,y) like MATLAB with format string and kwargs."""
    if y is None:
        plt.plot(x, *args, **kwargs)
    else:
        plt.plot(x, y, *args, **kwargs)

def stem(x, y=None, **kwargs):
    """Discrete-time plot."""
    if y is None:
        y = x
        x = np.arange(len(y))
    markerline, stemlines, baseline = plt.stem(x, y, **kwargs)
    plt.setp(baseline, 'color', 'k', 'linewidth', 0.5)

def grid(minor=False):
    """Enable major (and optionally minor) grid."""
    plt.grid(True, which='major', linestyle='-', linewidth=0.5)
    if minor:
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle=':', linewidth=0.3)

def title(txt):
    plt.title(txt)

def xlabel(txt):
    plt.xlabel(txt)

def ylabel(txt):
    plt.ylabel(txt)

def axis(mode='tight'):
    """Axis scaling: 'tight', 'equal', etc."""
    plt.axis(mode)

def hold(on=True):
    """MATLAB-style hold (pseudo-effect)."""
    # Only controls interactive mode; 'hold' behavior is implicit in matplotlib unless new figure() is called
    plt.ion() if on else plt.ioff()

def figure(num=None):
    """Create new figure, optionally with number."""
    if num is None:
        plt.figure()
    else:
        plt.figure(num=num)

def legend(*args, **kwargs):
    plt.legend(*args, **kwargs)

def show():
    plt.show()
    

def plotInit(bPlot):
    """Live plot of the block error over the window steps of the VND cascade."""
    if not bPlot:
        return None
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], '.-')
    ax.set_yscale('log')
    ax.set_xlabel('window step')
    ax.set_ylabel(r'$E(x,\hat b)$')
    ax.grid(True)
    return {'fig': fig, 'ax': ax, 'line': line, 'lE': []}


def plotDepth(dPlot, sDepth=None):
    """Blue marker where the cascade moves to the next neighbourhood size."""
    if dPlot is None:
        return
    sX = max(len(dPlot['lE']) - 1, 0)
    dPlot['ax'].axvline(sX, color='b', lw=0.8, alpha=0.6)
    if sDepth is not None:
        dPlot['ax'].annotate(f'd={sDepth}', xy=(sX, 1.0),
                             xycoords=('data', 'axes fraction'),
                             xytext=(2, -10), textcoords='offset points',
                             color='b', fontsize=8)


def plotStart(dPlot, sStart):
    """Green marker at the beginning of a new start vector (multistart)."""
    if dPlot is None or sStart == 0:
        return
    sX = max(len(dPlot['lE']) - 1, 0)
    dPlot['ax'].axvline(sX, color='g', lw=1.2)
    dPlot['ax'].annotate(f'#{sStart}', xy=(sX, 0.02),
                         xycoords=('data', 'axes fraction'),
                         xytext=(2, 0), textcoords='offset points',
                         color='g', fontsize=8)


def plotKick(dPlot):
    """Red marker where the solution was perturbed."""
    if dPlot is None:
        return
    dPlot['ax'].axvline(max(len(dPlot['lE']) - 1, 0),
                        color='r', lw=0.8, alpha=0.6)


def plotAdd(dPlot, sE):
    """Append one point and redraw."""
    if dPlot is None:
        return
    dPlot['lE'].append(sE)
    dPlot['line'].set_data(range(len(dPlot['lE'])), dPlot['lE'])
    dPlot['ax'].relim()
    dPlot['ax'].autoscale_view()
    dPlot['fig'].canvas.draw_idle()
    plt.pause(0.001)


def plotClose(dPlot, sHold=0.0):
    if dPlot is None:
        return
    if sHold > 0:
        plt.pause(sHold)
    plt.close(dPlot['fig'])
    plt.ioff()


# =========================================================================
#  Signal level: E_glob = ||W (x - b)||^2 over the block sweeps
# =========================================================================

def plotGlobInit(bPlot):
    if not bPlot:
        return None
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], '.-', lw=1)
    ax.set_yscale('log')
    ax.set_xlabel('block step')
    ax.set_ylabel(r'$E_{glob}$')
    ax.grid(True)
    return {'fig': fig, 'ax': ax, 'line': line, 'lE': []}


def plotGlobIterMark(dPlot, sIterIdx):
    """Blue marker at the start of an outer iteration."""
    if dPlot is None:
        return
    sX = max(len(dPlot['lE']) - 1, 0)
    dPlot['ax'].axvline(sX, color='b', lw=0.8, alpha=0.7)
    dPlot['ax'].annotate(f'it{sIterIdx}', xy=(sX, 1.0),
                         xycoords=('data', 'axes fraction'),
                         xytext=(2, -10), textcoords='offset points',
                         color='b', fontsize=8)


plotGlobAdd   = plotAdd          # same mechanics, different figure dict
plotGlobClose = plotClose