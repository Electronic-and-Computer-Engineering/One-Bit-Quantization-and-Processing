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

# Exportable API
__all__ = [
    'subplot', 'plot', 'stem', 'grid', 'title', 'xlabel', 'ylabel',
    'axis', 'hold', 'figure', 'legend', 'show'
]
