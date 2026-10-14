import matplotlib.pyplot as plt


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