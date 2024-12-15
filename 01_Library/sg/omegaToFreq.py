# system packages
import numpy as np

def freq2digFc(sFc, sFs):
    """Convert frequency from Hz to digNormalized Cutoff"""
    sDigFc = sFc / (sFs / 2.0)
    return sDigFc

def digFc2Freq(sDigFc, sFs):
    """Convert frequency from digNormalized Cutoff to Hz."""
    sFc = sDigFc * (sFs / 2.0)
    return sFc