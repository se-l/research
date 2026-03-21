'''
- Plot put call surfaces with the current methodology on release date. Smoothened and ~raw IVs. As a visual reference / baseline.
- Each of these has CalibrationItems. Need to extract the early exercise premium. Presumably also a surface across tenor/strike.
- How to extract it: For each calibrationItem, calculate the American IV.
Check where it causes a difference. Puts and long-dated options...

Basically saying - replace Euro option IV calc with American everywhere...
'''

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def plot_european_surfaces():
    """with continuous dividend yields"""
    pass

def plot_american_surfaces():
    """with continuous dividend yields"""
    pass

def plot_american_surfaces_dividends():
    """with discrete non-zero dividend"""
    pass

def run():
    pass

if __name__ == '__main__':
    run()