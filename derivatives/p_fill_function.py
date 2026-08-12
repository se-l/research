import numpy as np
import math
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from shared.plotting import show

if __name__ == '__main__':
    """
    would want to quadratic fill. at x=1, at least 1
    """
    def f(x):
        return x ** 4

    x = [x/10 for x in range(10)]
    y = np.array([f(i) for i in x])

    # total_contract_volume.
    base_vol = 500

    secsPerDay = 6.5 * 60 * 60
    pFillVolume = 50

    # Volume would be massive if I'd always cross the spread
    # If I never cross the spread, volume behaves as specified.
    # The more volume, the more draws I have. Every second I draw...
    # Not crossing the spread I stick with p_Spread * volume/secondsPerDay
    # But when crossing the spread, I want to increase the chances significantly...
    # Maybe can also make it curve up...
    def fv(x, pFillVolume):
        exp = math.log(1.1, pFillVolume)
        return x**exp

    y2 = np.array([fv(i, pFillVolume) if i > 0 else 0 for i in x])

    fig = make_subplots(rows=3, cols=1)
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers+lines', name='objectives', marker=dict(size=4)), row=1, col=1)
    # fig.add_trace(go.Scatter(x=x, y=y2, mode='markers+lines', name='objectives', marker=dict(size=4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=y*pFillVolume/base_vol, mode='markers+lines', name='objectives', marker=dict(size=4)), row=3, col=1)
    show(fig)
