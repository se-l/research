import numpy as np
import pandas as pd
import plotly.graph_objs as go

from options.helper import plot_surface
from shared.plotting import show


def sd(x, slope_neg=-0.005, slope_pos=-0.002, zero_sd_util=30):
    if x >= zero_sd_util:
        return 2 / (1 + np.exp(slope_pos * (x - zero_sd_util))) - 1
    else:
        return 2 / (1 + np.exp(slope_neg * (x - zero_sd_util))) - 1


def D(uD1, uD0, b=.002):
    if uD1 > uD0:
        return np.nan
    dUdD = uD1 - uD0
    d0 = sd(uD0)
    return d0 + b * d0 * dUdD


if __name__ == '__main__':
    X = dU1 = np.arange(-200, 200, 20)[::-1]
    Y = dU0 = np.arange(0, 200, 20)[::-1]
    Z = [[D(u1, u0) for u1 in X] for u0 in Y]
    plot_surface(Z, X, Y)
    pd.DataFrame(Z, index=dU0, columns=dU1)

    uD1 = -200
    uD0 = 300
    uMin = 100
    # uMinPc = uMinPct(uLow, uHigh)
    dUdD = uD1 - uD0
    print(dUdD)

    fig = go.Figure()
    steps = list(range(-5000, 5000, 10))
    fig.add_trace(go.Scatter(x=steps, y=[sd(i) for i in steps], mode='lines'))
    show(fig)

    # fig = go.Figure()
    # a = 0
    # b = -0.5
    # c = 0.005
    # steps = list(range(-500, 500, 10))
    # fig.add_trace(go.Scatter(x=steps, y=[a + b * i + c * i ** 2 if i < -100 else 0 for i in steps], mode='lines'))
    # show(fig)
