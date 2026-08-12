import numpy as np
import pandas as pd
import plotly.graph_objs as go
from shared.plotting import show


if __name__ == '__main__':
    b = 0.05
    c = 0.0005

    x = v_orderDelta = np.linspace(-500, 500, 50)
    y = v_deltaPfTotal0 = np.linspace(-500, 500, 50)

    thresholds = (-150, 150)
    v_utils = []

    for deltaPfTotal0 in v_deltaPfTotal0:
        for orderDelta in v_orderDelta:
            deltaPfTotal1 = orderDelta + deltaPfTotal0

            if (
                (deltaPfTotal0 > thresholds[0] and deltaPfTotal0 < thresholds[1]) and
                (deltaPfTotal1 > thresholds[0] and deltaPfTotal1 < thresholds[1])
            ):
                util = 0
            else:
                pfUtil0 = b * deltaPfTotal0 + c * deltaPfTotal0 ** 2
                pfUtil1 = b * deltaPfTotal1 + c * deltaPfTotal1 ** 2
                deltaPfUtil = pfUtil1 - pfUtil0
                util = -deltaPfUtil

            v_utils.append((orderDelta, deltaPfTotal0, util))

    df = pd.DataFrame(v_utils, columns=['orderDelta', 'deltaPfTotal0', 'util'])
    z = df.set_index(['orderDelta', 'deltaPfTotal0']).unstack(-1)

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    # fig.update_layout(title=f"")
    fig.update_layout(scene=dict(
        xaxis_title='X: Order Delta',
        yaxis_title='Y: Net Pf Delta',
        ))
    show(fig)
