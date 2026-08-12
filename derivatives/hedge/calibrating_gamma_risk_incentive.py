import numpy as np
import pandas as pd
import plotly.graph_objs as go

from shared.plotting import show

if __name__ == '__main__':
    b = 0.1
    c = 0.005

    v_gammaOrder = np.linspace(-100, 100, 20)
    v_gammaPfTotal = np.linspace(-500, 500, 100)

    gammaTargetZeroRange = (-75, 75)
    gammaPf_gammaOrder_util = []

    for pfGammaTotal0 in v_gammaPfTotal:
        for orderGamma in v_gammaOrder:
            pfGammaTotal1 = pfGammaTotal0 + orderGamma
            gammaRiskThresholdLow = gammaTargetZeroRange[0] - gammaTargetZeroRange[1]
            gammaRiskThresholdHigh = gammaTargetZeroRange[0] + gammaTargetZeroRange[1]
            if (pfGammaTotal0 > gammaRiskThresholdLow and pfGammaTotal1 > gammaRiskThresholdLow) and \
                (pfGammaTotal0 < gammaRiskThresholdHigh and pfGammaTotal1 < gammaRiskThresholdHigh):
                util = 0
            else:
                riskUtil0 = b * pfGammaTotal0 + c * pfGammaTotal0 ** 2
                riskUtil1 = b * pfGammaTotal1 + c * pfGammaTotal1 ** 2
                util = -(riskUtil1 - riskUtil0)

                # orderImpact = min(1, abs(gammaOrder / pfGammaTotal0))
                # absIncentive = orderImpact * abs(pfGammaTotal0 * b) + c * pfGammaTotal0 ** 2
                # util = -np.sign(pfGammaTotal0) * np.sign(gammaOrder) * absIncentive
            gammaPf_gammaOrder_util.append((pfGammaTotal0, orderGamma, util))

    x = v_gammaOrder
    y = v_gammaPfTotal
    df = pd.DataFrame(gammaPf_gammaOrder_util, columns=['gammaPfTotal', 'gammaOrder', 'util'])
    z = df.set_index(['gammaPfTotal', 'gammaOrder']).unstack(-1)

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title=f"Gamma Pf, Gamma Order, Util Surface")
    fig.update_layout(scene=dict(
        xaxis_title='X: Gamma Order',
        yaxis_title='Y: Gamma Pf',
        zaxis_title='Z: Util'))
    show(fig)
