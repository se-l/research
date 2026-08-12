import webbrowser
import numpy as np
import pandas as pd
import plotly.graph_objs as go

if __name__ == '__main__':
    b = 0.2
    c = 0.01

    pfEquityPos_gammaOrder_util = []
    TargetMaxEquityPositionUSD = 4000
    priceUnderlying = 47.795
    targetAbsMaxEquityQuantity = TargetMaxEquityPositionUSD / priceUnderlying
    print(f'targetAbsMaxEquityQuantity: {targetAbsMaxEquityQuantity}')

    deltaPfTotal = -5
    v_optionDelta = [-45 + deltaPfTotal]#np.linspace(-500, 500, 100)
    v_orderDelta = [-203]#np.linspace(-100, 100, 20)

    for optionDelta in v_optionDelta:
        for orderDelta in v_orderDelta:
            whatIfOptionDelta = optionDelta + orderDelta

            if abs(optionDelta) < targetAbsMaxEquityQuantity and abs(whatIfOptionDelta) < targetAbsMaxEquityQuantity:
                print('asdfsadfsadfsdf')
                util = 0
            else:
                scaleByOrder = 1 if optionDelta == 0 else abs(orderDelta / optionDelta)

                absCurrentPfUtil = abs(b * optionDelta + c * optionDelta**2)
                absWhatIfUtil = abs(b * whatIfOptionDelta + c * whatIfOptionDelta**2)
                absPfUtil = abs(absWhatIfUtil - absCurrentPfUtil)
                util = -np.sign(optionDelta) * np.sign(orderDelta) * absPfUtil * scaleByOrder
            print(util)

    for pfEquityPos in v_pfEquityPos:
        for orderDelta in v_orderDelta:
            if abs(pfEquityPos) < targetAbsMaxEquityQuantity and (pfEquityPos + orderDelta) < targetAbsMaxEquityQuantity:
                util = 0
            else:
                scaleByOrder = 1 if pfEquityPos == 0 else abs(orderDelta / pfEquityPos)
                absPfUtil = abs(b * pfEquityPos + c * pfEquityPos**2)
                util = -np.sign(pfEquityPos) * np.sign(orderDelta) * absPfUtil * scaleByOrder

            pfEquityPos_gammaOrder_util.append((pfEquityPos, orderDelta, util))

    x = v_orderDelta
    y = v_pfEquityPos
    df = pd.DataFrame(pfEquityPos_gammaOrder_util, columns=['pfEquityPos', 'orderDelta', 'util'])
    z = df.set_index(['pfEquityPos', 'orderDelta']).unstack(-1)

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title=f"DeltaPf, Delta Order, Util Surface")
    fig.update_layout(scene=dict(
                    xaxis_title='Delta Order',
                    yaxis_title='Delta Pf',
                    zaxis_title='Util'))
    fig.write_html('figure.html')
    webbrowser.open('figure.html')
