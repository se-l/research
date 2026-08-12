import os
import sys
from shared.plotting import show
from plotly.subplots import make_subplots
import numpy as np
from plotly import graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import norm

if __name__ == '__main__':
    dir_ = r'C:\repos\quantconnect\Lean\Launcher\bin\Analytics\backtesting\240520100616-EarningsAlgorithm-bt.dev.panw.2024-02-20.2024-02-21\Kalman\PANW'
    dfs = []
    for root, dirs, files in os.walk(dir_):
        for file in files:
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(root, file))
                except Exception as e:
                    print(f'Error reading {file}: {e}')
                    continue
                dfs.append(df)
    df = pd.concat(dfs)
    df = df[df['Expiry'] == '2024-02-23']
    df['std'] = (df['SSR'] / df['NUpdates'])**0.5
    df['std'].describe()
    # plotly the std as scatter plot
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['std'], mode='markers', name=f'STD', marker=dict(size=2)), row=1, col=1)
    show(fig)

    def bound(alpha):
        return df['IVMean'] + df['std'] * norm.ppf(alpha)
    print(max(abs(bound(0.8) - df['KalmanAskIV'])))
    print(max(abs(bound(0.2) - df['KalmanBidIV'])))
    print((abs(bound(0.9) - df['KalmanAskIV'])).mean())
    print((abs((bound(0.95) - bound(0.8)) / bound(0.8))).mean())

    # What % of the spread is the kalman bid price?
    df['Spread'] = df['AskPrice'] - df['BidPrice']
    df['KalmanBidMBidPrice'] = df['KalmanBidPrice'] - df['BidPrice']
    df['KalmanAskMAskPrice'] = df['KalmanAskPrice'] - df['AskPrice']

    df['KalmanBidPcOfSpread'] = (df['KalmanBidPrice'] - df['BidPrice']) / df['Spread']
    df['KalmanAskPcOfSpread'] = (df['AskPrice'] - df['KalmanBidPrice']) / df['Spread']
    print(df[['KalmanBidPcOfSpread', 'KalmanAskPcOfSpread']])

    ax = df['KalmanBidPcOfSpread'].hist(bins=100)
    ax.set_title('Kalman Bid Price as % of Spread')
    ax.set_xlabel('% of Spread')
    ax.figure.show()

    ax = df['KalmanAskPcOfSpread'].hist(bins=100)
    ax.set_title('Kalman Ask Price as % of Spread')
    ax.set_xlabel('% of Spread')
    ax.figure.show()

    ax = df['KalmanBidMBidPrice'].hist(bins=100)
    ax.set_title('KalmanBidMBidPrice')
    ax.set_xlabel('Kalman Bid Price - Best Bid')
    ax.figure.show()

    ax = df['KalmanAskMAskPrice'].hist(bins=100)
    ax.set_title('KalmanAskMAskPrice')
    ax.set_xlabel('Kalman Ask Price - Best Ask')
    ax.figure.show()

    print((df['KalmanBidIV'] < df['MidIV']).sum() / len(df))
    print((df['KalmanAskIV'] > df['MidIV']).sum() / len(df))

    ax = (df['KalmanBidIV'] / df['MidIV']).hist(bins=100)
    ax.set_title('Kalman Bid IV as % of Mid IV')
    ax.set_xlabel('% of MidIV')
    ax.figure.show()

    ax2 = (df['KalmanAskIV'] / df['MidIV']).hist(bins=100)
    ax2.set_title('Kalman Ask IV as % of Mid IV')
    ax2.set_xlabel('% of MidIV')
    ax2.figure.show()
