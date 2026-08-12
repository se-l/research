"""
First Time Exit Volatility Estimator
"""
import numpy as np
import pandas as pd
import datetime
import derivatives.client as mClient
from derivatives import Resolution, TickType, SecurityType
from derivatives import Equity


def volatility_close_to_close(prices):
    """Jensen inequality => biased low; Contains vola shocks from overnight returns => biased high"""
    returns = np.log((prices / prices.shift(1)).dropna())
    obs_per_day = len(returns) / len(set(pd.to_datetime(returns.index).date))
    sigma_close_to_close = returns.std() * np.sqrt(252 * obs_per_day)
    return sigma_close_to_close


if __name__ == '__main__':
    """
    Filtering away opening jump reduces close to close by ~ 5% to 2%
    
    1 Second
    First Exit Volatility 0.30937971304319994
    Close to Close Minute Volatility - unadjusted open 0.21004046543392668
    
    30 Seconds
    First Exit Volatility 0.2610727960341324
    Close to Close Minute Volatility - unadjusted open 0.2011196027803143
    
    1 Minute
    First Exit Volatility 0.24652185628643666
    Close to Close Minute Volatility - unadjusted open 0.20286471795754837
    
    90 Seconds
    First Exit Volatility 0.22551946202894677
    Close to Close Minute Volatility - unadjusted open 0.20162453580471937
    
    5 Min
    First Exit Volatility 0.17797868066520892
    Close to Close Minute Volatility - unadjusted open 0.20114180200365855
    """
    client = mClient.Client()
    pd.options.display.max_columns = 8
    start = datetime.date(2023, 6, 15)
    end = datetime.date(2023, 7, 25)
    start_time = datetime.time(9, 29)
    end_time = datetime.time(16, 00)
    resolution = Resolution.second
    resample_res = pd.Timedelta(seconds=1)
    grouping_res = pd.Timedelta(seconds=60**2 * 8)
    timespan = 5
    sym = Equity('hpe')
    sym_str = str(sym)

    quotes = client.history([sym], start, end, resolution, TickType.quote, SecurityType.equity)
    mid = quotes[sym_str].loc[:, ['bid_close', 'ask_close']].mean(axis=1)
    print('Total price points', len(mid))
    mid = mid[(start_time <= mid.index.time) & (mid.index.time <= end_time)]  # no pre post trading

    if resample_res:
        mid = mid.resample(resample_res).last().dropna()
        print(mid)

    # grouped_sigmas = []
    # for ts, ps in mid.groupby(pd.Grouper(freq=grouping_res)):
    #     if len(ps) < 2:
    #         continue
    #     if ts.weekday() in [5, 6]:  # ignore weekends. not expected to show up in data anyway
    #         print(f'Warning: Encountered weekend: {ts}')
    #         continue
    #     print(ts, len(ps), ps.index[0], ps.index[-1])
    #     grouped_sigmas.append(volatility_close_to_close(ps))
    #
    # grouped_sigmas = pd.Series(grouped_sigmas).dropna()
    # print(grouped_sigmas)
    # print('2nd Best Volatility', grouped_sigmas.mean())

    # for i in range(1, 10*60, 15):
    #     mid_resampled = mid.resample(pd.Timedelta(seconds=i)).last().dropna()
    #     sigma_close_to_close = volatility_close_to_close(mid_resampled.loc[mid_resampled.index[-1] - pd.Timedelta(days=timespan):])
    #     print(f'Close to Close Minute Volatility; Resample seconds {i}; samples: {len(mid_resampled)}', sigma_close_to_close)

    i = 300
    mid_resampled = mid.resample(pd.Timedelta(seconds=i)).last().dropna()
    sigma_close_to_close = volatility_close_to_close(mid_resampled.loc[mid_resampled.index[-1] - pd.Timedelta(days=timespan):])
    print(f'Close to Close Minute Volatility; Resample seconds {i}; samples: {len(mid_resampled)}', sigma_close_to_close)


