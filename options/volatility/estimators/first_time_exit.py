"""
First Time Exit Volatility Estimator
"""
import itertools
import numpy as np
import pandas as pd
import datetime

import options.client as mClient
from options.typess.enums import Resolution, TickType, SecurityType
from options.typess.equity import Equity


def volatility_close_to_close(prices):
    """Jensen inequality => biased low; Contains vola shocks from overnight returns => biased high"""
    returns = np.log((prices / prices.shift(1)).dropna())
    obs_per_day = len(returns) / len(set(pd.to_datetime(returns.index).date))
    sigma_close_to_close = returns.std() * np.sqrt(252 * obs_per_day)
    print('Close to Close Minute Volatility - unadjusted open', sigma_close_to_close)
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
    start = datetime.date(2023, 7, 5)
    end = datetime.date(2023, 7, 18)
    start_time = datetime.time(9, 29)
    end_time = datetime.time(16, 00)
    resolution = Resolution.second
    resample_res = pd.Timedelta(seconds=1)
    sym = Equity('hpe')
    sym_str = str(sym)

    quotes = client.history([sym], start, end, resolution, TickType.quote, SecurityType.equity)
    mid = quotes[sym_str].loc[:, ['bid_close', 'ask_close']].mean(axis=1)

    mid = mid[(start_time <= mid.index.time) & (mid.index.time <= end_time)]  # no pre post trading

    if resample_res:
        mid = mid.resample(resample_res).last().dropna()
        print(mid)
    p0 = mid.iloc[0]
    delta = mid.iloc[0] / 1000
    upper = p0 + delta
    lower = p0 - delta
    times = []
    for ts, p in mid.items():
        if p > upper or p < lower:
            times.append(ts)
            p0 = p
            upper = p0 + delta
            lower = p0 - delta
    # remove opening jump
    time_deltas = []

    for dt, group in itertools.groupby(times, lambda ts: ts.date()):
        if dt.weekday() in [5, 6]:  # ignore weekends. not expected to show up in data anyway
            print(f'Warning: Encountered weekend: {dt}')
            continue
        ps = pd.Series(group)
        ps = ps[ps.apply(lambda dt: start_time <= dt.time() <= end_time)]
        time_delta = list(ps.diff().dropna().dt.total_seconds() / (60**2 * 24))
        time_deltas += time_delta

        # FYI intraday sigma
        print(f'{dt}; Weekday: {dt.weekday()}', (delta / np.sqrt(np.average(time_delta))) * (1 + 1 / (4*len(ps))))

    tao = np.average(time_deltas)
    sample_to_true_population = (1 + 1 / (4*len(time_deltas)))
    sigma = (delta / np.sqrt(tao)) * sample_to_true_population
    print('First Exit Volatility', sigma)

    volatility_close_to_close(mid)
