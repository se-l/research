import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from scipy.stats import t
from datetime import date, datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from options.helper import get_density_for_bimodal_t_dist
from shared.paths import Paths
from shared.plotting import show
from shared.modules.logger import info

cal = USFederalHolidayCalendar()


def load():
    dfs = []
    for dir_, dirs, fns in os.walk(root):
        for fn in fns:
            df = pd.read_csv(os.path.join(dir_, fn))
            df['ReleaseDate'] = datetime.strptime(fn.split('.')[0], '%Y%m%d').date()
            dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def transform(df: pd.DataFrame):
    enrich_last_trading_session(df)
    df['Market Cap(M)'] = df['Market Cap(M)'].astype(str).str.replace(',', '').astype(float)
    df['%return'] = df['%Price Change**'].str.replace('--', '').str.replace('%', '').str.replace(',', '').apply(lambda x: float(x) if x else None)
    return df


def run():
    df = load()
    df = transform(df)
    if sys.platform == 'win32':
        summarize(df.copy())
    return df

def fit_bimodal_t_dist(returns: np.ndarray, p: np.ndarray):
    def model_func(x, dx, a, b, c):
        return (t.pdf(x + dx, a, b, c) + t.pdf(x - dx, a, b, c)) / 2

    popt, pcov = curve_fit(model_func, xdata=returns, ydata=p)
    return popt


def summarize(df: pd.DataFrame, min_bn_cap=10_000):
    """Many return% at exactly 0, appears to be an artifact."""
    mean_abs_return = df['%return'].dropna().drop(df.index[df['%return'] == 0]).abs().mean()
    print(f'Mean abs return: {mean_abs_return:.2f}% over {len(df)} announcements in {len(df["Symbol"].unique())} companies.')

    disk_tickers = [el.upper() for el in os.listdir(Paths.path_data.joinpath('option', 'usa', 'second'))]
    # Filter for relevant
    df_s = df[
        (df['Market Cap(M)'].fillna(0) > min_bn_cap)
        & (df['%return'].notna())
        & (df['Estimate'] != '--')
        & (df['Symbol'].isin(disk_tickers))
    ]  # & (df['%return'] != 0)]

    mean_return_capped = df_s['%return'].dropna().mean()
    median_return_capped = df_s['%return'].dropna().median()
    print(f'return for market caps over {min_bn_cap//1000}B: Mean: {mean_return_capped:.2f}%; Median: {median_return_capped}%; over {len(df_s)} announcements in {len(df_s["Symbol"].unique())} companies.')
    print(f'Companies: {df_s["Symbol"].unique()}')

    # plot histogram of %return and fit it
    nbins = 100

    df_s2 = df_s
    hist, bin_edges = np.histogram(df_s2['%return'], bins=nbins, density=True)
    p, returns = hist, bin_edges[:-1]
    popt = fit_bimodal_t_dist(returns, p)
    dx = popt[0]
    fit_params = popt[1:]

    # fit_params_orig = t.fit(df_s2['%return'])
    # fit_params = fit_params_orig
    # fit_params = fit_params_orig[0], fit_params_orig[1], fit_params_orig[2]#*1.5
    print(popt)
    x = np.linspace(bin_edges[0], bin_edges[-1], nbins)
    y = get_density_for_bimodal_t_dist(x, dx, *fit_params)
    info(f'get_density_for_bimodal_t_dist params: {dx}, {", ".join(map(str, fit_params))}')

    fig_hist = make_subplots(rows=3, cols=1, subplot_titles=(f'Student t params: {fit_params}', 'Histogram of %return', 'Desired dNLV curve'))
    fig_hist.add_trace(go.Scatter(y=hist, x=bin_edges, mode='lines', name='Histogram (density)'), row=1, col=1)
    fig_hist.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Fitted t-Distribution'), row=1, col=1)
    # fig_hist.add_trace(go.Scatter(x=x, y=t.pdf(x, *(5.167877640462322, 0.23684545664377854, 13.539506129642431)), mode='lines', name='Fitted t-Distribution (Scaled) Old'), row=1, col=1)
    fig_hist.add_trace(go.Histogram(x=df_s2['%return'], histnorm=None, nbinsx=100, name='Histogram (count)'), row=2, col=1)

    x_dnlv = np.linspace(-20, 20, 21)
    y_dlnv = get_density_for_bimodal_t_dist(x_dnlv, dx, *fit_params)
    fig_hist.add_trace(go.Scatter(x=x_dnlv, y=y_dlnv / np.max(y_dlnv), mode='lines', name='Fitted t-Distribution (normed)'), row=3, col=1)
    show(fig_hist)

    # from scipy.stats import skew, kurtosis
    #
    # # Calculate skewness and kurtosis of the data
    # data_skewness = skew(df_s2['%return'])
    # data_kurtosis = kurtosis(df_s2['%return'])
    #
    # # Calculate skewness and kurtosis of the fitted t-distribution
    # fitted_skewness = t.stats(df_estimated, moments='s')
    # fitted_kurtosis = t.stats(df_estimated, moments='k')


def enrich_last_trading_session(df: pd.DataFrame):
    window = 14
    holidays = [d.date() for d in cal.holidays(start=df['ReleaseDate'].min() - timedelta(days=window), end=df['ReleaseDate'].max()).to_pydatetime()]

    df['last_trading_session_date'] = None
    for ix, ps in df.iterrows():
        if ps['Time'] == 'bmo':
            for i in range(1, window):
                dt = ps['ReleaseDate'] - pd.Timedelta(days=i)
                if dt.weekday() < 5 and dt not in holidays:
                    df.loc[ix, 'last_trading_session_date'] = dt
                    break
        else:
            df.loc[ix, 'last_trading_session_date'] = ps['ReleaseDate']
    return df


def store_json(df_in):
    df = df_in.copy()
    df['Date'] = df['last_trading_session_date'].apply(lambda x: x.isoformat() if isinstance(x, date) else None)
    df['ReleaseDate'] = df['ReleaseDate'].apply(lambda x: x.isoformat() if isinstance(x, date) else None)
    df = df.sort_values(['Symbol', 'Date'])
    df.drop(df.index[df['Date'].isna()], axis=0, inplace=True)
    df.drop(df.index[df['Symbol'].isna()], axis=0, inplace=True)
    string = df[['Symbol', 'Date', 'Time', 'ReleaseDate', 'Company']].to_json(index=False, orient='records')

    with open(Paths.path_earnings, 'w') as f:
        f.write(string)


def next_x_opportunities(start: date, end: date, min_bn_cap=10_000):
    df = load()
    df = transform(df)

    # filter
    df_s = df[
        (df['Market Cap(M)'].fillna(0) > min_bn_cap)
        & (df['Estimate'] != '--')
        & (df['ReleaseDate'] >= start)
        & (df['ReleaseDate'] <= end)
        & (df['Time'].isin(('amc', 'bmo')))
        ]

    pd.options.display.max_columns = 10
    print(df_s[['last_trading_session_date', 'Symbol', 'ReleaseDate', 'Time', 'Company', 'Market Cap(M)']].sort_values(['last_trading_session_date', 'Symbol']))


if __name__ == '__main__':
    root = Paths.path_trade.joinpath('EarningsAnnouncements')
    next_x_opportunities(date.today(), date.today() + timedelta(days=8))
    df = run()
    store_json(df)
