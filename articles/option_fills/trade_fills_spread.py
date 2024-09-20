import numpy as np
import pandas as pd
import QuantLib as ql
import plotly.graph_objs as go

from functools import partial
from typing import Dict
from itertools import chain
from datetime import timedelta, datetime, time, date
from plotly.subplots import make_subplots
from options.client import Client
from options.helper import quotes2multi_index_df, aewma, get_dividend_yield, df2iv, get_v_tenor_from_index
from options.typess.enums import TickType, Resolution, SecurityType, OptionRight
from options.typess.equity import Equity
from options.typess.option_contract import OptionContract
from shared.constants import DiscountRateMarket, EarningsPreSessionDates
from shared.modules.logger import logger
from shared.plotting import plot_ps_trace, show
from markdown import markdown as md
from pykalman import KalmanFilter


def load_data(start: date, end: date, equity: Equity, take: int, resolution: Resolution = Resolution.second):
    client = Client()

    eq_trades = client.history([equity], start, end, resolution, TickType.trade, SecurityType.equity)
    ps_spot = eq_trades[str(equity)]['close']

    contracts = list(chain(*client.get_contracts(equity, start=start, end=end).values()))
    trades = client.history(contracts, start, end, resolution=resolution, tick_type=TickType.trade, security_type=SecurityType.option)
    trades = {OptionContract.from_contract_nm(k): v for k, v in trades.items()}

    quotes = client.history(contracts, start, end, resolution, TickType.quote, SecurityType.option)
    quotes: Dict[OptionContract, pd.DataFrame] = {OptionContract.from_contract_nm(k): v for k, v in quotes.items()}

    dct_c_df = {}
    for c, dfq in quotes.items():
        if c not in trades:
            continue
        dft = trades[c]
        df = dfq.merge(dft.rename(columns={'close': 'fill_price'}), how='outer', right_index=True, left_index=True)
        df = df.merge(ps_spot.rename('spot'), how='outer', right_index=True, left_index=True)
        for col in ['spot', 'bid_close', 'ask_close']:
            df[col].ffill(inplace=True)
        ix_keep = df.index[df['fill_price'].notna()]
        df = df.loc[ix_keep]
        df['option_contract'] = c
        df.index.rename('index', inplace=True)
        df.drop(['bid_open', 'bid_high', 'bid_low', 'ask_open', 'ask_high', 'ask_low', 'open', 'high', 'low', 'bid_size', 'ask_size'], axis=1, inplace=True)
        dct_c_df[c] = df

    df = quotes2multi_index_df(dct_c_df)

    # Dropping expired derivatives
    ix_drop = []
    for ts, sub_df in df.groupby(level='ts'):
        for expiry, sub_sub_df in sub_df.groupby(level='expiry'):
            if (expiry < ts.date()):
                ix_drop.extend(sub_sub_df.index)
    df.drop(ix_drop, inplace=True)
    return df


def enrich(equity: Equity, df: pd.DataFrame):
    # Deriving implied volatilities
    rate = DiscountRateMarket
    dividend_yield = dividend = get_dividend_yield(equity)
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    day_count = ql.Actual365Fixed()
    df['tenor'] = get_v_tenor_from_index(df)
    df['bid_iv'] = df2iv(df, price_col_nm='bid_close', rate=rate, dividend_yield=dividend_yield)
    df['ask_iv'] = df2iv(df, price_col_nm='ask_close', rate=rate, dividend_yield=dividend_yield)
    df['fill_price_iv'] = df2iv(df, price_col_nm='fill_price', rate=rate, dividend_yield=dividend_yield)
    # df['bid_iv'] = df.apply(partial(ps2iv, price_col='bid_close', calendar=calendar, day_count=day_count, rate=rate, dividend=dividend), axis=1)
    # df['ask_iv'] = df.apply(partial(ps2iv, price_col='ask_close', calendar=calendar, day_count=day_count, rate=rate, dividend=dividend), axis=1)
    # df['fill_price_iv'] = df.apply(partial(ps2iv, price_col='fill_price', calendar=calendar, day_count=day_count, rate=rate, dividend=dividend), axis=1)

    df['bid_iv_aewma'] = aewma(df['bid_iv'], 0.005, 0)  # no no. aewma follows delta/moneyness, not contracts
    df['ask_iv_aewma'] = aewma(df['ask_iv'], 0.005, 0)

    df['bid_delta'] = None
    df['ask_delta'] = None
    for right, right_df in df.groupby(level='right'):
        from options.typess.option import delta_call, delta_put
        f = delta_call if right == OptionRight.call else delta_put
        df.loc[right_df.index, 'bid_delta'] = f(right_df['spot'].values, right_df['strike_flt'].values, right_df['tenor'].values, right_df['bid_iv'].values, rate, dividend)
        df.loc[right_df.index, 'ask_delta'] = f(right_df['spot'].values, right_df['strike_flt'].values, right_df['tenor'].values, right_df['ask_iv'].values, rate, dividend)
    # df['bid_delta'] = df.apply(partial(ps2delta, iv_col='bid_iv', calendar=calendar, day_count=day_count, rate=rate, dividend=dividend), axis=1)
    # df['ask_delta'] = df.apply(partial(ps2delta, iv_col='ask_iv', calendar=calendar, day_count=day_count, rate=rate, dividend=dividend), axis=1)
    df['mid_delta'] = (df['bid_delta'] + df['ask_delta']) / 2

    df['moneyness'] = df.index.get_level_values('strike').astype(float) / df['spot']
    df['option_contract_str'] = df['option_contract'].values.astype(str)
    return df


def enrich_spread_measures(df: pd.DataFrame):
    df['spread'] = df['ask_close'] - df['bid_close']

    df['spread_iv'] = df['ask_iv'] - df['bid_iv']
    df['spread_iv_aewma'] = df['ask_iv_aewma'] - df['bid_iv_aewma']

    df['fill_as_%_of_spread'] = 100 * (df['fill_price'] - df['bid_close']) / df['spread']
    df['fill_as_%_of_spread_iv'] = 100 * (df['fill_price_iv'] - df['bid_iv']) / df['spread_iv']
    df['fill_as_%_of_spread_aewma_iv'] = 100 * (df['fill_price_iv'] - df['bid_iv_aewma']) / df['spread_iv_aewma']
    return df


def enrich_trade_direction_estimate(df: pd.DataFrame):
    df['direction'] = None
    df['direction'] = df['direction'].mask(df['fill_as_%_of_spread'] < 50, 'sell')
    df['direction'] = df['direction'].mask(df['fill_as_%_of_spread'] > 50, 'buy')
    ix_50_sell = df.index[df['direction'].isna()].values[::2]
    df.loc[ix_50_sell, 'direction'] = 'sell'
    df['direction'].fillna('buy', inplace=True)
    assert df['direction'].isna().sum() == 0
    return df


def plot_hist_fill_as_pc_of_spread(df: pd.DataFrame):
    len0 = len(df)
    df = df.loc[df['fill_as_%_of_spread'] <= 110]
    df = df.loc[df['fill_as_%_of_spread'] > -10]
    logger.info(f'rm {100 - 100 * len(df) / len0}%; #{len0 - len(df)} outliers')
    df.hist(column='fill_as_%_of_spread', by='direction', bins=50)


def plot_vol_by_moneyness(equity, df: pd.DataFrame, fn='vol_by_moneyness.html'):
    fig = plot_ps_trace(go.Scatter(x=df['moneyness'], y=df['volume'], mode='markers', marker=dict(size=4), name='volume'), show=False)
    fig.update_layout(title=f'Volume by moneyness for {equity} options', xaxis_title='Moneyness S/K', yaxis_title='Volume')
    show(fig, fn=fn)
    return [
        'vol_by_moneyness.html',
        md('As expected, volume is highest at ATM, and decreases as we move away from ATM.')
    ]


def plot_vol_by_delta(equity, df: pd.DataFrame, fn='vol_by_delta.html'):
    fig = plot_ps_trace(
        go.Scatter(x=abs(df['mid_delta']), y=df['volume'], mode='markers', marker=dict(size=4), name='volume'), show=False
    )
    fig.update_layout(title=f'Volume by delta for {equity} options', xaxis_title='Delta', yaxis_title='Volume')
    show(fig, fn='vol_by_delta.html')
    return ['vol_by_delta.html']


def plot_vol_by_tenor_delta(equity, dft: pd.DataFrame):
    fig_tenor = plot_ps_trace(
        go.Scatter(x=dft['tenor_bin'].apply(lambda i: i.left), y=dft['volume_rolled_sum'], mode='markers', marker=dict(size=4), name='volume_rolled_sum'), show=False
    )
    fig_tenor.update_layout(title=f'Volume by delta for {equity} options', xaxis_title='Tenor', yaxis_title='Volume')
    show(fig_tenor, fn='volume_by_tenor.html')

    fig_delta = plot_ps_trace(
        go.Scatter(x=dft['delta_bin'].apply(lambda i: i.left), y=dft['volume_rolled_sum'], mode='markers', marker=dict(size=4), name='volume_rolled_sum'), show=False
    )
    fig_delta.update_layout(title=f'Volume by delta for {equity} options', xaxis_title='Delta', yaxis_title='Volume')
    show(fig_delta, fn='volume_by_delta.html')
    return ['volume_by_tenor.html', 'volume_by_delta.html']


def enrich_bins(df: pd.DataFrame):
    df['volume_rolled_sum'] = None
    n_tenor_bins = 20
    n_delta_bins = 20
    df['tenor_bin'] = pd.cut(df['tenor'], bins=n_tenor_bins)
    df['delta_bin'] = pd.cut(abs(df['ask_delta']), bins=n_delta_bins)
    df['moneyness_bin'] = pd.cut(abs(df['moneyness']), bins=n_delta_bins)
    for ix, s_df in df.groupby(['tenor_bin', 'delta_bin', 'right', 'moneyness_bin']):
        df.loc[s_df.index, 'volume_rolled_sum'] = s_df['volume'].cumsum().values

    for ix, s_df in df.groupby(['tenor_bin', 'moneyness_bin']):
        df.loc[s_df.index, 'fill_as_%_of_spread_mean'] = s_df['fill_as_%_of_spread'].dropna().mean()
    return df


def plot_srf_moneyness_tenor_volume_cumsum(equity: Equity, dft: pd.DataFrame):
    df_srf = dft[['moneyness_bin', 'tenor_bin', 'volume_rolled_sum']].groupby(['moneyness_bin', 'tenor_bin']).sum().reset_index().pivot(index='moneyness_bin', columns='tenor_bin', values='volume_rolled_sum')

    x = [i.left for i in df_srf.index]
    y = [i.left for i in df_srf.columns]
    z = df_srf.values
    fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z.T, cmax=100, cmin=0, colorscale='Viridis'))
    fig.layout.scene.zaxis.range = [0, 100]
    fig.update_layout(title=f'Volume by moneyness and tenor for {equity} options', xaxis_title='Moneyness', yaxis_title='Tenor',
                      scene=dict(xaxis_title='Moneyness', yaxis_title='Tenor', zaxis_title='Volume'))
    show(fig, fn=f'{equity}_surface_moneyness_tenor_volume_cumsum_3d.html')
    return [f'{equity}_surface_moneyness_tenor_volume_cumsum_3d.html']


def plot_srf_moneyness_tenor_fill_as_pc_of_spread_mean_cumsum(equity: Equity, dft: pd.DataFrame):
    df_srf = (dft[['moneyness_bin', 'tenor_bin', 'fill_as_%_of_spread_mean']].groupby(['moneyness_bin', 'tenor_bin']).mean().reset_index().
              pivot(index='moneyness_bin', columns='tenor_bin', values='fill_as_%_of_spread_mean'))
    x = [i.left for i in df_srf.index]
    y = [i.left for i in df_srf.columns]

    p = []
    vals = []
    for ix, v in df_srf.iterrows():
        remains = v.dropna()
        p += [(i.left, v.name.left) for i in remains.index]
        vals += list(remains.values)

    from scipy.interpolate import LinearNDInterpolator  # , NearestNDInterpolator
    points = np.array(p)
    interp = LinearNDInterpolator(points, np.array(vals))
    # interp = NearestNDInterpolator(points, np.array(vals))
    z = np.array([[interp(x.left, y.left) for x in df_srf.columns] for y in df_srf.index])

    fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z, cmax=100, cmin=0, colorscale='Viridis'))
    fig.layout.scene.zaxis.range = [0, 100]
    fig.update_layout(title=f'fill_as_%_of_spread_mean by moneyness and tenor for {equity} options', xaxis_title='Moneyness', yaxis_title='Tenor',
                      scene=dict(xaxis_title='Moneyness', yaxis_title='Tenor', zaxis_title='fill_as_%_of_spread_mean'))
    show(fig, fn=f'{equity}_surface_moneyness_tenor_fill_as_%_of_spread_mean_cumsum_3d.html')
    return [f'{equity}_surface_moneyness_tenor_fill_as_%_of_spread_mean_cumsum_3d.html']


def plot_srf_delta_tenor_volume_cumsum(equity: Equity, dft: pd.DataFrame):
    df_srf = dft[['delta_bin', 'tenor_bin', 'volume_rolled_sum']].groupby(['delta_bin', 'tenor_bin']).sum().reset_index().pivot(index='delta_bin', columns='tenor_bin',
                                                                                                                                values='volume_rolled_sum')
    x = [i.left for i in df_srf.index]
    y = [i.left for i in df_srf.columns]
    z = df_srf.values
    fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z.T, cmax=100, cmin=0, colorscale='Viridis'))
    fig.layout.scene.zaxis.range = [0, 100]
    fig.update_layout(title=f'Volume by delta and tenor for {equity} options', xaxis_title='Delta', yaxis_title='Tenor',
                      scene=dict(xaxis_title='Delta', yaxis_title='Tenor', zaxis_title='Volume'))
    show(fig, fn=f'{equity}_surface_delta_tenor_volume_cumsum_3d.html')


def plot_delta_tenor_volume_cumsum_by_direction(equity: Equity, dft: pd.DataFrame):
    fig_sp = make_subplots(rows=2, cols=2, specs=[[{'is_3d': True}, {'is_3d': True}], [{'is_3d': True}, {'is_3d': True}]],
                           subplot_titles=['Call, Direction: Buy', 'Call, Direction: Sell', 'Put, Direction: Buy', 'Put, Direction: Sell'])
    for i, (right, s_df) in enumerate(dft.groupby('right')):
        for j, (direction, ss_df) in enumerate(s_df.groupby('direction')):
            df_srf_ij = ss_df[['delta_bin', 'tenor_bin', 'volume_rolled_sum']].groupby(['delta_bin', 'tenor_bin']).sum().reset_index().pivot(index='delta_bin', columns='tenor_bin',
                                                                                                                                             values='volume_rolled_sum')
            x = [i.left for i in df_srf_ij.index]
            y = [i.left for i in df_srf_ij.columns]
            z = df_srf_ij.values

            fig_sp.add_trace(go.Surface(x=x, y=y, z=z.T, cmax=100, cmin=0, colorscale='Viridis'), row=i + 1, col=j + 1)

    fig_sp.update_layout(title=f'Volume by delta and tenor for {equity} options',
                         scene1=dict(xaxis_title='Delta', yaxis_title='Tenor', zaxis_title='Volume', zaxis_range=[0, 100]),
                         scene2=dict(xaxis_title='Delta', yaxis_title='Tenor', zaxis_title='Volume', zaxis_range=[0, 100]),
                         scene3=dict(xaxis_title='Delta', yaxis_title='Tenor', zaxis_title='Volume', zaxis_range=[0, 100]),
                         scene4=dict(xaxis_title='Delta', yaxis_title='Tenor', zaxis_title='Volume', zaxis_range=[0, 100]),
                         )
    show(fig_sp, fn=f'{equity}_subplot_surface_delta_tenor_volume_cumsum_2d_by_right_direction.html')
    return [f'{equity}_subplot_surface_delta_tenor_volume_cumsum_2d_by_right_direction.html']


def plot_histogram_fill_as_pc_of_spread_by_direction(equity: Equity, df: pd.DataFrame):
    dft_hist = df[df['tenor'] < 0.5]
    fig_hist = make_subplots(rows=1, cols=2)
    fig_hist.add_trace(go.Histogram(x=dft_hist['fill_as_%_of_spread'], histnorm=None), row=1, col=1)
    fig_hist.add_trace(go.Histogram(x=dft_hist['fill_as_%_of_spread_iv'], histnorm=None), row=1, col=2)
    show(fig_hist)


def plot_srf_spread_iv_by_tenor_delta(equity: Equity, dft: pd.DataFrame):
    dft['spread_iv_pct'] = dft['spread_iv'] / (dft['ask_iv'] - dft['bid_iv']) - 1
    df_srf = dft[['delta_bin', 'tenor_bin', 'spread_iv']].groupby(['delta_bin', 'tenor_bin']).mean().reset_index().pivot(index='delta_bin', columns='tenor_bin',
                                                                                                                                values='spread_iv')
    x = [i.left for i in df_srf.index]
    y = [i.left for i in df_srf.columns]
    z = df_srf.values
    fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z.T, cmax=0.1, cmin=0, colorscale='Viridis'))
    fig.layout.scene.zaxis.range = [0, 0.1]
    fig.update_layout(title=f'Volume by delta and tenor for {equity} options', xaxis_title='Delta', yaxis_title='Tenor',
                      scene=dict(xaxis_title='Delta', yaxis_title='Tenor', zaxis_title='Spread IV'))
    show(fig, fn=f'{equity}_surface_delta_tenor_spread_iv_mean.html')
    return [f'{equity}_surface_delta_tenor_spread_iv_mean.html']


def plot_iv_quote_trade_by_ts(equity, df: pd.DataFrame):
    expiries = list(sorted(df.index.get_level_values('expiry').unique()))

    fig = make_subplots(rows=len(expiries), cols=1, subplot_titles=[dt.isoformat() for dt in expiries])

    for i, (expiry, s_df) in enumerate(df.groupby('expiry')):
        row = i+1
        for j, (strike, ss_df) in enumerate(s_df.groupby('strike')):
            moneyness0 = f"{ss_df['moneyness'].iloc[0]:.2f}"
            x = ss_df.index.get_level_values('ts')
            for col in ['bid_iv', 'ask_iv', 'fill_price_iv']:
                y = ss_df[col]
                fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'{expiry} | {moneyness0} | {col}', marker=dict(size=3)), row=row, col=1)

            col = 'fill_price_iv'
            ix_in = (ss_df[col] >= ss_df[col].quantile(0.01)) & (ss_df[col] <= ss_df[col].quantile(0.99))
            y = ss_df[ix_in][col].values
            if len(y) == 0:
                continue
            kf = KalmanFilter()
            # filtered_state_means, filtered_state_covariances = kf.filter(observations)
            smoothed_state_means, smoothed_state_covariances = kf.smooth(y)
            smoothed_y = smoothed_state_means[:, 0]
            smoothed_std_devs = np.sqrt(smoothed_state_covariances[:, 0].reshape(-1))

            # Compute upper and lower confidence intervals
            confidence_interval = 1  # 1.96  # 95% confidence interval
            upper_bound = smoothed_y + confidence_interval * smoothed_std_devs
            lower_bound = smoothed_y - confidence_interval * smoothed_std_devs

            fig.add_trace(go.Scatter(x=x[ix_in], y=upper_bound, mode='markers', name=f'Kalman U {expiry} | {moneyness0} | {col}', marker=dict(size=3)), row=row, col=1)
            fig.add_trace(go.Scatter(x=x[ix_in], y=smoothed_y, mode='markers', name=f'Kalman {expiry} | {moneyness0} | {col}', marker=dict(size=3)), row=row, col=1)
            fig.add_trace(go.Scatter(x=x[ix_in], y=lower_bound, mode='markers', name=f'Kalman D {expiry} | {moneyness0} | {col}', marker=dict(size=3)), row=row, col=1)

    title = f'{equity}_iv_quote_trade_by_ts'
    fn = f'{title}.html'
    fig.update_layout(title=title)
    show(fig, fn=fn)
    return [fn]


def plot_spread_measures_by_ts(equity, df: pd.DataFrame):
    expiries = list(sorted(df.index.get_level_values('expiry').unique()))

    fig = make_subplots(rows=len(expiries), cols=1, subplot_titles=[dt.isoformat() for dt in expiries])

    for i, (expiry, s_df) in enumerate(df.groupby('expiry')):
        row = i + 1
        for j, (strike, ss_df) in enumerate(s_df.groupby('strike')):
            print(f'{expiry} | {strike}')
            moneyness0 = f"{ss_df['moneyness'].iloc[0]:.2f}"
            x = ss_df.index.get_level_values('ts')
            # for col in ['spread_iv', 'spread_iv_aewma', 'fill_as_%_of_spread', 'fill_as_%_of_spread_iv', 'fill_as_%_of_spread_aewma_iv']:
            for col in ['spread_iv', 'spread_iv_aewma']:

                ix_in = (ss_df[col] >= ss_df[col].quantile(0.01)) & (ss_df[col] <= ss_df[col].quantile(0.99))
                y = ss_df[ix_in][col]
                fig.add_trace(go.Scatter(x=x[ix_in], y=y, mode='markers', name=f'{col} | {expiry} | {moneyness0}', marker=dict(size=3)), row=row, col=1)

            if len(y) == 0:
                continue
            col = 'spread_iv'
            ix_in = (ss_df[col] >= ss_df[col].quantile(0.01)) & (ss_df[col] <= ss_df[col].quantile(0.99))
            y = ss_df[ix_in][col].values
            kf = KalmanFilter()
            # filtered_state_means, filtered_state_covariances = kf.filter(observations)
            smoothed_state_means, smoothed_state_covariances = kf.smooth(y)
            smoothed_y = smoothed_state_means[:, 0]

            fig.add_trace(go.Scatter(x=x[ix_in], y=smoothed_y, mode='markers', name=f'Kalman {expiry} | {moneyness0} | {col}', marker=dict(size=3)), row=row, col=1)

    title = f'{equity}_spread_measured_by_ts'
    fn = f'{title}.html'
    fig.update_layout(title=title)
    show(fig, fn=fn)
    return [fn]


def run():
    """
    Given a trade count and a time period, find the optimal % of spread between 2 borders (bid/ask, ewma(bid/ask), etc.) to quote that ensures a certain fill rate.
    Objectives:
    - Improved earnings release option portfolio, because simulated fill price is using estimated discount rather than mid iv.
      Therefore, options with low chance of fill will pay the spread and less likely appear as condidate for the portfolio.

    - Plot how these borders rolls over time.
    - Calibrate an estimator of those borders.

    Before plotting anything "over time", try it for a single time period...

    If 100% discount, p_fill 100%. If 0% discount, it's also 100% if there is any volume on the opposite bid/ask. That's kinda wrong. Might be on a different exchange,
    dark pool, off-exchange. So would want a minimum volume, eg, vol / 100. Anything below 1 is considered 0, hence no fill. Towards mid iv, that volume increases, hence p_fill increases.
    So also need a cumulative volume metric per direction, from 0% discount towards mid iv.

    Observations:
    - volume depends a lot more on tenor, rather than delta/moneyness.
    - volume is highest at ATM, and decreases as we move away from ATM.

    Bucketing by expiry (not tenor) -> Histogram.
    Further bucketing by delta -> Histogram.
    How am I getting to an optimal discount here???
    Need to apply filters. What's the volume at 0% discount, 50% discount, 100% discount.
    Eventually regress exactly that or just pick one given overall volume trends up and down quite a bit.
    Simplify.
    Expecting a surface of volume by delta, tenor. So I need to cut the data into buckets, then plot the surface.

    Improvements: None of above considers the vol of vol. Using EWMA or other smoothing methods could avoid following aggressive limit orders. What may be mid-iv in minute 0, could be a big discount in minute 1.
    Therefore, need a reference column for buy/sell: 1) bid_close/ask_close, 2) ewma(bid_close)/ewma(ask_close)
    """
    equity = Equity('DELL')
    take = -1
    release_date = EarningsPreSessionDates(equity.symbol)[take]
    start = release_date
    end = release_date + timedelta(days=1)
    resolution = Resolution.second

    df = load_data(start, end, equity, take, resolution)
    enrich(equity, df)
    enrich_spread_measures(df)
    enrich_trade_direction_estimate(df)
    enrich_bins(df)

    plot_iv_quote_trade_by_ts(equity, df)
    plot_spread_measures_by_ts(equity, df)

    # # scope
    # r = []
    #
    # plot_hist_fill_as_pc_of_spread(df)
    # r += plot_vol_by_moneyness(equity, df)
    # r += plot_vol_by_delta(equity, df, fn='vol_by_delta.html')
    #
    # # Given a cutoff volume x per Period and fill_%, what is the min/max moneyness or delta worth pursuing. easier to use delta, encompasses expiry and moneyess
    # # assumption. cut 50% in half for buy sell. 0/100
    # df = df.sort_index(level='ts')
    # start_dt = datetime.combine(start, time(hour=9, minute=30))
    # end_dt = datetime.combine(end, time(hour=16))
    # df = df.loc[start_dt:end_dt]
    #
    # dfs = []
    # # Take the first ts of every expiry, strike, right (option)!
    # for ix, s_df in df.groupby(['expiry', 'strike', 'right']):
    #     dfs.append(s_df.loc[s_df.index.get_level_values('ts').min()])
    # dft = pd.concat(dfs, axis=0)
    #
    # plot_vol_by_tenor_delta(equity, dft)
    # r += plot_srf_moneyness_tenor_volume_cumsum(equity, dft)
    # r += plot_srf_moneyness_tenor_fill_as_pc_of_spread_mean_cumsum(equity, dft)
    # r += plot_srf_delta_tenor_volume_cumsum(equity, dft)
    # r += plot_delta_tenor_volume_cumsum_by_direction(equity, dft)
    #
    # plot_histogram_fill_as_pc_of_spread_by_direction(equity, df)
    # plot_srf_spread_iv_by_tenor_delta(equity, dft)


if __name__ == "__main__":
    run()
