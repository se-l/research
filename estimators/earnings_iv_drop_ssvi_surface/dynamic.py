import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta, datetime, time, date
from functools import lru_cache
from itertools import groupby, chain
from typing import Iterable, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


from options.frame_builder import available_sym_dates, get_option_frame
from estimators.earnings_iv_drop_ssvi_surface.kalman_filter import initialize_kalman_filter, get_measurement, get_params
from estimators.earnings_iv_drop_ssvi_surface.train_estimator import plot_ssvi_params_dist_across_dates, ScopePrePost, get_v_ivs, scoped_dates
from options.helper import get_pkl_cache_key
from options.types.enums import OptionRight, Resolution
from options.types.equity import Equity
from options.types.iv_surface_essvi import IVSurface
from options.types.option import Option
from options.types.option_contract import OptionContract
from options.types.portfolio import Portfolio
from options.types.quote_side import QuoteSide
from options.types.sym_date import SymDate
from shared.modules.logger import info
from shared.paths import Paths
from shared.plotting import show


@dataclass
class BacktestResult:
    equity: Equity
    option: Option
    ts: pd.Timestamp | datetime

    iv_ma: float
    iv_market: float
    vega_market: float

    q: float


class ParamsMA:
    def __init__(self, ma_len: int):
        self.ma_len = ma_len
        self.params_hist = []
        self.params_ma = {}

    def __add__(self, other: Dict):
        self.params_hist.append(other)
        if len(self.params_hist) > self.ma_len:
            self.params_hist.pop(0)

        merged_keys = set(chain(*[p.keys() for p in self.params_hist]))
        for tenor_dt, right in merged_keys:
            v_theta = [p.get((tenor_dt, right), (np.nan, np.nan, np.nan))[0] for p in self.params_hist]
            v_rho = [p.get((tenor_dt, right), (np.nan, np.nan, np.nan))[1] for p in self.params_hist]
            v_psi = [p.get((tenor_dt, right), (np.nan, np.nan, np.nan))[2] for p in self.params_hist]
            self.params_ma[(tenor_dt, right)] = (np.mean(v_theta), np.mean(v_rho), np.mean(v_psi))
        return self

    def get(self, tenor_dt: pd.Timedelta, right: OptionRight):
        self.params_ma.get((tenor_dt, right), np.nan)


@lru_cache(maxsize=1000)
def get_option(equity, tenor_dt, strike, right, calc_date):
    return Option(OptionContract('', equity.symbol, tenor_dt, strike, right), calc_date)


def bt_ivs(ivs_model: IVSurface, pf, df, ts):
    equity = ivs_model.underlying
    calc_date = ts.date()

    bt_results = []
    # for each tenor, right....
    for right, s_df in df.groupby('right'):
        for tenor_dt, ss_df in s_df.groupby('expiry'):
            # get the iv from the market data

            v_strike = ss_df.index.get_level_values('strike').values
            v_mny = ss_df['moneyness_fwd_ln'].values

            v_iv_market = ss_df['mid_iv'].values
            # v_iv_bid = ss_df['bid_iv'].values
            # v_iv_ask = ss_df['ask_iv'].values
            v_vega_market = ss_df['vega_mid_iv'].values
            iv_model = ivs_model.iv(v_mny, tenor_dt, right, calc_date)

            # Buy Sell - Ideally when some z-score is crossed

            # Buy when iv_model is higher than iv market
            ix_buy = iv_model > v_iv_market
            strikes_buy = v_strike[ix_buy]

            for i, strike in enumerate(strikes_buy):
                security = get_option(equity, tenor_dt, strike, right, calc_date)
                position_q = pf.portfolio.get(security, 0)
                if position_q > 0:
                    continue
                else:
                    q = position_q * -1 + 1
                    pf.add_holding(security, q)
                    bt_results.append(
                        BacktestResult(equity, security, ts, iv_ma=iv_model[ix_buy][i], iv_market=v_iv_market[ix_buy][i], vega_market=v_vega_market[ix_buy][i], q=q))

            ix_sell = iv_model < v_iv_market
            strikes_sell = v_strike[ix_sell]
            for i, strike in enumerate(strikes_sell):
                security = get_option(equity, tenor_dt, strike, right, calc_date)
                current_q = pf.portfolio.get(security, 0)
                if current_q < 0:
                    continue
                else:
                    q = current_q * -1 - 1
                    pf.add_holding(security, q)
                    bt_results.append(
                        BacktestResult(equity, security, ts, iv_ma=iv_model[ix_sell][i], iv_market=v_iv_market[ix_sell][i], vega_market=v_vega_market[ix_sell][i], q=q))
    return bt_results


def backtest_kalman_filtered_model_param_over_time(v_ivs: Iterable[IVSurface]) -> List[BacktestResult]:
    all_bt_results = []

    v_ivs = sorted(list(v_ivs), key=lambda ivs: ivs.underlying.symbol)

    for equity, v_ivs_by_equity in groupby(v_ivs, lambda ivs: ivs.underlying):
        # Need to split by release date... train on -1, test on 0

        # Get the options frame for the time period
        v_ivs_by_equity = sorted(list(v_ivs_by_equity), key=lambda x: x.last_calibration_ts())
        if not v_ivs_by_equity:
            continue

        df_options = get_option_frame(equity, dates=sorted(list({ivs.last_calibration_ts().date() for ivs in v_ivs_by_equity})), resolution=Resolution.second,
                                      seq_ret_threshold=0.002).df_options

        gen = groupby(v_ivs_by_equity, lambda ivs: ivs.last_calibration_ts().date())
        dt, surfaces_by_dt = next(gen)
        surfaces = sorted(surfaces_by_dt, key=lambda ivs: ivs.last_calibration_ts())
        if not surfaces:
            continue

        kf, state_0 = initialize_kalman_filter(equity, surfaces)
        X, P = state_0.X, state_0.P

        for dt, surfaces_by_dt in gen:
            surfaces = sorted(surfaces_by_dt, key=lambda ivs: ivs.last_calibration_ts())
            if not surfaces:
                continue

            pf = Portfolio()
            bt_results = []

            for i, ivs in enumerate(surfaces):

                z = get_measurement(ivs)
                X, P = kf.filter_update(X, P, observation=z, observation_matrix=kf.observation_matrices)

                ts = ivs.last_calibration_ts()
                ivs_model = IVSurface(equity)
                ivs_model.set_last_calibration_ts(ts)
                ivs_model.set_params(get_params(z, state_0.tenors_right_pairs))

                # from the market data vector, pick v_strike
                v_ts = df_options.index.get_level_values('ts')
                last_df_ts = v_ts[v_ts <= ts].max()
                df = df_options.loc[last_df_ts]

                bt_results += bt_ivs(ivs_model, pf, df, ts)

            for security, position in pf.holdings.items():
                option: Option = security

                if position != 0:
                    iv_market = df.loc[(option.expiry, option.strike, option.right), 'mid_iv']
                    vega_market = df.loc[(option.expiry, option.strike, option.right), 'vega_mid_iv']
                    mny_market = df.loc[(option.expiry, option.strike, option.right), 'moneyness_fwd_ln']
                    iv_model = ivs_model.iv(mny_market, option.expiry, option.right, dt)

                    pf.add_holding(option, -position)
                    bt_results.append(
                        BacktestResult(equity, option, datetime(dt.year, dt.month, dt.day, 16, 0, 0), iv_ma=iv_model, iv_market=iv_market, vega_market=vega_market, q=-position))

            all_bt_results.extend(bt_results)
    return all_bt_results


def backtest_mean_model_param_over_time(v_ivs: Iterable[IVSurface], ma_len=10) -> List[BacktestResult]:
    """need to derive surface at every timestamp. reiterate the smoothing process and make buy/sell/hold decision"""

    all_bt_results = []

    v_ivs = sorted(list(v_ivs), key=lambda ivs: ivs.underlying.symbol)

    for equity, v_ivs_by_equity in groupby(v_ivs, lambda ivs: ivs.underlying):
        # Get the options frame for the time period
        v_ivs_by_equity = sorted(list(v_ivs_by_equity), key=lambda ivs: ivs.last_calibration_ts())
        if not v_ivs_by_equity:
            continue

        df_options = get_option_frame(equity, dates=sorted(list({ivs.last_calibration_ts().date() for ivs in v_ivs_by_equity})), resolution=Resolution.second, seq_ret_threshold=0.002).df_options

        gen = groupby(v_ivs_by_equity, lambda ivs: ivs.last_calibration_ts().date())
        dt, surfaces_by_dt = next(gen)

        for dt, surfaces_by_dt in gen:
            surfaces = list(surfaces_by_dt)
            params_ma = ParamsMA(ma_len)
            pf = Portfolio()
            bt_results = []

            for i, ivs in enumerate(sorted(surfaces, key=lambda ivs: ivs.last_calibration_ts())):

                params_ma += ivs.params
                if i < ma_len:
                    continue

                ivs_model = IVSurface(equity)
                ivs_model.set_params(params_ma.params_ma)

                ts = ivs.last_calibration_ts()
                # from the market data vector, pick v_strike
                v_ts = df_options.index.get_level_values('ts')
                last_df_ts = v_ts[v_ts <= ts].max()
                df = df_options.loc[last_df_ts]

                bt_results += bt_ivs(ivs_model, pf, df, ts)

            for security, position in pf.holdings.items():
                option: Option = security

                if position != 0:
                    iv_market = df.loc[(option.expiry, option.strike, option.right), 'mid_iv']
                    vega_market = df.loc[(option.expiry, option.strike, option.right), 'vega_mid_iv']
                    mny_market = df.loc[(option.expiry, option.strike, option.right), 'moneyness_fwd_ln']
                    iv_model = ivs_model.iv(mny_market, option.expiry, option.right, dt)

                    pf.add_holding(option, -position)
                    bt_results.append(BacktestResult(equity, option, datetime(dt.year, dt.month, dt.day, 16, 0, 0), iv_ma=iv_model, iv_market=iv_market, vega_market=vega_market, q=-position))

            all_bt_results.extend(bt_results)
    return all_bt_results


def log_bt_results(bt_results):
    results = sorted(bt_results, key=lambda x: x.option)
    iv_gained = defaultdict(float)
    usd_gain = defaultdict(float)
    bool_gain = []
    for security, gp in groupby(results, lambda res: res.option):
        pos = 0
        for bt_res in sorted(gp, key=lambda x: x.ts):
            iv_diff = bt_res.iv_market - bt_res.iv_ma
            iv_gain = iv_diff * pos
            bool_gain.append(iv_gain > 0)
            iv_gained[security] += iv_gain
            usd_gain[security] += iv_gain * bt_res.vega_market * 100
            pos += bt_res.q

    print(f'iv_gained: {sum(iv_gained.values())}')
    print(f'usd_gain: {sum(usd_gain.values())}')
    print(f'cnt_gain: {sum(bool_gain) / len(bool_gain)}')


def plot_backtests_by_option(bt_results):
    results = sorted(bt_results, key=lambda x: x.option)

    # plot each option series
    # little green up triangle for buy, red down triangle for sell
    # Plot bid, ask, smoothed mid and actual mid

    f = lambda r: r.equity
    for equity, r_by_equity in groupby(sorted(results, key=f), f):
        r_by_equity = list(r_by_equity)
        dates = list({r.ts.date() for r in r_by_equity})
        df_options = get_option_frame(equity, dates=dates, resolution=Resolution.second, seq_ret_threshold=0.002).df_options

        f = lambda r: r.option
        for security, r_by_option in groupby(sorted(r_by_equity, key=f), f):
            r_by_option = list(r_by_option)
            if len(r_by_option) < 10:
                continue
            option: Option = security

            dates = list(set([r.ts.date() for r in r_by_option]))
            if not dates:
                continue
            fig = make_subplots(rows=len(dates), cols=1)

            df_sec = df_options.loc[(slice(None), option.expiry, option.optionContract.strike, option.right)]

            f = lambda r: r.ts.date()
            for i, (dt, gp) in enumerate(groupby(sorted(r_by_option, key=f), f)):
                gp = list(gp)

                ix_row = i + 1
                v_ts = [ts for ts in df_sec.index.get_level_values('ts') if dt <= ts.date() < dt + timedelta(days=1) and ts.time() > time(9, 35)]
                df_dt = df_sec.loc[v_ts]

                v_ts = df_dt.index.get_level_values('ts')
                v_mny = df_dt['moneyness_fwd_ln'].values
                v_bid_iv = df_dt['bid_iv'].values
                v_ask_iv = df_dt['ask_iv'].values
                v_mid_iv = df_dt['mid_iv'].values

                fig.add_trace(go.Scatter(x=v_ts, y=v_bid_iv, mode='markers+lines', name=f'{option} bid', line=dict(width=1, color='grey'), marker=dict(size=2, color='grey')),
                              row=ix_row, col=1)
                # fig.add_trace(go.Scatter(x=v_ts, y=v_ask_iv, mode='markers+lines', name=f'{option} ask', line=dict(width=1, color='grey'), marker=dict(size=2, color='grey')),
                #               row=ix_row, col=1)
                fig.add_trace(go.Scatter(x=v_ts, y=v_mid_iv, mode='markers+lines', name=f'{option} mid', line=dict(width=1, color='grey'), marker=dict(size=2, color='grey')),
                              row=ix_row, col=1)

                v_ma_ts_buy = [r.ts for r in gp if r.q > 0]
                v_iv_ma_buy = [r.iv_market for r in gp if r.q > 0]
                fig.add_trace(go.Scatter(x=v_ma_ts_buy, y=v_iv_ma_buy, mode='markers', name=f'{option} buy', marker=dict(size=8, color='green', symbol='triangle-up')), row=ix_row,
                              col=1)

                v_ma_ts_sell = [r.ts for r in gp if r.q < 0]
                v_iv_ma_sell = [r.iv_market for r in gp if r.q < 0]
                fig.add_trace(go.Scatter(x=v_ma_ts_sell, y=v_iv_ma_sell, mode='markers', name=f'{option} sell', marker=dict(size=8, color='red', symbol='triangle-down')),
                              row=ix_row, col=1)

            fig.update_layout(title=str(option.optionContract))
            show(fig, fn=f'{option.optionContract.ib_symbol()}.html')


def available_dates(tickers: List[str]=None, arb_free=False):
    tickers_in = ','.join(tickers or os.listdir(r'D:\trade\data\option\usa\second')).upper()

    sym_dates = available_sym_dates(tickers_in)
    clear_prefix = 'v_ivs'
    root = Paths.path_analysis_frames

    payloads = []
    for sym_date in sym_dates:
        sym = sym_date.symbol.lower()
        release_date = sym_date.date
        equity = Equity(sym)
        resolution = Resolution.second
        seq_ret_threshold = 0.002

        for dt in scoped_dates(release_date, ScopePrePost.all):
            fn = get_pkl_cache_key(clear_prefix, equity, [dt], resolution, seq_ret_threshold, QuoteSide.mid, arb_free)
            if os.path.exists(os.path.join(root, fn)):
                payloads.append((sym, dt))
    info(f'Found {len(payloads)} payloads\n{payloads}')



if __name__ == "__main__":
    """
    pending model improvements:
    - weight by tenor
    - IV of nearest neighbor stocks, sector and overall market
    - historical stock vol changes. so actual vol vs implied vol ( can be accomplished with daily data )
    """
    # tickers_ho = ['ORCL', 'DELL', 'ONON']
    tickers_ho = ['FDX']

    sym_dates_ho = available_sym_dates(','.join(tickers_ho))
    # available_dates(['FDX'])

    # for sym_date in sym_dates_ho:
    for sym_date in [SymDate('FDX', date(2024, 2, 29))]:
        # v_ivs = get_ivs_mp([sym_date], mp=False, scope=ScopePrePost.all)
        v_ivs = get_v_ivs(Equity('FDX'), [date(2024, 2, 29)], resolution=Resolution.second, seq_ret_threshold = 0.002)
        # plot_ssvi_params_over_time(v_ivs)
        plot_ssvi_params_dist_across_dates(v_ivs)
        info(sym_date)
        # ma_results = backtest_mean_model_param_over_time(v_ivs, 5)
        # log_bt_results(ma_results)
        # print('-'*20)
        # results = backtest_kalman_filtered_model_param_over_time(v_ivs)
        # log_bt_results(results)
        # plot_backtests_by_option(results)

