import hashlib
import json
import operator
import os
import pickle
import traceback

import QuantLib as ql
import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from hashlib import sha256
from datetime import date, datetime, time, timedelta
from functools import reduce, lru_cache
from typing import List, Union, Tuple, Dict
from importlib import reload
from itertools import chain

import py_vollib.black_scholes_merton.implied_volatility
import py_vollib_vectorized

from arbitragerepair import constraints, repair
from matplotlib import gridspec
from scipy.interpolate.interpnd import NDInterpolatorBase
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

import options.client as mClient
from options.typess.enums import Resolution, TickType, SecurityType, GreeksEuOption, SkewMeasure, OptionRight
from options.typess.holding import Holding
from options.typess.option_contract import OptionContract
from options.typess.equity import Equity
from options.typess.option_frame import OptionFrame
from options.typess.portfolio import Portfolio
from options.typess.security import Security
from shared.constants import EarningsPreSessionDates, DividendYield, DiscountRateMarket
from shared.modules.logger import logger, warning

from shared.paths import Paths

reload(mClient)
client = mClient.Client()


def ps2mid_iv_if_nonzero(ps: pd.Series) -> pd.Series:
    # if ps['ask_iv'] == 0 or ps['bid_iv'] == 0:
    #     return np.nan  # pd.Series(np.empty(ps.shape), index=ps.index)
    return (ps['ask_iv'] + ps['bid_iv']) / 2


def load_option_trades(contracts, start, end, resolution) -> Dict[OptionContract, pd.DataFrame]:
    trades = client.history(contracts, start, end, resolution, TickType.trade, SecurityType.option)
    return {OptionContract.from_contract_nm(k): v for k, v in trades.items()}


def join_spot(dct_option_df: Dict[OptionContract, pd.DataFrame], ps_spot: pd.Series):
    for k, df in dct_option_df.items():
        df.index.name = 'ts'
        df = df.merge(ps_spot.rename('spot'), how='outer', right_index=True, left_index=True).sort_index()
        df['spot'].ffill(inplace=True)
        dct_option_df[k] = df.dropna()
    return dct_option_df


def join_quotes(trades: Dict[OptionContract, pd.DataFrame], quotes: Dict[OptionContract, pd.DataFrame], col_join=('bid_close', 'ask_close')):
    for k, df in trades.items():
        df_quote = quotes.get(k)
        if df_quote is None:
            continue
        df.index.name = 'ts'
        df = df.merge(df_quote[list(col_join)], how='outer', right_index=True, left_index=True).sort_index()
        for c in col_join:
            df[c].ffill(inplace=True)
        trades[k] = df.dropna()
    return trades


def regression_report(df, x, y):
    print(f'Regression report {x} vs {y}:')
    reg = LinearRegression().fit(df[[x]], df[y])
    r2 = reg.score(df[[x]], df[y])
    print(f'''R2 Linear Regr.: {r2}''')

    def a_bx_cx2(x, a, b, c):
        return a + b * x + c * x ** 2

    popt, pcov = curve_fit(a_bx_cx2, df[x], df[y])
    # print(popt)

    y_pred = a_bx_cx2(df[x], *popt)
    r2 = r2_score(df[y], y_pred)
    print(f'R2 Quadratic Regr.: {r2}')


def iv_of_expiry(optionContracts: List[OptionContract], trades, quotes, resolution='60min'):
    from options.typess.option import Option
    mat_df = {}
    for contract in optionContracts:
        symbol = str(contract)
        print(symbol)
        underlying_str = symbol.split('_')[0]
        df_q = quotes[symbol]
        df_t = trades[underlying_str]
        if resolution:
            df_q = client.resample(df_q, resolution=resolution or '60min')
            df_t = client.resample(df_t, resolution=resolution or '60min')
        df_t['mid_close_underlying'] = (df_t['bid_close'] + df_t['ask_close']) / 2
        df_q['mid_close'] = (df_q['bid_close'] + df_q['ask_close']) / 2
        df = client.union_vertically([df_q, df_t[['mid_close_underlying']]])
        # exclude out of hours trading
        if 'D' not in resolution:
            df = df[(time(9, 30) <= df.index.time) & (df.index.time <= time(16, 0))]
        df = df[[d.weekday() not in [5, 6] for d in df.index.date]]
        df = df.sort_index()
        if (len(df)) == 0:
            print(f'No data for {symbol}')
            continue

        # Drop any null bid close or ask close
        df.dropna(subset=['bid_close', 'ask_close'], inplace=True)
        # assert df.isna().sum().sum() == 0
        option = Option(contract)
        print(f'Calculating IV for {symbol}')
        print(f'DF len: {len(df)}, DF columns: {df.columns}')
        df['bid_iv'] = list(option.ivs(df['bid_close'], df['mid_close_underlying'], df.index))
        df['ask_iv'] = list(option.ivs(df['ask_close'], df['mid_close_underlying'], df.index))
        df['mid_iv'] = (df['bid_iv'] + df['ask_iv']) / 2

        # removing outliers. remove 3 z-scores away from the mean
        confidence_level = 3
        lookback_period = timedelta(days=5)
        rolling_iv_mean = pd.Series(df['mid_iv']).rolling(window=lookback_period, min_periods=0).mean()
        rolling_iv_std = pd.Series(df['mid_iv']).rolling(window=lookback_period, min_periods=0).std()
        upper_bound = rolling_iv_mean + confidence_level * rolling_iv_std
        lower_bound = rolling_iv_mean - confidence_level * rolling_iv_std
        df = df[(df['mid_iv'] < upper_bound) & (df['mid_iv'] > lower_bound)]
        # df = df[(df['mid_iv'] < df['mid_iv'].quantile(0.95)) & (df['mid_iv'] > df['mid_iv'].quantile(0.05))]
        mat_df[symbol] = df
        # print(f'{symbol} loaded.')
    return mat_df


# Some utility functions used later to plot 3D vol surfaces, generate paths, and generate vol surface from Heston params
def plot_vol_surface(vol_surface, plot_years=None, plot_strikes=None, funct='blackVol'):
    plot_strikes = plot_strikes if plot_strikes is not None else np.arange(vol_surface.minStrike(), vol_surface.maxStrike(), 1)
    plot_years = plot_years if plot_years is not None else np.arange(0.01, 2, 0.1)
    if type(vol_surface) != list:
        surfaces = [vol_surface]
    else:
        surfaces = vol_surface

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y = np.meshgrid(plot_strikes, plot_years)

    for surface in surfaces:
        method_to_call = getattr(surface, funct)

        Z = np.array([method_to_call(float(y), float(x))
                      for xr, yr in zip(X, Y)
                      for x, y in zip(xr, yr)]
                     ).reshape(len(X), len(X[0]))

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0.1)

    fig.colorbar(surf, shrink=0.5, aspect=5)


def plot_surface(list_of_list, X, Y):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y = np.meshgrid(X, Y)
    Z = np.array(list_of_list)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)


def np1d_from_df(df2, c):
    dft = df2.groupby(['strike', 'expiry'])[c].aggregate('first').unstack()
    return np.array(list(chain(*dft.transpose().values)))


def plot_prices_iv_repair(T, F, K, C, iv):
    fig = plt.figure(figsize=(12, 6))
    spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)

    ax = fig.add_subplot(spec[0, :])  # plot the forward curve
    ax.set_title('Forward curve')
    unq_Ts, idx_T = np.unique(T, return_index=True)
    ax.plot(unq_Ts, F[idx_T], '--ok')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$F(T)$')

    ax = fig.add_subplot(spec[1:, 0], projection='3d')
    ax.set_title('Call price surface')
    ax.scatter3D(T, K, C, s=3, color='k')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$K$')
    ax.view_init(30, 170)

    ax = fig.add_subplot(spec[1:, 1], projection='3d')
    ax.set_title('Implied volatility surface')
    ax.scatter3D(T, K, iv, s=3, color='k')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$K$')
    ax.view_init(30, 20)

    plt.tight_layout()
    plt.show()


def plot_after_repair(T, F, K, C, iv, expiries, epsilon1, epsilon2, spread_bid, spread_ask, C_bid, C_ask):
    tol = 1e-8
    n_quote = len(C)
    unq_Ts, idx_T = np.unique(T, return_index=True)

    expiry_str = sorted(set([c.isoformat() for c in expiries]))

    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(211)
    ax.plot(epsilon1 / C, 'ko', markersize=3, alpha=.8)
    mask1 = epsilon1[epsilon1 > 0] - spread_ask[epsilon1 > 0] > tol
    ax.plot(np.arange(n_quote)[epsilon1 > 0][mask1],
            ((epsilon1 / C)[epsilon1 > 0])[mask1], 'ro', markersize=8, alpha=.5)
    mask2 = epsilon1[epsilon1 < 0] + spread_bid[epsilon1 < 0] < -tol
    ax.plot(np.arange(n_quote)[epsilon1 < 0][mask2],
            ((epsilon1 / C)[epsilon1 < 0])[mask2], 'ro', markersize=8, alpha=.5)
    for t in unq_Ts:
        mask_t = T == t
        ax.fill_between(np.arange(n_quote)[mask_t],
                        (-spread_bid / C)[mask_t], (spread_ask / C)[mask_t], color='C0', alpha=.2)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x * 100) for x in vals])
    ax.set_title(r"$\ell^1$")
    ax.set_xticks(np.ceil((idx_T + np.hstack([idx_T[1:], len(T) - 1])) / 2))
    ax.set_xticklabels(expiry_str)

    ax = fig.add_subplot(212)
    ax.plot(epsilon2 / C, 'ko', markersize=3, alpha=.8, label='Perturbation')
    mask1 = epsilon2[epsilon2 > 0] - spread_ask[epsilon2 > 0] > tol
    ax.plot(np.arange(n_quote)[epsilon2 > 0][mask1],
            ((epsilon2 / C)[epsilon2 > 0])[mask1], 'ro', markersize=8, alpha=.5)
    mask2 = epsilon2[epsilon2 < 0] + spread_bid[epsilon2 < 0] < -tol
    ax.plot(np.arange(n_quote)[epsilon2 < 0][mask2],
            ((epsilon2 / C)[epsilon2 < 0])[mask2], 'ro', markersize=8, alpha=.5,
            label='Perturbation out of bid-ask price bounds')
    for t in unq_Ts:
        mask_t = T == t
        ax.fill_between(np.arange(n_quote)[mask_t],
                        (-spread_bid / C)[mask_t], (spread_ask / C)[mask_t], color='C0', alpha=.2)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x * 100) for x in vals])
    ax.set_title(r"$\ell^1$-BA")
    ax.set_xticks(np.ceil((idx_T + np.hstack([idx_T[1:], len(T) - 1])) / 2))
    ax.set_xticklabels(expiry_str)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_raw_repaired_prices(expiry_str, unq_Ts, T1, K1, C1, epsilon1, epsilon2):
    idx_T_sub = range(len(expiry_str))  # [2,3,4,7,8]

    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(131)
    ax.set_title('Raw prices')
    for i in idx_T_sub:
        mask_t = T1 == unq_Ts[i]
        ax.plot(K1[mask_t], C1[mask_t], '--o', label=r'$T=$' + expiry_str[i])
    ax.legend()

    ax = fig.add_subplot(132)
    ax.set_title(r'Repaired prices by the $\ell^1$-norm')
    for i in idx_T_sub:
        mask_t = T1 == unq_Ts[i]
        ax.plot(K1[mask_t], (C1 + epsilon1)[mask_t], '--o', label=r'$T=$' + expiry_str[i])
    ax.legend()

    ax = fig.add_subplot(133)
    ax.set_title(r'Repaired prices by the $\ell^1$-BA')
    for i in idx_T_sub:
        mask_t = T1 == unq_Ts[i]
        ax.plot(K1[mask_t], (C1 + epsilon2)[mask_t], '--o', label=r'$T=$' + expiry_str[i])
    ax.legend()

    plt.tight_layout()
    plt.show()


def pkl_nm(tickers: str, takes: List[int]) -> str:
    sha256_hash = sha256()
    obj_str = tickers + str(takes)
    sha256_hash.update(obj_str.encode('utf-8'))
    digest = sha256_hash.hexdigest()
    return f'earnings_iv_drop_{digest}.pkl'


def repair_prices(df_q, calculation_date: date, n_repairs=1, plot=False, right='call', net_yield=0, col_nm_map=None, solver='glpk'):
    """
    C+PV(K)=P+S
    Put call parity: S + p = c + K / (1 + r)^T
    p = c - S + K / (1 + r)^T
    c = p + S - K / (1 + r)^T

    solver: 'glpk' or 'ipopt' or 'cbc'
    """
    col_nm_map = col_nm_map or {}
    col_nm_map = {**{'mid_close': 'mid_close', 'bid_close': 'bid_close', 'ask_close': 'ask_close', 'mid_iv': 'mid_iv', 'strike': 'strike', 'tenor': 'tenor', 'expiry': 'expiry'},
                  **col_nm_map}

    nm_mid_close = col_nm_map['mid_close']

    C = np1d_from_df(df_q, nm_mid_close)
    C_bid = np1d_from_df(df_q, col_nm_map['bid_close'])
    C_ask = np1d_from_df(df_q, col_nm_map['ask_close'])
    iv = np1d_from_df(df_q, col_nm_map['mid_iv'])
    K = np1d_from_df(df_q, col_nm_map['strike']).astype(float)
    T = np1d_from_df(df_q, col_nm_map['tenor'])
    Tdt = np1d_from_df(df_q, col_nm_map['expiry'])
    S = np1d_from_df(df_q, col_nm_map['mid_close_underlying'])
    F = S * np.exp(net_yield * T)

    if right == 'put':
        # c = p + S - K / (1 + r)^T
        C = C + S - K / (1 + net_yield)**T
        C_bid = C_bid + S - K / (1 + net_yield)**T
        C_ask = C_ask + S - K / (1 + net_yield)**T

    # print(T.shape)
    # print(K.shape)
    # print(C.shape)
    # print(C_bid.shape)
    # print(C_ask.shape)
    # print(F.shape)
    # print(iv.shape)

    ix_nna = []
    for arr in [F, C, iv]:
        ix_nna += list(np.argwhere(~np.isnan(arr)).flatten())
    ix_nna = list(sorted(set(ix_nna)))
    # print(len(ix_nna))
    T, K, C, C_bid, C_ask, F, iv, Tdt, S = tuple([a[ix_nna] for a in [T, K, C, C_bid, C_ask, F, iv, Tdt, S]])
    # print(len(T))

    if plot:
        plot_prices_iv_repair(T, F, K, C, iv)

    epsilon1 = None
    epsilon2 = []
    for i in range(n_repairs):
        # normalise strikes and prices
        normaliser = constraints.Normalise()
        normaliser.fit(T, K, C, F)
        F1 = normaliser._F
        T1, K1, C1 = normaliser.transform(T, K, C)
        S1 = S[normaliser._order_mask]
        Tdt1 = Tdt[normaliser._order_mask]

        _, _, C1_bid = normaliser.transform(T, K, C_bid)
        _, _, C1_ask = normaliser.transform(T, K, C_ask)

        mat_A, vec_b, _, _ = constraints.detect(T1, K1, C1, verbose=True)

        # repair arbitrage - l1-norm objective
        eps1 = repair.l1(mat_A, vec_b, C1, solver='glpk')
        # eps1cbc = repair.l1(mat_A, vec_b, C1, solver='cbc')
        # eps1ipopt = repair.l1(mat_A, vec_b, C1, solver='ipopt')
        if len(eps1) > 0:
            epsilon1 = eps1

        # repair arbitrage - l1ba objective
        spread_ask = C1_ask - C1
        spread_bid = C1 - C1_bid
        spread = [spread_ask, spread_bid]

        eps2 = repair.l1ba(mat_A, vec_b, C1, spread=spread, solver='glpk')
        if len(eps2) > 0:
            epsilon2 = eps2

        if epsilon1 is None:
            epsilon1 = 0
        K, C = normaliser.inverse_transform(K1, C1 + epsilon1)
        print('-' * 10)

        if len(epsilon2) == 0:
            epsilon2 = np.zeros(len(T))

    # print(max(abs(epsilon1)))
    # print(max(abs(epsilon2)))
    # print(max(abs(epsilon1 + epsilon2)))

    K2, C2 = normaliser.inverse_transform(K1, C1 + epsilon1)
    _, C2_bid = normaliser.inverse_transform(K1, C1_bid + epsilon1)
    _, C2_ask = normaliser.inverse_transform(K1, C1_ask + epsilon1)

    if right == 'put':
        # p = c - S + K / (1 + r)^T
        strike = K1 * F1
        C2 = C2 - S1 + strike / (1 + net_yield) ** T1
        C2_bid = C2_bid - S1 + strike / (1 + net_yield) ** T1
        C2_ask = C2_ask - S1 + strike / (1 + net_yield) ** T1

    iv2 = [calc_iv_for_repair(C2[i], S1[i], calculation_date, right, calculation_date + timedelta(days=T[i] * 365), (K1[i]*F1[i])) for i in range(len(C2))]

    if plot:
        plot_after_repair(T1, F1, K2, C2, iv2, Tdt1, epsilon1, epsilon2, spread_bid, spread_ask, C2_bid, C2_ask)
        plot_prices_iv_repair(T1, F1, K2, C2, iv2)

    df = pd.DataFrame([C2, K2, T1, iv2, C2_bid, C2_ask, Tdt1, S1], index=[nm_mid_close, 'strike', 'tenor', 'mid_iv', 'bid_close', 'ask_close', 'expiry', 'spot']).transpose()
    df['mid_iv'] = df['mid_iv'].mask(df['mid_iv'] <= 0, np.nan)
    return df


def find_loc_every_x_pc(ps_spot: pd.Series, min_r_pc=0.005) -> List:
    loc = []
    p0 = None
    for ix, p in ps_spot.items():
        if p0 is None:
            p0 = p
            loc.append(ix)
            continue
        if abs(p / p0 - 1) > min_r_pc:
            p0 = p
            loc.append(ix)
    return loc


def to_ql_dt(dt) -> ql.Date:
    return ql.Date(dt.day, dt.month, dt.year)


def skew_measure2target_metric(skew_measure: SkewMeasure, right=None) -> Tuple[float, float]:
    if skew_measure == SkewMeasure.ThirdMoment:
        raise NotImplementedError
    elif skew_measure == SkewMeasure.Delta25Delta50 and right == 'call':
        return 0.25, 0.50
    elif skew_measure == SkewMeasure.Delta25Delta50 and right == 'put':
        return -0.25, -0.5
    elif skew_measure == SkewMeasure.Delta25Delta25:
        raise NotImplementedError
    # elif skew_measure == SkewMeasure.M90M100:
    #     return 0.9, 1.0
    # elif skew_measure == SkewMeasure.M90M110:
    #     return 0.9, 0.9
    # elif skew_measure == SkewMeasure.M100M110:
    #     raise NotImplementedError
    #     return 0.9, 1
    else:
        raise ValueError(f'Unknown skew_measure: {skew_measure}')


def enrich_atm_iv_by_right(df, col_nm='atm_iv_by_right'):
    df[col_nm] = None
    for expiry, s_df in df.groupby(level='expiry'):
        for right, ss_df in s_df.groupby(level='right'):
            for ts, sss_df in ss_df.groupby(level='ts'):
                s = sss_df.iloc[0]['spot']
                v_atm_iv = atm_iv(sss_df.loc[ts], expiry, s, right=right)
                df.loc[sss_df.index, col_nm] = v_atm_iv


@lru_cache(maxsize=2**10)
def get_tenor(dt: date, calculation_dt: Union[date, ql.Date]) -> float:
    if isinstance(dt, pd.Timestamp):
        return ((dt.date() - calculation_dt).days + 1) / 365
    elif isinstance(dt, (datetime, date)):
        return ((dt - calculation_dt).days + 1) / 365
    elif isinstance(dt, ql.Date):
        return ((dt.to_date() - calculation_dt).days + 1) / 365
    else:
        raise Exception('Unsupported type')


def timedelta2days(td: timedelta) -> int:
    return td.days


to_days = np.frompyfunc(timedelta2days, 1, 1)


def get_v_tenor(dt: np.ndarray[date], calculation_dt: np.ndarray[Union[date, ql.Date]]) -> np.ndarray:
    if dt is None or len(dt) == 0:
        return np.array([])
    return ((to_days((dt - calculation_dt)) + 1) / 365).astype(float)


def get_v_tenor_from_index(df):
    v_calc_date = np.array(list(map(lambda x: x.date(), df.index.get_level_values('ts').to_pydatetime())))
    return get_v_tenor(df.index.get_level_values('expiry').values, v_calc_date)


def get_v_tenor_dt(tenor: np.ndarray[float], calculation_dt: np.ndarray[Union[date, ql.Date]]) -> List[date]:
    if tenor is None or len(tenor) == 0:
        return []
    return [(calculation_dt + timedelta(days=365 * t)).date() - timedelta(days=1) for t in tenor]


def get_moneyness(strike, spot) -> float:
    return strike / spot


def get_moneyness_fwd(equity: Equity, strike: float, spot: float, tenor: float) -> float:
    """K/Se(r−δ)τ"""
    rate = DiscountRateMarket
    dividend_yield = get_dividend_yield(equity)
    net_yield = rate - dividend_yield

    strike_fwd = strike * np.exp(net_yield * tenor)
    moneyness_fwd = strike_fwd / spot
    return moneyness_fwd


def get_moneyness_fwd_ln(equity: Equity,
                         strike: float | np.ndarray,
                         spot: float | np.ndarray,
                         tenor: float | np.ndarray
                         ) -> float | np.ndarray:
    """k = log(K/Se(r−δ)τ)"""
    return np.log(get_moneyness_fwd(equity, strike, spot, tenor))


def interpolate_pt(interp: NDInterpolatorBase, x: float, y: float) -> np.array:
    _x = np.array([x]) if isinstance(x, (int, float)) else x
    _y = np.array([y]) if isinstance(y, (int, float)) else y
    return interp(_x, _y)


def calc_iv_for_repair(price, priceUnderlying, start, option_right, maturityDate, strike):
    calculationDate = ql.Date(start.day, start.month, start.year)
    ql.Settings.instance().evaluationDate = calculationDate
    optionType = ql.Option.Call if option_right == 'call' else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(optionType, strike)
    eu_exercise = ql.EuropeanExercise(ql.Date(maturityDate.day, maturityDate.month, maturityDate.year))
    # am_exercise = ql.AmericanExercise(self.calculationDate, self.maturityDate)
    dayCount = ql.Actual365Fixed()
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

    underlyingQuote = ql.SimpleQuote(priceUnderlying)
    underlyingQuoteHandle = ql.QuoteHandle(underlyingQuote)
    riskFreeRateQuote = ql.SimpleQuote(0.0)
    riskFreeRateQuoteHandle = ql.QuoteHandle(riskFreeRateQuote)
    dividendRateQuote = ql.SimpleQuote(0.0)
    dividendRateQuoteHandle = ql.QuoteHandle(dividendRateQuote)
    volQuote = ql.SimpleQuote(0)
    volQuoteHandle = ql.QuoteHandle(volQuote)

    qlCalculationDate = calculationDate
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(qlCalculationDate, riskFreeRateQuoteHandle, dayCount))
    dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(qlCalculationDate, dividendRateQuoteHandle, dayCount))
    flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(qlCalculationDate, calendar, volQuoteHandle, dayCount))
    bsmProcess = ql.BlackScholesMertonProcess(underlyingQuoteHandle, dividend_yield, flat_ts, flat_vol_ts)
    binomial_engine = ql.BinomialVanillaEngine(bsmProcess, "crr", 200)
    eu_option = ql.VanillaOption(payoff, eu_exercise)
    eu_option.setPricingEngine(binomial_engine)
    # print(eu_option.impliedVolatility(price, bsmProcess))
    try:
        return eu_option.impliedVolatility(price, bsmProcess)
        #return eu_option.impliedVolatility(price, bsmProcess, 1.0e-4, 100, 1.0e-4, 4)
    except RuntimeError as e:
        print(e)
        return np.nan


# def implied_volatility(price, priceUnderlying, start, option_right, maturityDate, strike):
#     calculationDate = ql.Date(start.day, start.month, start.year)
#     ql.Settings.instance().evaluationDate = calculationDate
#     optionType = ql.Option.Call if option_right == 'call' else ql.Option.Put
#     payoff = ql.PlainVanillaPayoff(optionType, strike)
#     eu_exercise = ql.EuropeanExercise(ql.Date(maturityDate.day, maturityDate.month, maturityDate.year))
#     # am_exercise = ql.AmericanExercise(self.calculationDate, self.maturityDate)
#     dayCount = ql.Actual365Fixed()
#     calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
#
#     underlyingQuote = ql.SimpleQuote(priceUnderlying)
#     underlyingQuoteHandle = ql.QuoteHandle(underlyingQuote)
#     riskFreeRateQuote = ql.SimpleQuote(0.0)
#     riskFreeRateQuoteHandle = ql.QuoteHandle(riskFreeRateQuote)
#     dividendRateQuote = ql.SimpleQuote(0.0)
#     dividendRateQuoteHandle = ql.QuoteHandle(dividendRateQuote)
#     volQuote = ql.SimpleQuote(0)
#     volQuoteHandle = ql.QuoteHandle(volQuote)
#
#     qlCalculationDate = calculationDate
#     flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(qlCalculationDate, riskFreeRateQuoteHandle, dayCount))
#     dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(qlCalculationDate, dividendRateQuoteHandle, dayCount))
#     flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(qlCalculationDate, calendar, volQuoteHandle, dayCount))
#     bsmProcess = ql.BlackScholesMertonProcess(underlyingQuoteHandle, dividend_yield, flat_ts, flat_vol_ts)
#     binomial_engine = ql.BinomialVanillaEngine(bsmProcess, "crr", 200)
#     eu_option = ql.VanillaOption(payoff, eu_exercise)
#     eu_option.setPricingEngine(binomial_engine)
#     # print(eu_option.impliedVolatility(price, bsmProcess))
#     return eu_option.impliedVolatility(price, bsmProcess)


def create_vol_surface_mesh_from_heston_params(today, calendar, spot, v0, kappa, theta, rho, sigma,
                                               rates_curve_handle, dividend_curve_handle,
                                               strikes=np.linspace(40, 200, 161), tenors=np.linspace(0.1, 3, 60)):
    """
    – St Equity spot price, financial index
    – vt Variance.
    – C European call option price.
    – K Strike price.
    – W1,2 Standard Brownian movements.
    – r Interest rate.
    – κ Mean reversion rate.
    – θ Long run variance.
    – v0 Initial variance.
    – σ Volatility of variance.
    – ρ Correlation parameter.
    – t Current date.
    – T Maturity date.
    """
    quote = ql.QuoteHandle(ql.SimpleQuote(spot))

    heston_process = ql.HestonProcess(rates_curve_handle, dividend_curve_handle, quote, v0, kappa, theta, sigma, rho)
    heston_model = ql.HestonModel(heston_process)
    heston_handle = ql.HestonModelHandle(heston_model)
    heston_vol_surface = ql.HestonBlackVolSurface(heston_handle)

    data = []
    for strike in strikes:
        data.append([heston_vol_surface.blackVol(tenor, strike) for tenor in tenors])

    expiration_dates = [calendar.advance(today, ql.Period(int(365 * t), ql.Days)) for t in tenors]
    implied_vols = ql.Matrix(data)
    feller = 2 * kappa * theta - sigma ** 2

    return expiration_dates, strikes, implied_vols, feller


def ts_merge(df1, ps):
    col = ps.name
    ix = df1.index
    df = pd.concat([df1, ps], axis=1).sort_index()
    return df[col].ffill().loc[ix]


def historical_volatility(ps: pd.Series, window: pd.Timedelta = pd.Timedelta(days=3), sampling_period: pd.Timedelta = pd.Timedelta(minutes=5)):
    """
    _volatility = std * (decimal)Math.Sqrt(252.0 * _samplesPerDay);
    :param ps: A series of prices.
    :param window: The window size in days.
    :param sampling_period_sec: The sampling period in seconds.
    :return: A series of historical volatilities.
    """
    window = pd.Timedelta(days=10)
    ps_resampled = ps.resample(sampling_period, closed='right').last().ffill()
    ix_time = ps_resampled.index.time
    ps_resampled = ps_resampled.loc[ps_resampled.index[(ix_time >= time(9, 30)) & (ix_time <= time(16, 0))]]
    # exclude holidays
    ps_std = ps_resampled.pct_change().rolling(window=window).std()
    ps_std.name = 'std'
    ps_annualize = ps_std.groupby(ps_std.index.date).count().apply(lambda x: np.sqrt(252 * x))
    ps_annualize.name = 'f_annualize'
    ready_date = (ps_std.index[0] + window).date()
    ps_std = pd.DataFrame(ps_std.loc[ready_date:])
    ps_std['date'] = ps_std.index.date

    # Merge samples_per_day to ps_std pandas series based on date in both indices
    df = pd.merge(ps_std, ps_annualize, left_on='date', right_index=True)
    hv = df['std'] * df['f_annualize']
    hv.name = 'hv'
    return ts_merge(df, hv)


def load(sym: Equity | str, start, end, n=1, resolution=Resolution.minute):
    equity = Equity(sym.lower()) if isinstance(sym, str) else sym
    contracts = client.central_volatility_contracts(equity, start, end, n=n)
    # contracts = client.get_contracts(sym, None, (14, 18))
    if resolution in (Resolution.hour, Resolution.daily):
        trades = client.history([equity], start, end, resolution, TickType.trade, SecurityType.equity)
    else:
        trades = client.history([equity], start, end, resolution, TickType.quote, SecurityType.equity)
    quotes = client.history(list(chain(*contracts.values())), start, end, resolution, TickType.quote, SecurityType.option)
    return trades, quotes, contracts


def contract_lower(contracts, spot, dt, pc):
    return list(sorted([c for c in contracts if pc == c.right and c.expiry >= dt + timedelta(days=14) and (float(c.strike) - spot) < 0],
                       key=lambda x: x.expiry.isoformat() + str(1 / x.strike)))[0]


def contract_upper(contracts, spot, dt, pc):
    return list(sorted([c for c in contracts if pc == c.right and c.expiry >= dt + timedelta(days=14) and (float(c.strike) - spot) > 0],
                       key=lambda x: x.expiry.isoformat() + str(x.strike)))[0]


# Define the rolling cone function
def rolling_cone(implied_volatility, lookback_period, confidence_levels):
    # Compute the rolling standard deviation of returns and implied volatilities
    rolling_iv_mean = pd.Series(implied_volatility).rolling(window=lookback_period, min_periods=0).mean()
    rolling_iv_std = pd.Series(implied_volatility).rolling(window=lookback_period, min_periods=0).std()

    # Compute the rolling volatility cones for each confidence level
    cones = []
    for confidence_level in confidence_levels:
        # Compute the z-score based on the rolling population of implied volatilities
        # z_score = (rolling_implied_volatility - rolling_implied_volatility_mean) / rolling_implied_volatility_std

        # Compute the confidence interval based on the confidence level
        lower_bound = -confidence_level * rolling_iv_std + rolling_iv_mean
        upper_bound = confidence_level * rolling_iv_std + rolling_iv_mean

        # Append the lower and upper bounds to the cones list
        cones.append((lower_bound, upper_bound, implied_volatility))

    # Return the cones
    return cones


def atm_iv_old_method(trades, quotes, optionContracts, n=1, resolution='60min'):
    """
    consider adding outlier remover. cut any beyond 3 sigma
    """
    mat_df = iv_of_expiry(optionContracts, trades, quotes, resolution=resolution)
    df_strike_iv = pd.DataFrame({float(k.split('_')[3]) / 10_000: (df['mid_iv']) for k, df in mat_df.items()}).fillna(method='ffill')
    df_strike_iv = df_strike_iv.sort_index(axis=1)
    ps_t = pd.concat(df['mid_close_underlying'] for df in mat_df.values()).dropna()
    ps_t = ps_t[~ps_t.index.duplicated(keep='first')].sort_index()

    strikes = np.array([float(el) for el in df_strike_iv.columns])
    df_strike_distance = client.strike_to_atm_distance(ps_t, strikes)
    intersect_index = df_strike_iv.index.intersection(df_strike_distance.index)

    df_strike_iv = df_strike_iv.loc[intersect_index]
    df_strike_distance = df_strike_distance.loc[intersect_index]
    df_strike_distance = df_strike_distance.sort_index(axis=1)
    assert df_strike_iv.shape == df_strike_distance.shape, 'require identical shapes. timeseries and strikes'
    assert df_strike_distance.columns.to_list() == df_strike_iv.columns.to_list(), 'require identical strikes'
    strike_levels = list(range(-n, n + 1))

    df_atm_iv = df_strike_iv * ((df_strike_distance.isin(strike_levels)) * 1)
    count_ivs = (df_atm_iv != 0).sum(axis=1)

    return pd.Series(df_atm_iv.sum(axis=1) / count_ivs, index=df_strike_iv.index)


def aewma(vec: pd.Series | np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """adaptive exponentially weighted moving average. alpha-1, no smoothing always latest value. alpha 0 - max smooth always first value"""
    ewmas = np.zeros_like(vec)
    vec = vec.values if isinstance(vec, pd.Series) else vec
    for i, v in enumerate(vec):
        if i == 0:
            ewmas[i] = previous_ewma = v
        else:
            previous_ewma = ewmas[i - 1]
        ewmas[i] = alpha * v + (1 - alpha) * previous_ewma
        eps = abs(v - ewmas[i])
        alpha = (1 - gamma) * alpha + gamma * (eps / (eps + previous_ewma)) if gamma < 0 else alpha
    return ewmas


def get_ivs_file(ps, surf_otm, surf_itm):
    try:
        if ps['is_otm'] == True:
            return surf_otm.loc[(str(ps['expiry']), str(ps['time_5'])), str(ps['bin']).split('.')[0]]
        else:
            return surf_itm.loc[(str(ps['expiry']), str(ps['time_5'])), str(ps['bin']).split('.')[0]]
    except KeyError:
        return np.nan


def get_ivs_file2(ps, surf_call, surf_put):
    try:
        if ps['right'] == 'call':
            return surf_call.loc[(str(ps['expiry']), str(ps['time_5'])), str(ps['bin']).split('.')[0]]
        else:
            return surf_put.loc[(str(ps['expiry']), str(ps['time_5'])), str(ps['bin']).split('.')[0]]
    except KeyError:
        return np.nan


def iv_surface_normed(ivs, is_otm):
    """
    ivs for a particular maturity date
    Put Call Parity: e.g.: 120 % Put == 80% Call
    """
    skew = []
    for contract_str, df_iv in ivs.items():
        df = df_iv.copy()
        df['right'] = contract_str.split('_')[2]
        strike = float(contract_str.split('_')[3]) / 10_000
        df['strike'] = strike
        df['strike_%_price'] = 100 * strike / df['mid_price_underlying']
        df['otm'] = False
        ix_otm = df[((df['right'] == 'call') & (df['strike_%_price'] <= 100)) | ((df['right'] == 'put') & (df['strike_%_price'] >= 100))].index
        df.loc[ix_otm, 'otm'] = True
        if is_otm:
            df = df[df['otm'] == True]
        else:
            df = df[df['otm'] == False]
        skew.append(df)

    df = pd.concat(skew).sort_index()
    df = df[['strike_%_price', 'strike', 'bid_iv', 'ask_iv']].reset_index()
    df = df[~df.duplicated(['time', 'strike_%_price'])].set_index('time')
    df['bin'] = df['strike_%_price'].round(0)

    s_bin = df[['strike_%_price', 'strike']].pivot(columns=['strike'])
    s_bin = s_bin.resample(pd.Timedelta(minutes=5)).last()
    # s_bin = s_bin.fillna(method='ffill', limit=12)
    # s_bin = s_bin[(time(9, 30) <= s_bin.index.time) & (s_bin.index.time <= time(16, 0))]
    s_bin.columns = s_bin.columns.get_level_values(1).values
    s_strike_pct = s_bin.iloc[:, 1:]

    surf_bid = map_strike_to_strike_pct(s_side(df, 'bid'), s_strike_pct)
    surf_ask = map_strike_to_strike_pct(s_side(df, 'ask'), s_strike_pct)

    return surf_bid, surf_ask


def smooth_wall(ps):
    _ps = ps.copy()
    ixclean = _ps.dropna().index
    if len(ixclean) < 3:
        return _ps
    _ps.loc[ixclean] = savgol_filter(_ps.loc[ixclean], len(ixclean), 2)
    return _ps


def smooth(Cb, alpha=0.005, gamma=0, smooth_wall=True):
    # Smoothening along tenor
    df = Cb.copy()
    for c in df:
        ix = df[c].dropna().index
        df.loc[ix, c] = aewma(df.loc[ix, c].values, alpha, gamma)

    # Smoothening along smile
    if smooth_wall:
        df = df.apply(smooth_wall, axis=0)
    return df


def map_strike_to_strike_pct(s_side, s_strike_pct):
    x = np.unique(s_strike_pct.values)
    x = x[~np.isnan(x)]
    x = np.sort(x)
    bins = sorted(np.unique(np.round(x)))

    Cb = pd.DataFrame(index=s_side.index, columns=x).astype(float)
    for c in bins:
        if c not in Cb:
            Cb[c] = np.nan
    Cb = Cb.sort_index(axis=1)

    for c in s_strike_pct:
        ps_side = s_side[c].dropna().values
        for i, (ts, strikepct) in enumerate(s_strike_pct[c].dropna().items()):
            Cb.loc[ts, strikepct] = ps_side[i]
    Cb = Cb.mask(Cb == 0, np.nan)
    Cb = Cb.interpolate(method='index', axis=1, limit_area='inside')  # not equally spaced - method: index
    Cb = Cb.drop([c for c in Cb.columns if c not in bins], axis=1)

    # Cb = smooth(Cb)

    return Cb


def sq(i):
    return i ** i


def test_mp_task(n):
    with multiprocessing.Pool(min(multiprocessing.cpu_count(), n)) as pool:
        results = pool.map(sq, range(n))
    return results


fPos2profile = {
    'Delta': lambda df, change: reduce(lambda res, ps: res + ps['Greeks1.Delta'] * ps['Mid1Underlying'] * change * ps['Multiplier'] * ps['Quantity'],
                                       (t[1] for t in df.iterrows()), 0),
    'Gamma': lambda df, change: reduce(
        lambda res, ps: res + 0.5 * ps['Greeks1.Gamma'] * (ps['Mid1Underlying'] * change) ** 2 * ps['Multiplier'] * ps['Quantity'],
        (t[1] for t in df.iterrows()), 0),
    'Speed': lambda df, change: reduce(
        lambda res, ps: res + (1 / 6) * ps['Greeks1.DS3'] * (ps['Mid1Underlying'] * change) ** 3 * ps['Multiplier'] * ps['Quantity'],
        (t[1] for t in df.iterrows()), 0),
    'DeltaIVdS': lambda df, change: reduce(
        lambda res, ps: res + ps['Greeks1.Vega'] * ps['SurfaceIVdS'] * change * ps['Mid1Underlying'] * change * ps['Multiplier'] * ps['Quantity'],
        (t[1] for t in df.iterrows()), 0),
    'Vega': lambda df, change: reduce(lambda res, ps: res + ps['Greeks1.Vega'] * change * ps['Multiplier'] * ps['Quantity'], (t[1] for t in df.iterrows()), 0),
    'Volga': lambda df, change: reduce(lambda res, ps: res + ps['Greeks1.DIV2'] * change ** 2 * ps['Multiplier'] * ps['Quantity'],
                                       (t[1] for t in df.iterrows()), 0),
    'Vanna': lambda df, change: reduce(lambda res, ps: res + ps['Greeks1.Vanna'] * ps['Mid1Underlying'] * change * ps['Multiplier'] * ps['Quantity'],
                                       (t[1] for t in df.iterrows()), 0),
}
fPos2profile['Vomma'] = fPos2profile['Volga']
fPos2profile['Greeks1.Delta'] = fPos2profile['Delta']
fPos2profile['Greeks1.Gamma'] = fPos2profile['Gamma']
fPos2profile['Greeks1.DS3'] = fPos2profile['Speed']
fPos2profile['Greeks1.Vega'] = fPos2profile['Vega']
fPos2profile['Greeks1.DIV2'] = fPos2profile['Volga']
fPos2profile['Greeks1.Vanna'] = fPos2profile['Vanna']


def position2riskPnlProfile(df, i_range=(-5, 5)):
    rows = []
    risk_metrics = list(fPos2profile.keys())
    for ix, dfg in df.groupby(['Ts1', 'UnderlyingSymbol']):
        for metric in risk_metrics:
            dims = [ix[0], ix[1], metric]
            risks = [fPos2profile[metric](dfg, i / 100) for i in range(*i_range)]
            rows.append(dims + risks)
    return pd.DataFrame(rows, columns=['Time', 'Underlying', 'Metric'] + [f'{i}' for i in range(*i_range)])


def s_side(dfa, side='ask'):
    s_ask = dfa[[f'{side}_iv', 'strike']].pivot(columns=['strike'])
    s_ask = s_ask.resample(pd.Timedelta(minutes=5)).last()
    s_ask = s_ask.fillna(method='ffill', limit=12)
    s_ask = s_ask[(time(9, 30) <= s_ask.index.time) & (s_ask.index.time <= time(16, 0))]
    s_ask.columns = s_ask.columns.get_level_values(1).values
    s_ask = s_ask.iloc[:, 1:]
    return s_ask


def make_eu_option(optionType: ql.Option, strike, maturityDate) -> ql.VanillaOption:
    eu_exercise = ql.EuropeanExercise(ql.Date(maturityDate.day, maturityDate.month, maturityDate.year))
    payoff = ql.PlainVanillaPayoff(optionType, strike)
    return ql.VanillaOption(payoff, eu_exercise)


def make_am_option(optionType, strike, maturityDate, calculation_date) -> ql.Option:
    am_exercise = ql.AmericanExercise(calculation_date, ql.Date(maturityDate.day, maturityDate.month, maturityDate.year))
    payoff = ql.PlainVanillaPayoff(optionType, strike)
    return ql.VanillaOption(payoff, am_exercise)


def set_ql_calculation_date(calculation_date_ql: Union[ql.Date, date]) -> ql.Date:
    if isinstance(calculation_date_ql, date):
        _calculation_date_ql = ql.Date(calculation_date_ql.day, calculation_date_ql.month, calculation_date_ql.year)
    else:
        _calculation_date_ql = calculation_date_ql
    if ql.Settings.instance().evaluationDate != _calculation_date_ql:
        ql.Settings.instance().evaluationDate = _calculation_date_ql
    return _calculation_date_ql


def tenor2date(tenor, calculation_date) -> date:
    return tenor if isinstance(tenor, date) else calculation_date + timedelta(days=int(tenor * 365))


def prepare_eu_option(strike, tenor, spot, put_call, vol, calculation_date, calendar, day_count, yield_ts, dividend_ts):
    calculation_date_ql = set_ql_calculation_date(calculation_date)
    expiry = tenor2date(tenor, calculation_date)
    spot_quote = ql.QuoteHandle(ql.SimpleQuote(spot))
    _put_call = str2ql_option_right(put_call)
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date_ql, calendar, ql.QuoteHandle(ql.SimpleQuote(vol)), day_count))
    eu_option = make_eu_option(_put_call, strike, expiry)
    bsm_process = ql.BlackScholesMertonProcess(spot_quote, dividend_ts, yield_ts, vol_ts)
    analytical_engine = ql.AnalyticEuropeanEngine(bsm_process)
    eu_option.setPricingEngine(analytical_engine)
    return eu_option


def str2ql_option_right(right: str | OptionRight):
    return {
        'call': ql.Option.Call,
        'put': ql.Option.Put
    }[right]


def delta_bsm(*args, **kwargs):
    return greek_bsm(GreeksEuOption.delta, *args, **kwargs)


def vega_bsm(*args, **kwargs):
    return greek_bsm(GreeksEuOption.vega, *args, **kwargs)


def greek_bsm(greek: str, strike, tenor, spot, put_call, vol, calculation_date, calendar, day_count, rate, dividend) -> float:
    if np.isnan(vol):
        return np.nan
    elif vol == 0:
        return 0

    yield_ts = ql.YieldTermStructureHandle(ql.FlatForward(to_ql_dt(calculation_date), rate, day_count))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(to_ql_dt(calculation_date), dividend, day_count))

    if greek in (GreeksEuOption.delta, GreeksEuOption.theta, GreeksEuOption.vega):
        eu_option = prepare_eu_option(strike, tenor, spot, put_call, vol, calculation_date, calendar, day_count, yield_ts, dividend_ts)
        return getattr(eu_option, greek)()
    elif greek == GreeksEuOption.gamma:
        return gamma_fd(strike, tenor, spot, put_call, vol, calculation_date, calendar, day_count, yield_ts, dividend_ts)
    else:
        # FD methods...
        raise ValueError(f'Unknown greek: {greek}')


def gamma_fd(strike, tenor, spot, put_call, vol, calculation_date, calendar, day_count, yield_ts, dividend_ts, step_pct=0.01) -> float:
    eu_option_lt = prepare_eu_option(strike, tenor, spot * (1 - step_pct), put_call, vol, calculation_date, calendar, day_count, yield_ts, dividend_ts)
    delta_lt = eu_option_lt.delta()
    if np.isnan(delta_lt) or delta_lt == 0:
        return np.nan

    eu_option_gt = prepare_eu_option(strike, tenor, spot * (1 + step_pct), put_call, vol, calculation_date, calendar, day_count, yield_ts, dividend_ts)
    delta_gt = eu_option_gt.delta()
    if np.isnan(delta_gt) or delta_lt == 0:
        return np.nan

    return (delta_gt - delta_lt) / (2 * spot * step_pct)


def intrinsic_value(strike, priceUnderlying, put_call):
    if put_call == 'call':
        return max(priceUnderlying - strike, 0)
    elif put_call == 'put':
        return max(strike - priceUnderlying, 0)


def ps2intrinsic_value(ps: pd.Series, spot_column='spot') -> float:
    ts, expiry, strike, right = unpack_mi_df_index(ps)
    return intrinsic_value(float(strike), ps[spot_column], right)


def npv(iv: float, strike: float, tenor, spot, put_call, calculation_date, calendar, day_count, rate, dividend) -> float:
    expiry = tenor2date(tenor, calculation_date)
    if calculation_date == expiry:
        return intrinsic_value(strike, spot, put_call)
    else:
        yield_ts = ql.YieldTermStructureHandle(ql.FlatForward(to_ql_dt(calculation_date), rate, day_count))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(to_ql_dt(calculation_date), dividend, day_count))
        eu_option = prepare_eu_option(strike, tenor, spot, put_call, iv, calculation_date, calendar, day_count, yield_ts, dividend_ts)
        return eu_option.NPV()


def implied_volatility(price: float, strike: float, tenor, spot, put_call, calculation_date, calendar, day_count, rate, dividend) -> float:
    expiry = tenor2date(tenor, calculation_date)
    if expiry < calculation_date:
        return 0
    if expiry == calculation_date:
        # RuntimeError: option expired
        # But, by the time it's exercised / assigned - option may still end up OTM/ITM. Therefore, would want an IV.
        # THat's why setting calc date to expiry-1
        calculation_date = calculation_date - timedelta(days=1)

    yield_ts = ql.YieldTermStructureHandle(ql.FlatForward(to_ql_dt(calculation_date), rate, day_count))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(to_ql_dt(calculation_date), dividend, day_count))

    calculation_date_ql = set_ql_calculation_date(calculation_date)
    eu_option = prepare_eu_option(strike, tenor, spot, put_call, 0, calculation_date, calendar, day_count, yield_ts, dividend_ts)
    spot_quote = ql.QuoteHandle(ql.SimpleQuote(spot))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date_ql, calendar, ql.QuoteHandle(ql.SimpleQuote(0)), day_count))

    bsm_process = ql.BlackScholesMertonProcess(spot_quote, dividend_ts, yield_ts, vol_ts)
    iv_val = get_iv(eu_option, price, bsm_process)
    extrinsic_value = price - intrinsic_value(strike, spot, put_call)
    if iv_val == 0 and extrinsic_value > 0:
        # Dealing with very high volatilities or still zero if extrinsic value is close to zero.
        iv_val = get_iv(eu_option, price, bsm_process, max_iterations=10000, max_vol=10)
        if iv_val == 0 and extrinsic_value > spot * 0.01:
            iv_val = 5
    return iv_val


def get_iv(eu_option, price, bsm_process, accuracy=0.0001, max_iterations=100, min_vol=1.0e-4, max_vol=4.0):
    try:
        return eu_option.impliedVolatility(price, bsm_process, accuracy, max_iterations, min_vol, max_vol)
    except RuntimeError as e:
        if 'root not bracketed' in str(e):
            return 0
        raise e


def unpack_mi_df_index(ps: pd.Series) -> tuple:
    ts = ps.name[0]
    expiry = ps.name[1]
    strike = ps.name[2]
    right = ps.name[3]
    return ts, expiry, strike, right


def ps_index2option_contract(ps: pd.Series, equity: Equity) -> OptionContract:
    if len(ps.name) == 3:
        expiry, strike, right = ps.name
    elif len(ps.name) == 4:
        ts, expiry, strike, right = ps.name
    else:
        raise ValueError(f'Invalid index: {ps.name}')

    return OptionContract('', equity.underlying_symbol, expiry, strike, right)

# delete soon
# def ps2delta(ps: pd.Series, iv_col: str, calendar, day_count, rate, dividend, spot_column='spot') -> float:
#     ts, expiry, strike, right = unpack_mi_df_index(ps)
#     return greek_bsm(GreeksEuOption.delta, float(strike), expiry, ps[spot_column], right, ps[iv_col], ts.date(), calendar, day_count, rate, dividend)
#
#
# def ps2gamma(ps: pd.Series, iv_col: str, calendar, day_count, rate, dividend, spot_column='spot') -> float:
#     ts, expiry, strike, right = unpack_mi_df_index(ps)
#     return greek_bsm(GreeksEuOption.gamma, float(strike), expiry, ps[spot_column], right, ps[iv_col], ts.date(), calendar, day_count, rate, dividend)
#
#
# def ps2vega(ps: pd.Series, iv_col: str, calendar, day_count, rate, dividend, spot_column='spot') -> float:
#     ts, expiry, strike, right = unpack_mi_df_index(ps)
#     return greek_bsm(GreeksEuOption.vega, float(strike), expiry, ps[spot_column], right, ps[iv_col], ts.date(), calendar, day_count, rate, dividend)
#
# def ps2theta(ps: pd.Series, iv_col: str, calendar, day_count, rate, dividend, spot_column='spot') -> float:
#     ts, expiry, strike, right = unpack_mi_df_index(ps)
#     return greek_bsm(GreeksEuOption.theta, float(strike), expiry, ps[spot_column], right, ps[iv_col], ts.date(), calendar, day_count, rate, dividend)
#
#
# def ps2iv(ps: pd.Series, price_col, calendar, day_count, rate, dividend, spot_column='spot') -> float:
#     ts, expiry, strike, right = unpack_mi_df_index(ps)
#     return implied_volatility(ps[price_col], float(strike), expiry, ps[spot_column], right, ts.date(), calendar, day_count, rate, dividend)


def df2iv(df: pd.DataFrame, price_col_nm: str, rate: float, dividend_yield: float, spot_col_nm: str = 'spot', tenor_col_nm='tenor', na2zero=True) -> np.ndarray:
    v: np.ndarray = get_v_iv(
        p=df[price_col_nm].values,
        s=df[spot_col_nm].values,
        k=df.index.get_level_values('strike').astype(float).values,
        t=df[tenor_col_nm].values,
        r=rate,
        right=df.index.get_level_values('right').map({OptionRight.call: 'c', OptionRight.put: 'p'}).values,
        q=dividend_yield,
        )
    if na2zero:
        # This should be handled better by first checking which prices are below intrinsic value and only send remaining values here.
        v[np.isnan(v)] = 0
    return v


def get_v_iv(p: np.ndarray, s: np.ndarray, k: np.ndarray, t: np.ndarray, r: float, right: np.ndarray, q: float) -> np.ndarray:
    return py_vollib_vectorized.vectorized_implied_volatility(p, s, k, t, r, right, q=q, model='black_scholes_merton', return_as='numpy')


def ps2npv(ps: pd.Series, iv_col: str, calendar, day_count, rate, dividend, spot_column='spot') -> float:
    ts, expiry, strike, right = unpack_mi_df_index(ps)
    return npv(ps[iv_col], float(strike), expiry, ps[spot_column], right, ts.date(), calendar, day_count, rate, dividend)


def spot_from_df_equity_into_options(option_frame: OptionFrame):
    # looks terrible, why not a join?
    option_frame.df_options['spot'] = None
    for ts, sub_df in option_frame.df_options.groupby(level='ts'):
        option_frame.df_options.loc[(ts, slice(None), slice(None), slice(None)), 'spot'] = option_frame.df_equity.loc[ts, 'close']
    option_frame.df_options['spot'] = option_frame.df_options['spot'].astype(float)


def val_from_df(df, expiry, strike, right, col_nm):
    if 'ts' in df.index.names:
        return df[col_nm].loc[(slice(None), expiry, strike, right)].iloc[-1]
    return df[col_nm].loc[(expiry, strike, right)]


def year_quarter(dt: date) -> str:
    quarter = (dt.month - 1) // 3 + 1
    return f'{dt.strftime("%y")}Q{quarter}'


def moneyness_iv(df, moneyness, expiry, fwd_s) -> np.ndarray:
    atmny_strike = fwd_s * moneyness
    strikes = pd.Series(np.unique(df.loc[(expiry, slice(None))].index.get_level_values('strike')))
    strikes_at_moneyness = strikes.iloc[(strikes.astype(float) / fwd_s - moneyness).abs().sort_values().head(2).index].values
    iv_at_moneyness = df.loc[(expiry, strikes_at_moneyness)].values if isinstance(df, pd.Series) else df.loc[(expiry, strikes_at_moneyness), 'mid_iv'].values
    iv = pd.Series(iv_at_moneyness, index=strikes_at_moneyness).sort_index()
    if len(iv) == 1 or np.nan in iv.values or 0 in iv.values:
        logger.error(f'No IV for {moneyness} moneyness.')
        return np.nan
    return np.interp(atmny_strike, iv.index, iv.values)


def atm_iv(df, expiry, s, right=None) -> np.ndarray:
    if right:
        return moneyness_iv(df.loc[(expiry, slice(None), right), 'mid_iv'], 1, expiry, s)
    else:
        call_iv = moneyness_iv(df.loc[(expiry, slice(None), 'call'), 'mid_iv'], 1, expiry, s)
        put_iv = moneyness_iv(df.loc[(expiry, slice(None), 'put'), 'mid_iv'], 1, expiry, s)
        return (call_iv + put_iv) / 2


def is_holiday(dt: date):
    return dt.weekday() in (5, 6) or dt in get_market_hours_holidays()


def earnings_download_dates_start_end(release_date: date, biz_days_prior=3, biz_days_after=2) -> Tuple[date, date]:
    holidays = get_market_hours_holidays()

    start = None
    end = None

    prior_days = 0
    for i in range(30):
        dt = release_date - timedelta(days=i)
        if dt not in holidays and dt.weekday() < 5:
            prior_days += 1
        if prior_days == biz_days_prior:
            start = dt
            break

    after_days = 0
    for i in range(1, 30):
        dt = release_date + timedelta(days=i)
        if dt not in holidays and dt.weekday() < 5:
            after_days += 1
        if after_days == biz_days_after:
            end = dt
            break

    end = min(end, date.today() - timedelta(days=1))
    return start, end


def earnings_download_dates(release_date: date, biz_days_prior=3, biz_days_after=2) -> List[date]:
    return [dt.date() for dt in pd.date_range(*earnings_download_dates_start_end(release_date, biz_days_prior, biz_days_after)) if not is_holiday(dt)]


def add_trade_days(dt: date, n: int) -> date:
    holidays = get_market_hours_holidays()
    while n != 0:
        dt += timedelta(days=int(np.sign(n)))
        if dt.weekday() < 5 and dt not in holidays:
            n -= int(np.sign(n))
    return dt


def n_trade_days(dt1: date, dt2: date) -> int:
    """Absolute number of trade dates between dt1 and dt2 including each"""
    holidays = get_market_hours_holidays()
    n = 0
    for dt in pd.date_range(min((dt1, dt2)), max((dt1, dt2))):
        if dt.weekday() < 5 and dt.date() not in holidays:
            n += 1
    return n


def trade_days_between_dates(dt1: date, dt2: date) -> List[date]:
    holidays = get_market_hours_holidays()
    return [dt.date() for dt in pd.date_range(min((dt1, dt2)), max((dt1, dt2))) if dt.weekday() < 5 and dt.date() not in holidays]


def apply_ds_ret_weights(m_dnlv01, f_weight_ds, cfg):
    n_ds = len(cfg.v_ds_ret)
    assert m_dnlv01.shape[0] == n_ds
    v_f_weight_ds = np.array([f_weight_ds(ds_ret) for ds_ret in cfg.v_ds_ret])
    for k in range(n_ds):
        m_dnlv01[k, :] *= v_f_weight_ds[k]
    return m_dnlv01


# def get_spread_modified_iv_caches(cache_iv0, cache_iv1, calcDate0: date, cfg):
#     """
#     These are too many configs and should be part of a dedicated model.
#     The idea is that the higher the liquidity, the more spread I earn. Or differently, I won't get filled as mid IV for far expirations.
#
#     Liquidity is sharply increasing for the nearest expiration and week after, rest low, especially after tenor 0.5
#     The assumptions about spread pct needs to be removed. Problem with using bid IV, often 0, creating a huge spread.
#     The hard cut on tenor also to be replaced.
#     """
#     # # Pay spread for large tenor:
#     cache_iv1_buy = defaultdict(dict)
#     cache_iv1_sell = defaultdict(dict)
#     cache_iv0_buy = defaultdict(dict)
#     cache_iv0_sell = defaultdict(dict)
#     for o, dct_ds_iv in cache_iv1.items():
#         if isinstance(o, Equity):
#             cache_iv1_sell[o] = {1: 0}
#             cache_iv1_buy[o] = {1: 0}
#             continue
#         tenor = (o.expiry - calcDate0).days / 365
#         if tenor > 0.5:
#             # depends on buy / sell
#             spread_pct_of_iv = 0.06
#             # Assume I pay 50% of spread for long tenors.
#             cache_iv1_buy[o] = {ds: cache_iv1[o][ds] * (1 + spread_pct_of_iv / 2) for ds in cfg.v_ds_ret}
#             cache_iv1_sell[o] = {ds: cache_iv1[o][ds] * (1 - spread_pct_of_iv / 2) for ds in cfg.v_ds_ret}
#         else:
#             spread_pct_of_iv = 0.02
#             # Assume I earn 50% of spread for short tenors.
#             cache_iv1_buy[o] = {ds: cache_iv1[o][ds] * (1 - spread_pct_of_iv / 2) for ds in cfg.v_ds_ret}
#             cache_iv1_sell[o] = {ds: cache_iv1[o][ds] * (1 + spread_pct_of_iv / 2) for ds in cfg.v_ds_ret}
#
#     for o, dct_ds_iv in cache_iv0.items():
#         if isinstance(o, Equity):
#             cache_iv0_sell[o] = {1: 0}
#             cache_iv0_buy[o] = {1: 0}
#             continue
#         tenor = (o.expiry - calcDate0).days / 365
#         if tenor > 0.5:
#             # depends on buy / sell
#             spread_pct_of_iv = 0.06
#             # Assume I pay 50% of spread for long tenors.
#             cache_iv0_buy[o] = {ds: cache_iv0[o][ds] * (1 + spread_pct_of_iv / 2) for ds in [1]}
#             cache_iv0_sell[o] = {ds: cache_iv0[o][ds] * (1 - spread_pct_of_iv / 2) for ds in [1]}
#         else:
#             spread_pct_of_iv = 0.02
#             # Assume I earn 50% of spread for short tenors.
#             cache_iv0_buy[o] = {ds: cache_iv0[o][ds] * (1 - spread_pct_of_iv / 2) for ds in [1]}
#             cache_iv0_sell[o] = {ds: cache_iv0[o][ds] * (1 + spread_pct_of_iv / 2) for ds in [1]}
#
#     return cache_iv0_buy, cache_iv0_sell, cache_iv1_buy, cache_iv1_sell


def ps2mid_iv(ps: pd.Series) -> pd.Series:
    return (ps['ask_iv'] + ps['bid_iv']) / 2


class ATMHelper:
    def __init__(self, expiry: datetime.date, strike: float, calculation_date: datetime.date, net_yield: float):
        self.expiry = expiry
        self.strike = strike
        self.calculation_date = calculation_date
        self.net_yield = net_yield

    def PV_K(self):
        # PV(K) = K * exp(-(r - q) * (T - t))
        # net_yield = rate - dividend_yield = r - q
        # T-t = DTE/365
        return self.strike * np.exp(-self.net_yield * (self.expiry - self.calculation_date).days / 365)


def quotes2multi_index_df(quotes: Dict[OptionContract, pd.DataFrame]) -> pd.DataFrame:
    dfs = []
    for k, v in quotes.items():
        if v.empty:
            continue
        v['expiry'] = k.expiry
        v['strike'] = k.strike
        v['right'] = k.right
        dfs.append(v.reset_index().rename(columns={'index': 'ts'}).set_index(['ts', 'expiry', 'strike', 'right']))
    return pd.concat(dfs)


def get_dividend_yield(equity: str | Equity, ts: datetime.date = None) -> float:
    key = (equity if isinstance(equity, str) else equity.symbol).upper()
    if key not in DividendYield:
        traceback_str = ''.join(traceback.format_stack())
        warning(f'No dividend yield for {key}. Defaulting to 0. {traceback_str}')
    return DividendYield.get(key, 0)


def spread_pc(moneyness, tenor):
    return 1 / (abs(moneyness - 1) + 1) - (tenor / 2) ** 0.3


def df2atm_iv(df: pd.DataFrame, ps_spot, net_yield: float, min_dte=14) -> pd.Series:
    """
    Weighted average of atm strikes for call & put. Expiry is tricky: too early and it slopes up too much. Cannot pick a single expiry
    as DTE would vary too much leaving DTE variance behind. If fixing DTE at 14, just weighted average of closest expiries or more?
    Simplest: Just take weighted average of all expiries... While it's ATM, it's not as useful as we're interested in getting a market estimate/indicator of
    future volatility.
    """
    v_ts = []
    v_atm_iv = []

    # Loop through time
    for ts, sub_df in df.groupby(level='ts'):
        v_ts.append(ts)
        spot = ps_spot.loc[ts]['close']
        sub_v_atm_iv = []

        for expiry, sub_sub_df in sub_df.groupby(level='expiry'):
            if (expiry - ts.date()).days <= min_dte:
                continue
            # ATM strike K when K ~= FV(S) = S * exp((r - q) * (T - t)); Same as using S ~= PV(K) = K * exp(-(r - q) * (T - t))

            pv_strikes = {strike: ATMHelper(expiry, float(strike), ts.date(), net_yield).PV_K() for strike in sub_sub_df.index.get_level_values('strike').unique()}
            pv_strikes_minus_spot = {s: pv - spot for s, pv in pv_strikes.items()}
            try:
                strike_low = pd.Series({s: v for s, v in pv_strikes_minus_spot.items() if v < 0}).idxmax()
                strike_high = pd.Series({s: v for s, v in pv_strikes_minus_spot.items() if v > 0}).idxmin()
            except ValueError as e:
                print(e)
                continue

            for right in ['call', 'put']:
                try:
                    iv_low = sub_sub_df.loc[(ts, expiry, strike_low, right), 'mid_iv']
                    iv_high = sub_sub_df.loc[(ts, expiry, strike_high, right), 'mid_iv']
                except KeyError as e:
                    print(e)
                    continue

                # Linearly interpolate ATM
                iv_atm = iv_low + (iv_high - iv_low) * (spot - float(strike_low)) / (float(strike_high) - float(strike_low))
                if iv_atm > 0:
                    sub_v_atm_iv.append(iv_atm)
            # Normalize IVs across expiries to today ?

        if sub_v_atm_iv:
            v_atm_iv.append(np.mean(sub_v_atm_iv))
        else:
            v_atm_iv.append(np.nan)

    return pd.Series(v_atm_iv, index=v_ts)


@lru_cache(maxsize=1)
def load_market_hours():
    with open(Paths.path_market_hours_database, 'r') as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_market_hours(market="Equity-usa-[*]"):
    return load_market_hours()["entries"][market]


@lru_cache(maxsize=1)
def get_market_hours_holidays(market="Equity-usa-[*]") -> List[date]:
    return [datetime.strptime(d, '%m/%d/%Y').date() for d in get_market_hours(market)["holidays"]]


@lru_cache(maxsize=1)
def get_market_hours_early_closes(market="Equity-usa-[*]") -> List[datetime]:
    return [datetime.strptime(k + ' ' + v, '%m/%d/%Y %H:%M:%S') for k, v in get_market_hours(market)["earlyCloses"].items()]


def convert_to_hashable(arg):
    if isinstance(arg, np.ndarray):
        return str(arg)
    elif isinstance(arg, dict):
        return {k: convert_to_hashable(v) for k, v in arg.items()}
    return arg


def get_pkl_cache_key(clear_prefix='', *args, **kwargs):
    b = clear_prefix.encode()
    _args = [convert_to_hashable(arg) for arg in args]
    _kwargs = {k: convert_to_hashable(v) for k, v in kwargs.items()}
    try:
        b += pickle.dumps(_args)
    except Exception:
        b += str(_args).encode()
    try:
        b += pickle.dumps(_kwargs)
    except Exception:
        b += str(_kwargs).encode()

    digest = hashlib.md5(b).hexdigest()
    fn = f'{clear_prefix}-{digest}.pkl'
    for char in ['[', ']', ' ', ':', ',', "'", '(', ')', '{', '}']:
        fn = fn.replace(char, '')
    return fn


def cache_to_disk(clear_prefix, root):
    def cache_object(func):
        def wrapper(*args, **kwargs):
            fn = get_pkl_cache_key(clear_prefix, *args, **kwargs)

            if os.path.exists(os.path.join(root, fn)):
                try:
                    with open(os.path.join(root, fn), 'rb') as f:
                        obj = pickle.load(f)
                except EOFError:
                    obj = func(*args, **kwargs)
                    with open(os.path.join(root, fn), 'wb') as f:
                        pickle.dump(obj, f)
            else:
                obj = func(*args, **kwargs)
                with open(os.path.join(root, fn), 'wb') as f:
                    pickle.dump(obj, f)

            return obj

        return wrapper

    return cache_object


def timer(func):
    def wrapper(*args, **kwargs):
        # nonlocal total
        start = datetime.now()
        result = func(*args, **kwargs)
        duration = (datetime.now() - start).total_seconds()
        # total += duration
        print(f"Execution time: {duration:.2f} seconds")
        return result

    # total = 0
    return wrapper


def security_from_symbol(symbol: str, calc_date: date) -> Security:
    from options.typess.option import Option

    if ' ' in symbol:
        return Option(OptionContract.from_ib_symbol(symbol), calc_date)
    else:
        return Equity(symbol)


def exclude_outlier_quotes(df: pd.DataFrame, pf: Portfolio, equity: Equity) -> pd.DataFrame:
    """
    Outlier rows are removed unless they pertain to a holding. Those tagged to avoid modeling or pricing for these.
    """
    pd.options.display.max_columns = 8
    # removing nonsense rows, may contain a holding. Keep but log it.
    conditions = []
    if all((c in df.columns for c in ('ask_iv', 'bid_iv'))):
        conditions.append((df['ask_iv'] < df['bid_iv']))
    if 'ask_iv' in df.columns:
        conditions.append((df['ask_iv'] <= 0))
    if all((c in df.columns for c in ('ask_iv', 'bid_iv', 'mid_price_iv'))):
        conditions.append((df['mid_price_iv'] < df['bid_iv']) | (df['mid_price_iv'] > df['ask_iv']))
    if all((c in df.columns for c in ('tenor', 'mid_iv'))):
        conditions.append((df['tenor'] > 0.08) & (df['mid_iv'] > 2))
        conditions.append((df['tenor'] > 0.3) & (df['mid_iv'] > 1))

    df_outlier = df[reduce(operator.or_, conditions)] if conditions else pd.DataFrame(None)

    if len(df_outlier) == 0:
        return df

    elif pf:
        v_symbol = df_outlier.apply(lambda ps: ps_index2option_contract(ps, equity).ib_symbol(), axis=1)
        v_holding_symbol = {h.symbol.upper() for h in pf.holdings}
        df_outlier = df_outlier[~v_symbol.isin(v_holding_symbol)]
        warning(f'''# Data Issues={len(df_outlier)}: {df_outlier[['ask_iv', 'mid_iv', 'bid_iv', 'tenor']]}''')
        return df.loc[df.index.difference(df_outlier.index)]

    else:
        warning(f'''# Data Issues={len(df_outlier)}: {df_outlier[['ask_iv', 'mid_iv', 'bid_iv', 'tenor']]}''')
        return df.loc[df.index.difference(df_outlier.index)]


if __name__ == '__main__':
    print(n_trade_days(date(2024, 8, 30), date(2024, 9, 3)))
    print(n_trade_days(date(2024, 8, 27), date(2024, 8, 30)))
    print(n_trade_days(date(2024, 8, 27), date(2024, 8, 26)))
#     print(add_trade_days(date(2024, 8, 27), 1))
#     print(add_trade_days(date(2024, 8, 27), -1))
#     print(add_trade_days(date(2024, 8, 30), 1))
#     print(add_trade_days(date(2024, 8, 26), -1))
#     print(get_market_hours_holidays())
#     print(get_market_hours_early_closes())
#     sym, take = 'ON', -2
#     release_date = EarningsPreSessionDates(sym)[take]
#     print(release_date)
#     print(earnings_download_dates_start_end(release_date))
