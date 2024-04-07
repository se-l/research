import QuantLib as ql
import pandas as pd
import numpy as np
import arch
import multiprocessing
import matplotlib.pyplot as plt

from datetime import date, datetime, time, timedelta
from collections import defaultdict
from functools import reduce, lru_cache
from typing import List, Union, Tuple, Dict
from importlib import reload
from itertools import chain
from arbitragerepair import constraints, repair
from matplotlib import gridspec
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

import options.client as mClient
import options.volatility.implied as mImplied
from options.typess.enums import Resolution, TickType, SecurityType, GreeksEuOption, SkewMeasure
from options.typess.option_contract import OptionContract
from options.typess.equity import Equity
from options.typess.option_frame import OptionFrame
from shared.constants import EarningsPreSessionDates
from shared.modules.logger import logger

reload(mClient)
client = mClient.Client()


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
        option = mImplied.Option(contract)
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


def generate_multi_paths_df(sequence, num_paths):
    spot_paths = []
    vol_paths = []

    for i in range(num_paths):
        sample_path = seq.next()
        values = sample_path.value()

        spot, vol = values

        spot_paths.append([x for x in spot])
        vol_paths.append([x for x in vol])

    df_spot = pd.DataFrame(spot_paths, columns=[spot.time(x) for x in range(len(spot))])
    df_vol = pd.DataFrame(vol_paths, columns=[spot.time(x) for x in range(len(spot))])

    return df_spot, df_vol


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

    expiry_str = [c.isoformat() for c in expiries]

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


def repair_prices(df_q, calculation_date: date, n_repairs=1, plot=False, right='call'):
    C = np1d_from_df(df_q, 'mid_close')
    C_bid = np1d_from_df(df_q, 'bid_close')
    C_ask = np1d_from_df(df_q, 'ask_close')
    F = np1d_from_df(df_q, 'mid_price_underlying')
    iv = np1d_from_df(df_q, 'mid_iv')
    K = np1d_from_df(df_q, 'strike')
    T = np1d_from_df(df_q, 'tenor')
    Tdt = np1d_from_df(df_q, 'expiry')

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
    T, K, C, C_bid, C_ask, F, iv, Tdt = tuple([a[ix_nna] for a in [T, K, C, C_bid, C_ask, F, iv, Tdt]])
    # print(len(T))

    if plot:
        plot_prices_iv_repair(T, F, K, C, iv)

    epsilon1 = None
    epsilon2 = []
    for i in range(n_repairs):
        # normalise strikes and prices
        normaliser = constraints.Normalise()
        normaliser.fit(T, K, C, F)
        T1, K1, C1 = normaliser.transform(T, K, C)

        _, _, C1_bid = normaliser.transform(T, K, C_bid)
        _, _, C1_ask = normaliser.transform(T, K, C_ask)

        mat_A, vec_b, _, _ = constraints.detect(T1, K1, C1, verbose=True)

        # repair arbitrage - l1-norm objective
        eps1 = repair.l1(mat_A, vec_b, C1)
        if len(eps1) > 0:
            epsilon1 = eps1

        # repair arbitrage - l1ba objective
        spread_ask = C1_ask - C1
        spread_bid = C1 - C1_bid
        spread = [spread_ask, spread_bid]

        eps2 = repair.l1ba(mat_A, vec_b, C1, spread=spread)
        if len(eps2) > 0:
            epsilon2 = eps2

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

    iv2 = [calc_iv_for_repair(C2[i], F[i], calculation_date, right, calculation_date + timedelta(days=T[i] * 365), (K1 * F)[i]) for i in
           range(len(C2))]
    if plot:
        plot_after_repair(T1, F, K2, C2, iv2, Tdt, epsilon1, epsilon2, spread_bid, spread_ask, C2_bid, C2_ask)

    if plot:
        plot_prices_iv_repair(T, F, K2, C2, iv2)

    df = pd.DataFrame([C2, K2, T, iv2, C2_bid, C2_ask, Tdt], index=['mid_close', 'strike', 'tenor', 'mid_iv', 'bid_close', 'ask_close', 'expiry']).transpose()
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
        return 0.25, -0.25
    # elif skew_measure == SkewMeasure.M90M100:
    #     return 0.9, 1.0
    # elif skew_measure == SkewMeasure.M90M110:
    #     return 0.9, 0.9
    # elif skew_measure == SkewMeasure.M100M110:
    #     raise NotImplementedError
    #     return 0.9, 1
    else:
        raise ValueError(f'Unknown skew_measure: {skew_measure}')


def enrich_atm_iv(df):
    df['atm_iv'] = None
    for expiry, s_df in df.groupby(level='expiry'):
        for right, ss_df in s_df.groupby(level='right'):
            for ts, sss_df in ss_df.groupby(level='ts'):
                s = sss_df.iloc[0]['spot']
                v_atm_iv = atm_iv(sss_df.loc[ts], expiry, s, right=right)
                df.loc[sss_df.index, 'atm_iv'] = v_atm_iv


@lru_cache(maxsize=2**10)
def tenor(dt: date, calculation_dt: Union[date, ql.Date]) -> float:
    if isinstance(dt, (datetime, date)):
        return (dt - calculation_dt).days / 365
    elif isinstance(dt, ql.Date):
        return (dt.to_date() - calculation_dt).days / 365
    else:
        raise Exception('Unsupported type')


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
    return eu_option.impliedVolatility(price, bsmProcess)


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
    ps_resampled = ps.resample(sampling_period, closed='right').last().ffill()
    ix_time = ps_resampled.index.time
    ps_resampled = ps_resampled.loc[ps_resampled.index[(ix_time >= time(9, 30)) & (ix_time <= time(16, 0))]]
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


def grid_search_garch_params():
    # GRID Search. acf plot shows best p
    p_range = range(1, 5)
    q_range = range(1, 5)
    o_range = range(0, 3)

    # Initialize the best BIC and best parameters
    best_bic = np.inf
    best_params = None

    # Iterate over all possible combinations of p, q, and o
    for p in p_range:
        for q in q_range:
            for o in o_range:
                # Create a GARCH(p, q, o) model
                model = arch.arch_model(ps_iv, vol='GARCH', p=p, q=q, o=o)

                # Fit the model to the data
                results = model.fit(disp='off')

                # Calculate the BIC for the current model
                bic = results.bic

                # Update the best BIC and best parameters if necessary
                if bic < best_bic:
                    best_bic = bic
                    best_params = (p, q, o)
    # Print the best BIC and best parameters
    print(f'Best BIC: {best_bic:.2f}')
    print(f'Best parameters: p={best_params[0]}, q={best_params[1]}, o={best_params[2]}')


def exportAtmIVBySym(syms=["HPE", "IPG", "AKAM", "AOS", "MO", "FL", "AES", "LNT", "A", "ALL", "ARE", "ZBRA", "APD", "ALLE", "ZTS", "ZBH", "PFE"]):
    atmIVsBySym = defaultdict(dict)
    # syms = ["PFE"]
    for sym in syms:
        _sym = sym.lower()
        try:
            trades, quotes, contracts = load(_sym, start, end, n=1)
            for expiry, optionContracts in contracts.items():
                atmIVsBySym[_sym][expiry] = atm_iv(trades, quotes, optionContracts, n=1)  # , resolution=None)
        except Exception as e:
            print(e)
    return atmIVsBySym


def exportAtmIVBySym2(start, end, n=1,
                      syms=["HPE", "IPG", "AKAM", "AOS", "MO", "FL", "AES", "LNT", "A", "ALL", "ARE", "ZBRA", "APD", "ALLE", "ZTS", "ZBH", "PFE"]):
    start = start
    end = end
    client = mClient.Client()
    atmIVsBySym = defaultdict(dict)
    for sym in syms:
        _sym = sym.lower()

        equity = Equity(sym.lower()) if isinstance(sym, str) else sym
        contracts = client.central_volatility_contracts(equity, start, end, n=n)
        ivs = client.history(list(chain(*contracts.values())), start, end, Resolution.second, TickType.iv_quote, SecurityType.option)

        for expiry, optionContracts in contracts.items():  # run across all core
            mat_df = {}
            for contract in optionContracts:
                if str(contract) not in ivs:
                    print(f'no data for {contract}')
                    continue
                df = ivs[str(contract)]
                df['mid_iv'] = (df['ask_iv'] + df['bid_iv']) / 2
                mat_df[str(contract)] = df[~df['mid_iv'].isna()].sort_index()
            if not mat_df:
                continue
            # Outlier removal
            # confidence_level = 3
            # lookback_period = timedelta(days=5)
            # rolling_iv_mean = pd.Series(df['mid_iv']).rolling(window=lookback_period, min_periods=0).mean()
            # rolling_iv_std = pd.Series(df['mid_iv']).rolling(window=lookback_period, min_periods=0).std()
            # upper_bound = rolling_iv_mean + confidence_level * rolling_iv_std
            # lower_bound = rolling_iv_mean - confidence_level * rolling_iv_std
            # df = df[(df['mid_iv'] < upper_bound) & (df['mid_iv'] > lower_bound)]

            df_strike_iv = pd.DataFrame({float(k.split('_')[3]) / 10_000: df['mid_iv'] for k, df in mat_df.items()})  # .fillna(method='ffill')
            df_strike_iv = df_strike_iv.sort_index(axis=1)
            df_strike_iv = df_strike_iv.fillna(method='ffill')  # hmmm...
            df_strike_iv = df_strike_iv[~df_strike_iv.index.duplicated(keep='first')].sort_index()

            ps_t = pd.concat(df['mid_price_underlying'] for df in mat_df.values()).dropna()
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
            ps = pd.Series(df_atm_iv.sum(axis=1) / count_ivs, index=df_strike_iv.index)
            atmIVsBySym[_sym][expiry] = ps
    return atmIVsBySym


def aewma(vec: Union[pd.Series, np.ndarray], alpha: float, gamma: float) -> np.ndarray:
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


def str2ql_option_right(right: str):
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
    if expiry <= calculation_date:
        return 0
    try:
        yield_ts = ql.YieldTermStructureHandle(ql.FlatForward(to_ql_dt(calculation_date), rate, day_count))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(to_ql_dt(calculation_date), dividend, day_count))

        calculation_date_ql = set_ql_calculation_date(calculation_date)
        eu_option = prepare_eu_option(strike, tenor, spot, put_call, 0, calculation_date, calendar, day_count, yield_ts, dividend_ts)
        spot_quote = ql.QuoteHandle(ql.SimpleQuote(spot))
        vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date_ql, calendar, ql.QuoteHandle(ql.SimpleQuote(0)), day_count))
        yield_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date_ql, ql.QuoteHandle(ql.SimpleQuote(0)), day_count))
        bsm_process = ql.BlackScholesMertonProcess(spot_quote, dividend_ts, yield_ts, vol_ts)
        return eu_option.impliedVolatility(price, bsm_process)
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


def ps2delta(ps: pd.Series, iv_col: str, calendar, day_count, rate, dividend, spot_column='spot') -> float:
    ts, expiry, strike, right = unpack_mi_df_index(ps)
    return greek_bsm(GreeksEuOption.delta, float(strike), expiry, ps[spot_column], right, ps[iv_col], ts.date(), calendar, day_count, rate, dividend)


def ps2gamma(ps: pd.Series, iv_col: str, calendar, day_count, rate, dividend, spot_column='spot') -> float:
    ts, expiry, strike, right = unpack_mi_df_index(ps)
    return greek_bsm(GreeksEuOption.gamma, float(strike), expiry, ps[spot_column], right, ps[iv_col], ts.date(), calendar, day_count, rate, dividend)


def ps2vega(ps: pd.Series, iv_col: str, calendar, day_count, rate, dividend, spot_column='spot') -> float:
    ts, expiry, strike, right = unpack_mi_df_index(ps)
    return greek_bsm(GreeksEuOption.vega, float(strike), expiry, ps[spot_column], right, ps[iv_col], ts.date(), calendar, day_count, rate, dividend)


def ps2theta(ps: pd.Series, iv_col: str, calendar, day_count, rate, dividend, spot_column='spot') -> float:
    ts, expiry, strike, right = unpack_mi_df_index(ps)
    return greek_bsm(GreeksEuOption.theta, float(strike), expiry, ps[spot_column], right, ps[iv_col], ts.date(), calendar, day_count, rate, dividend)


def ps2iv(ps: pd.Series, price_col, calendar, day_count, rate, dividend, spot_column='spot') -> float:
    ts, expiry, strike, right = unpack_mi_df_index(ps)
    return implied_volatility(ps[price_col], float(strike), expiry, ps[spot_column], right, ts.date(), calendar, day_count, rate, dividend)


def ps2npv(ps: pd.Series, iv_col: str, calendar, day_count, rate, dividend, spot_column='spot') -> float:
    ts, expiry, strike, right = unpack_mi_df_index(ps)
    return npv(ps[iv_col], float(strike), expiry, ps[spot_column], right, ts.date(), calendar, day_count, rate, dividend)


def spot_from_df_equity_into_options(option_frame: OptionFrame):
    option_frame.df_options['spot'] = None
    for ts, sub_df in option_frame.df_options.groupby(level='ts'):
        option_frame.df_options.loc[(ts, slice(None), slice(None), slice(None)), 'spot'] = option_frame.df_equity.loc[ts, 'close']


def val_from_df(df, expiry, strike, right, col_nm):
    return df[col_nm].loc[(expiry, strike, right)]


def year_quarter(dt: date) -> str:
    quarter = (dt.month - 1) // 3 + 1
    return f'{dt.strftime("%y")}Q{quarter}'


def moneyness_iv(df, moneyness, expiry, fwd_s):
    atmny_strike = fwd_s * moneyness
    strikes = pd.Series(np.unique(df.loc[(expiry, slice(None))].index.get_level_values('strike')))
    strikes_at_moneyness = strikes.iloc[(strikes.astype(float) / fwd_s - moneyness).abs().sort_values().head(2).index].values
    iv_at_moneyness = df.loc[(expiry, strikes_at_moneyness)].values if isinstance(df, pd.Series) else df.loc[(expiry, strikes_at_moneyness), 'mid_iv'].values
    iv = pd.Series(iv_at_moneyness, index=strikes_at_moneyness).sort_index()
    if len(iv) == 1 or np.nan in iv.values or 0 in iv.values:
        logger.error(f'No IV for {moneyness} moneyness.')
        return np.nan
    return np.interp(atmny_strike, iv.index, iv.values)


def atm_iv(df, expiry, s, right=None):
    if right:
        return moneyness_iv(df.loc[(expiry, slice(None), right), 'mid_iv'], 1, expiry, s)
    else:
        call_iv = moneyness_iv(df.loc[(expiry, slice(None), 'call'), 'mid_iv'], 1, expiry, s)
        put_iv = moneyness_iv(df.loc[(expiry, slice(None), 'put'), 'mid_iv'], 1, expiry, s)
        return (call_iv + put_iv) / 2


def delta_heston(strike, tenor, engine):
    eu_option = make_eu_option(ql.Option.Call, strike, start + timedelta(days=int(tenor * 365)))
    eu_option.setPricingEngine(engine)
    return eu_option.delta()


def delta_mv_term(strike, tenor, rho, vv):
    eu_option = make_eu_option(ql.Option.Call, strike, start + timedelta(days=int(tenor * 365)))
    bsmProcess = ql.BlackScholesMertonProcess(spotQuote, dividend_ts, yield_ts, flat_vol_ts)
    analytical_engine = ql.AnalyticEuropeanEngine(bsmProcess)
    eu_option.setPricingEngine(analytical_engine)

    heston_iv = heston_vol_surface.blackVol(tenor, strike)
    bsm_vega = eu_option.vega()
    return rho * bsm_vega * vv / (heston_iv * spotQuote.value())


def earnings_download_dates(sym: str, take=-1, days_prior=7, days_after=3):
    try:
        release_date = EarningsPreSessionDates(sym)[take]
    except IndexError:
        raise ValueError(f'No earnings dates found for {sym}.')
    start = release_date - timedelta(days=days_prior)
    end = min(release_date + timedelta(days=days_after), date.today() - timedelta(days=1))
    return start, end


def apply_ds_ret_weights(m_dnlv01, f_weight_ds, cfg):
    n_ds = len(cfg.v_ds_ret)
    assert m_dnlv01.shape[0] == n_ds
    v_f_weight_ds = np.array([f_weight_ds(ds_ret) for ds_ret in cfg.v_ds_ret])
    for k in range(n_ds):
        m_dnlv01[k, :] *= v_f_weight_ds[k]
    return m_dnlv01


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
        v['expiry'] = k.expiry
        v['strike'] = k.strike
        v['right'] = k.right
        dfs.append(v.reset_index().rename(columns={'index': 'ts'}).set_index(['ts', 'expiry', 'strike', 'right']))
    return pd.concat(dfs)


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

