import datetime
import itertools
import pandas as pd
import scipy.stats
import numpy as np
import plotly.graph_objects as go

from collections import defaultdict
from typing import List, Dict, Callable
from decimal import Decimal
from plotly.subplots import make_subplots
from dataclasses import dataclass
from itertools import combinations

from options.helper import val_from_df, moneyness_iv, atm_iv, year_quarter
from options.typess.portfolio import Portfolio
from shared.plotting import show
from shared.constants import EarningsPreSessionDates
from options.typess.enums import Resolution
from options.typess.equity import Equity
from options.typess.option_frame import OptionFrame
from options.typess.option_contract import OptionContract
from options.typess.option import Option


@dataclass
class StressScenario:
    portfolio: Portfolio
    dNLV: float
    dS: float = 0
    dIVT1: float = 0
    ts_time_power: float = 0.5
    df0: pd.DataFrame = None


def implied_dS(dIV: float) -> float:
    return dIV / np.sqrt(365)


def get_f_weight_dS(implied_jump, plot=False) -> Callable:
    f_std = 1 / (1 + implied_jump)
    dS_weight_dist_left = scipy.stats.norm(1 - implied_jump, f_std * implied_jump)
    dS_weight_dist_right = scipy.stats.norm(1 + implied_jump, f_std * implied_jump)
    weight_dS_one = (dS_weight_dist_left.pdf(1) + dS_weight_dist_right.pdf(1))  # normalizing factor
    f_weight_dS = lambda dS: (dS_weight_dist_left.pdf(dS) + dS_weight_dist_right.pdf(dS)) / weight_dS_one
    if plot:
        fig = go.Figure(go.Scatter(x=np.linspace(0.7, 1.3, 100), y=[f_weight_dS(i) for i in np.linspace(0.7, 1.3, 100)]))
        fig.update_layout(title=f'Weight dS. implied_jump: {100 * implied_jump :.2f}%', xaxis_title='dS [%]', yaxis_title='Weight')
        show(fig, f'weight_dS_{implied_jump:.2f}.html')
    return f_weight_dS


def rank_hedges(pf0: Portfolio, universe: List[Option], s0, calcDate0, calcDate1: datetime.date, exp_jumps, diffusion_iv, df0,
                min_tenor: float = 1,
                v_dS=np.linspace(0.8, 1.2, 21),
                f_weight_ds: Callable[[float], float] = lambda x: 1,
                f_weight_risk: Callable[[float], float] = lambda risk: 1 if risk < 0 else 0.2,
                dIVT1=-0.01,
                ts_time_power=0.5
                ) -> Dict[Option, float]:
    """
    minimize risk reduction / expected loss (due to IV change)
    with +/- 20% dS.

    Begin with assuming dIV = 0.
    Add short option's IV falling (jump vol)... to diffusion iv, which is? earlier expiry level! - dIV level move / T**0.5
    Add long option's IV overall level falling... use historical releases as estimate. For past, use actual values and log each moneyness and IV move before and after.
    Add Equity EOD delta hedge

    [never quite worked for MV hedge optimization] Add long option's IV moving along smile... consider using AH surface for clean values...
    """
    risk_by_holding = {}

    # Stress test profile of current portfolio.
    pnl_base_risk_intercept0 = stress_pf(pf0, s0, 1, calcDate0, calcDate1, exp_jumps, diffusion_iv, df0=df0)
    risk0 = {dS: f_weight_ds(dS) * stress_pf(pf0, s0, dS, calcDate0, calcDate1, exp_jumps, diffusion_iv, df0=df0) for dS in v_dS}

    for o in universe:
        # Want large tenors as their IV decreases less after release date.
        tenor_o = (o.expiry - calcDate1).days / 365
        if tenor_o < min_tenor:
            continue

        p0 = val_from_df(df0, o.expiry, o.optionContract.strike, o.right, 'mid_price')
        iv0 = o.iv(p0, s0, calcDate0)
        if iv0 == 0 or np.isnan(iv0):
            continue

        pf1 = Portfolio({**pf0.holdings, **{o: pf0.holdings.get(o, 0) + 1}})
        pnl_base_risk_intercept1 = stress_pf(pf1, s0, 1, calcDate0, calcDate1, exp_jumps, diffusion_iv, dIVT1=dIVT1, df0=df0, ts_time_power=ts_time_power)
        risk1 = {dS: f_weight_ds(dS) * stress_pf(pf1, s0, dS, calcDate0, calcDate1, exp_jumps, diffusion_iv, dIVT1=dIVT1, df0=df0, ts_time_power=ts_time_power) for dS in v_dS}
        # {dS: f_weight_ds(dS) for dS in v_dS}
        # {dS: stress_pf(pf1, s0, dS, calcDate0, calcDate1, exp_jumps, diffusion_iv, dIVT1=dIVT1, df0=df0, ts_time_power=ts_time_power) for dS in v_dS}
        # [f_weight_risk(r-pnl_base_risk_intercept1) for r in risk1.values()]

        risk_by_holding[o] = sum([f_weight_risk(r - pnl_base_risk_intercept1) * r for r in risk1.values()]) - sum(
            [f_weight_risk(r - pnl_base_risk_intercept0) * r for r in risk0.values()])

    return risk_by_holding


def simulate_pf_pnl(pf, df0, df1, ts):
    nlv0 = 0
    nlv1 = 0
    for sec, q in pf.items():
        if isinstance(sec, Equity):
            p0 = val_from_df(df0, o.expiry, o.optionContract.strike, o.right, 'spot')
            p1 = val_from_df(df1, o.expiry, o.optionContract.strike, o.right, 'spot')
            nlv0 += q * p0
            nlv1 += q * p1
            continue

        o = sec
        p0 = val_from_df(df0, o.expiry, o.optionContract.strike, o.right, 'mid_price')

        try:
            p1 = val_from_df(df1, o.expiry, o.optionContract.strike, o.right, 'mid_price')
            if np.isnan(p0) or np.isnan(p1):
                print(f'Error Simulate_pf_pnl at {ts}: NaN price for {o} {p0} {p1}')
                continue

            nlv0 += p0 * q * 100
            nlv1 += p1 * q * 100

        except KeyError as e:
            # Expired option - intrinsic value only
            spot = val_from_df(df0, slice(None), slice(None), o.right, 'spot')[0]
            p1 = spot - float(o.strike) if o.right == 'call' else float(o.strike) - spot
            p1 = max(p1, 0)
            nlv0 += p0 * q * 100
            nlv1 += p1 * q * 100
            # print(f'simulate_pf_pnl: KeyError for {o}')

    return nlv1 - nlv0


@dataclass
class IVByTime:
    ts: datetime.datetime
    iv: float
    moneyness: float
    tenor: float
    right: str


def iv_at_tenor_over_time(df, moneyness, tenor) -> pd.DataFrame: pass


def iv_at_date_over_time(df, dt, moneyness=1, net_yield=0) -> pd.DataFrame:
    """
    Given a list of surface snaps, plot desired tenor; K/S. Likely moving around a mean or correlated to something, e.g., IVX, 30d HV etc. Plot'em...

    Class that holds multiple IV surfaces or at least corresponding data. Can interpolate between them. primarily ATM (expiry) + skew * dIV
    for any given date. Common scale cannot be strike, but must be K/S or ln(K/S) or delta.

    Should be generic enough to support general research on how IV surfaces change over time.

    For earning's release should see the speed of IV change increasing over time.
    """
    results = []
    for ts, s_df in df.groupby(level='ts'):
        for right, ss_df in s_df.droplevel(0).groupby(level='right'):
            right_moneyness = {'call': 1 / moneyness, 'put': moneyness}[right]
            s = ss_df.iloc[0]['spot']
            expiries = ss_df.index.get_level_values('expiry').unique()
            at_the_date_expiries = sorted(expiries[pd.Series(expiries - dt).apply(lambda x: x.days).abs().sort_values().head(2).index])
            v_atexpiry_iv = []
            for expiry in at_the_date_expiries:
                fwd_s = s * np.exp(net_yield * (expiry - datetime.date.today()).days / 365)
                iv = moneyness_iv(ss_df.droplevel(-1), right_moneyness, expiry, fwd_s)
                v_atexpiry_iv.append(iv)
            atexpiry_iv = np.interp((dt - at_the_date_expiries[0]).days, [0, (at_the_date_expiries[1] - at_the_date_expiries[0]).days], v_atexpiry_iv)
            tenor = (dt - ts.date()).days / 365
            results.append(IVByTime(ts, atexpiry_iv, right_moneyness, tenor, right))
    return pd.DataFrame(results)


@dataclass
class SkewByTime:
    ts: datetime.datetime
    moneyness_iv: Dict[float, float]
    tenor: float
    right: str
    skew_cross: float = None
    skew_itm: float = None
    skew_otm: float = None

    def __post_init__(self):
        max_moneyness = max(self.moneyness_iv.keys())
        min_moneyness = min(self.moneyness_iv.keys())
        self.skew_cross = (self.moneyness_iv[max_moneyness] - self.moneyness_iv[min_moneyness]) / (max_moneyness - min_moneyness)
        if self.right == 'call':
            self.skew_itm = (self.moneyness_iv[1] - self.moneyness_iv[min_moneyness]) / (1 - min_moneyness)
            self.skew_otm = (self.moneyness_iv[1] - self.moneyness_iv[max_moneyness]) / (1 - max_moneyness)
        elif self.right == 'put':
            self.skew_itm = (self.moneyness_iv[1] - self.moneyness_iv[max_moneyness]) / (1 - max_moneyness)
            self.skew_otm = (self.moneyness_iv[1] - self.moneyness_iv[min_moneyness]) / (1 - min_moneyness)
        else:
            raise ValueError('Right must be call or put')


def skew_at_date_over_time(df, dt, net_yield=0) -> pd.DataFrame:
    results = []
    for ts, s_df in df.groupby(level='ts'):
        tenor = (dt - ts.date()).days / 365
        for right, ss_df in s_df.droplevel(0).groupby(level='right'):
            s = ss_df.iloc[0]['spot']
            expiries = ss_df.index.get_level_values('expiry').unique()
            at_the_date_expiries = sorted(expiries[pd.Series(expiries - dt).apply(lambda x: x.days).abs().sort_values().head(2).index])

            dct_moneyness_iv = {}

            for right_moneyness in [1.1, 1, 0.9]:
                v_atexpiry_iv = []
                for expiry in at_the_date_expiries:
                    fwd_s = s * np.exp(net_yield * (expiry - datetime.date.today()).days / 365)
                    iv = moneyness_iv(ss_df.droplevel(-1), right_moneyness, expiry, fwd_s)
                    v_atexpiry_iv.append(iv)
                atexpiry_iv = np.interp((dt - at_the_date_expiries[0]).days, [0, (at_the_date_expiries[1] - at_the_date_expiries[0]).days], v_atexpiry_iv)
                dct_moneyness_iv[right_moneyness] = atexpiry_iv

            results.append(SkewByTime(ts, dct_moneyness_iv, tenor, right))
    return pd.DataFrame(results)


def stress_pf(pf: Portfolio, s0, dS: float, calcDate0, calcDate1: datetime.date, jump_expiries: List[datetime.date], diffusion_iv, dIVT1=-0.01, df0=None,
              ts_time_power=0.5, use_skew=True):
    """
    At the very least, return pf nlv after dS.
    Then in addition, also change dIV.
    No equity hedging here
    """
    s1 = s0 * dS
    nlv0 = 0
    nlv1 = 0
    for sec, q in pf.items():
        if isinstance(sec, Equity):
            nlv0 += q * s0
            nlv1 += q * s1
            continue

        o = sec
        if df0 is not None:
            p0 = val_from_df(df0, o.expiry, o.optionContract.strike, o.right, 'mid_price')
            iv0 = o.iv(p0, s0, calcDate0)
            if iv0 == 0 or np.isnan(iv0):
                raise ValueError(f'IV0 is 0 or NaN for {o} {p0} {s0} {calcDate0}')
        else:
            raise ValueError('Need to provide df0')

        nlv0 += o.npv(iv0, s0, calcDate0) * q * 100

        if o.expiry in jump_expiries:  # shouldn't rely on me defining what drops. every expiry will drop. gotta estimate a iv_srf_1(s)
            iv1 = diffusion_iv
        else:
            # square root of time rule starting from a presume IV level move for tenor 1.
            if use_skew:
                try:
                    skew = df0.loc[(o.expiry, o.optionContract.strike, o.right), f'div_skew_ds_{(dS - 1):.2f}_rolling']
                except KeyError:
                    print(f'KeyError: No Skew for {o.expiry} {o.optionContract.strike} {o.right} {(dS - 1):.2f}')
                    skew = 0
                skew = 0 if np.isnan(skew) else skew
                dIV_skew = (s1 - s0) * skew
            else:
                dIV_skew = 0

            dIV_tenor = d_iv_power_law(dIVT1, o.expiry, calcDate1, ts_time_power=ts_time_power)
            # even better: identify which expiries are currently off the expected power law, and modify ts_time_power factor anticipating their correction. Rather than power law
            # might rather compare normalized term structures.

            iv1 = iv0 + dIV_tenor + dIV_skew
            # print(f'{o.optionContract} {iv0:.4f} -> {iv1:.4f} dIV_tenor: {dIV_tenor:.4f} dIV_skew: {dIV_skew:.4f} tenor: {tenor:.2f} dS: {(s1 - s0):.4f} is_otm: {o.is_otm(s0)}')

        nlv1 += o.npv(iv1, s1, calcDate1) * q * 100
    return nlv1 - nlv0


def d_iv_power_law(d_iv_tenor_1, expiry: datetime.date, calcDate: datetime.date, ts_time_power=0.5):
    tenor = (expiry - calcDate).days / 365
    if tenor <= 0:
        return d_iv_tenor_1
    return d_iv_tenor_1 / tenor ** ts_time_power


def infinite_volatility(df) -> pd.Series:
    """
    Vola infinity could be something along max expiry sample 3-2 months before reporting date. Averaged + error bounds.
    Also sample the usual term structure power law. Average + err bounds.
    From avg. power law and infite vola, calc expected post reporting date IVs. Issue, questionable whether surface returns immediately to this state. Check previous earnings…
    Assume for each month how much level of expiry is gonna drop.
    """

    @dataclass
    class Point:
        ts: datetime.datetime
        iv: float

    max_expiry = df.index.get_level_values('expiry').max()

    ivs = []
    for ts, s_df in df.groupby(level='ts'):
        if max_expiry not in s_df.index.get_level_values('expiry'):
            continue
        max_expiry_atm_iv = atm_iv(s_df.loc[ts], max_expiry, s_df.iloc[0]['spot'])
        ivs.append(Point(ts, max_expiry_atm_iv))
    ps_iv = pd.DataFrame(ivs).set_index('ts')['iv'].where(lambda x: x > 0).dropna()
    return ps_iv


def power_law_term_structure(df, min_expiry, min_tenor_near=0.0):
    res = []
    for ts, s_df in df.groupby(level='ts'):
        s_df = df.loc[ts]
        s_expiries = s_df.index.get_level_values('expiry').unique()
        s = s_df.iloc[0]['spot']
        for combo in combinations(s_expiries, 2):
            exp_near = min(combo)
            if exp_near != min_expiry:
                continue
            exp_far = max(combo)
            tenor_near = (exp_near - ts.date()).days / 365
            tenor_far = (exp_far - ts.date()).days / 365

            if tenor_near < min_tenor_near:
                continue

            near_atm_iv = atm_iv(s_df, exp_near, s)
            far_atm_iv = atm_iv(s_df, exp_far, s)
            if near_atm_iv == 0 or far_atm_iv == 0:
                continue

            z = (far_atm_iv - near_atm_iv) * np.sqrt(tenor_far * tenor_near) / np.sqrt(tenor_far - tenor_near)
            res.append({'ts': ts, 'exp_near': exp_near, 'exp_far': exp_far, 'near_atm_iv': near_atm_iv, 'far_atm_iv': far_atm_iv, 'z': z, 'tenor_near': tenor_near,
                        'tenor_far': tenor_far})
    res_df = pd.DataFrame(res).set_index('ts')

    fig = make_subplots(rows=3, cols=1)

    # ATM IVs by expiry
    for exp_far, ss_df in res_df.groupby('exp_far'):
        fig.add_trace(go.Scatter(x=ss_df.index, y=ss_df['far_atm_iv'], mode='lines+markers', name=f'{exp_far}', marker=dict(size=2)), row=1, col=1)

    # Unscaled
    for ix, s_df in res_df.groupby(['exp_near', 'exp_far']):
        fig.add_trace(go.Scatter(x=s_df.index, y=s_df['z'], mode='lines+markers', name=f'{ix[0]}-{ix[1]}', marker=dict(size=2)), row=2, col=1)

    # Scaled
    exp_T3m = res_df['exp_near'].iloc[(res_df['tenor_near'] - 0.25).abs().argmin()]
    exp_T3m = min_expiry
    exp_T1y = res_df['exp_far'].iloc[(res_df['tenor_far'] - 1).abs().argmin()]
    ps_scale = res_df.reset_index().set_index(['exp_near', 'exp_far']).loc[exp_T3m, exp_T1y][['z', 'ts']].set_index('ts')['z']
    for ts, s_df in res_df.groupby(level='ts'):
        res_df.loc[ts, 'z'] = s_df['z'] / ps_scale.loc[ts]

    for ix, s_df in res_df.groupby(['exp_near', 'exp_far']):
        fig.add_trace(go.Scatter(x=s_df.index, y=s_df['z'], mode='lines+markers', name=f'{ix[0]}-{ix[1]}', marker=dict(size=2)), row=3, col=1)

    show(fig, 'power_law_term_structure.html')

    return res_df


def regress_stressed_delta_total(pf: Portfolio, s0: float, calcDate0: datetime.date, calcDate1: datetime.date, exp_jumps: List[datetime.date], diffusion_iv, implied_jump: float,
                                 df0: pd.DataFrame):
    v_dS = np.linspace(1 - implied_jump, 1 + implied_jump, 10)
    v_dIVdSdPL_pf = [stress_pf(pf, s0, dS, calcDate0, calcDate1, exp_jumps, diffusion_iv, dIVT1=0, df0=df0) for dS in v_dS]

    # regress v_dS and v_dIVdSdPL_pf
    from scipy.stats import linregress
    res = linregress(v_dS, v_dIVdSdPL_pf)
    deltaTotal = res.slope * s0
    return deltaTotal


def estimate_diffusion_iv(df, release_date):
    ts = df.index.get_level_values('ts').unique()
    expiries_df = list(sorted(set(df.index.get_level_values('expiry').values)))

    # This calc requires about 14 days worth of data if weeklies are traded.
    # 7 days to avoid typical ramp of expiring options
    diffusion_iv = None
    if [i for i in expiries_df if i < release_date]:
        previous_non_jump_expiry = sorted([i for i in expiries_df if i < release_date])[-1]
        v_ts_pre_releaseTm7 = [i for i in ts if i.date() == previous_non_jump_expiry - datetime.timedelta(days=7)]
        if v_ts_pre_releaseTm7:
            ts_pre_releaseTm7 = v_ts_pre_releaseTm7[-1]
            dfm7 = df.loc[ts_pre_releaseTm7]
            sm7 = dfm7.iloc[0]['spot']
            diffusion_iv = atm_iv(dfm7, previous_non_jump_expiry, sm7)  # 7 days is already elevated. not jump iv
    if not diffusion_iv:
        max_exp_vola = infinite_volatility(df).mean()
        print(f'estimate_diffusion_iv: No data 7 days before previous_non_jump_expiry. Using max_exp_vola={max_exp_vola}. '
              # f'Therefore assuming all intrinsic value will be earned for shorted options'
              # f'Dont use if expiry is further away than end of week.'
              )
        diffusion_iv = max_exp_vola
    return diffusion_iv


def enrich_skew_metrics(df: pd.DataFrame, plot_dates: List[datetime.date]):
    figSkewOverTime = make_subplots(rows=len(plot_dates), cols=1, subplot_titles=[dt.isoformat() for dt in plot_dates])
    for i, dt in enumerate(plot_dates):
        df_mny1_tenor_dt = skew_at_date_over_time(df, dt, 1)
        for right, _df in df_mny1_tenor_dt.groupby('right'):
            print(f'''{dt} {right} Mean Skew OTM: {_df['skew_otm'][_df['skew_otm'] != 0].mean()}''')
            print(f'''{dt} {right} Mean Skew ITM: {_df['skew_itm'][_df['skew_itm'] != 0].mean()}''')
            print('-' * 30)
            figSkewOverTime.add_trace(go.Scatter(x=_df['ts'], y=_df['skew_otm'], mode='lines', name=f'{right}-skew otm: {dt}'), row=i + 1, col=1)
            figSkewOverTime.add_trace(go.Scatter(x=_df['ts'], y=_df['skew_itm'], mode='lines', name=f'{right}-skew itm: {dt}'), row=i + 1, col=1)
    show(figSkewOverTime, 'skew_over_time.html')


def add_plot_term_structure(df, calcDate, fig, row, dIV_levels=(0, -0.01), ts_time_power=0.5):
    for dIV_level in dIV_levels:
        x = []
        y = []
        for expiry, ss_df in df.groupby('expiry'):
            ts = ss_df.index.get_level_values('ts').unique()[0]
            iv = atm_iv(ss_df.loc[ts], expiry, ss_df.iloc[0]['spot'])
            if iv <= 0 or np.isnan(iv):
                continue
            dIV_tenor = d_iv_power_law(dIV_level, expiry, calcDate, ts_time_power=ts_time_power)
            iv = iv + dIV_tenor
            x.append((expiry - df.index.get_level_values('ts').min().date()).days)
            y.append(iv)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f'IV Term Structure @ {calcDate} dIV={dIV_level}', marker=dict(size=3)), row=row, col=1)


def add_plot_iv(df, ts, pf_iv, fig, row, col, marker_size_iv=3, marker_size_holdings=8):
    calcDate = ts.date()
    holding_expiries = list(set([sec.expiry for sec in pf_iv.holdings.keys() if isinstance(sec, Option)]))

    for expiry in holding_expiries:
        s_df = df.loc[(ts, expiry, slice(None), slice(None))]

        # Plot IVs
        for right, ss_df in s_df.groupby(level='right'):
            sample_df = ss_df[ss_df['mid_iv'].fillna(0) != 0]
            fig.add_trace(go.Scatter(x=sample_df['moneyness'], y=sample_df['mid_iv'], mode='lines+markers', name=f'IV {expiry} {right}', marker=dict(size=marker_size_iv)), row=row,
                          col=col)

    right_symbol_map = {'call': 'x-dot', 'put': 'circle-dot'}
    for right, s_df in df.loc[ts].groupby(level='right'):
        holdings = {sec: q for sec, q in pf_iv.holdings.items() if isinstance(sec, Option) and sec.right == right}
        s = s_df.iloc[0]['spot']
        x_moneyness = [float(sec.optionContract.strike) / s for sec in holdings.keys()]
        y_iv = [sec.iv(s_df.loc[(sec.expiry, sec.optionContract.strike, sec.right), 'mid_price'], s, calcDate) for sec in holdings.keys()]
        if 0 in y_iv:
            raise ValueError(f'Zero IV for {right} {ts} {x_moneyness} {y_iv}')

        fig.add_trace(go.Scatter(
            x=x_moneyness, y=y_iv, mode='markers', name=f'IV Holdings {right}',
            text=[f'{str(sec)}: {q}' for sec, q in holdings.items()],
            marker=dict(size=marker_size_holdings, symbol=right_symbol_map[right])), row=row, col=col)


def add_plot_skew(df, ts, pf_iv, v_dS, fig, row, col, marker_size=4):
    holding_expiries = list(set([sec.expiry for sec in pf_iv.holdings.keys() if isinstance(sec, Option)]))

    for expiry in holding_expiries:
        s_df = df.loc[(ts, expiry, slice(None), slice(None))]
        for right, ss_df in s_df.groupby(level='right'):
            # Plot Skew from 0 to dS IV change
            x = []
            y = []
            for dS in v_dS:
                try:
                    skew = ss_df[f'div_skew_ds_{(dS - 1):.2f}_rolling'].iloc[0]
                except KeyError:
                    print(f'KeyError: No Skew for {expiry} {dS:.2f}')
                    continue
                y.append(skew)
                x.append(1 + (dS - 1) / 2)
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'Skew {expiry} {right}', marker=dict(size=marker_size)), row=row, col=col)


def run():
    """
        Refinement:
        - Long hedges are bought 2 days before release date.
        - The ultimate short is bought 1 day before release date.
            - There may be a mismatch of strikes now. Understand how to offset, calculate purchase/sell of risk adjusting options.

        Todo: Check how much the extrinsic value decreases towards release date. Want to short/long more than 1 D in advance?? IV goes up. Leaving money on the table?

        # Questions:
        # - What's the optimal T-x days to sell near expiry options? Understands how far values change. How intrinsic value changes as release date comes up.
        # - Optimal near/long hedge ratios? Risk / Pnl Trade off... relate risk space and cost to hedge?

        # C) Run for different release date - x days...

        # Sell near ATM, hedge with far slightly OTM; build position along the day low delta, gamma inevitably increasing. Continuous delta hedge. Match assignments with exercises. Liquidate after expiry.
        # Trickier: picking the right options to hedge with. Give no gain to pos risk. Reduce area under dPL/dS curve. Metric to minimize: dPL/dS integrated over dS up to +/-10% have weight of 1. Find full solution first - the trade plan.
        # From every existing position towards trade plan should arrive at same existing plan…

        ITM options are more sensitive to IV changes than OTM options. Are ITM or OTM far options more suitable for hedging near short? IV wise: better; delta/gamma wise? Need to simulate for a range of dS.
        """
    sym = 'ORCL'
    # sym = 'ONON'
    # sym = 'DLTR'
    # sym = 'PATH'
    equity = Equity(sym)
    resolution = Resolution.second
    seq_ret_threshold = 0.005
    release_date = EarningsPreSessionDates(sym)[-1]

    option_frame = OptionFrame.load_frame(equity, resolution, seq_ret_threshold, year_quarter(release_date))
    df = option_frame.df_options.sort_index()

    marker_size = 4
    min_tenor = 0.8
    ranking_iterations = 20
    add_equity_holdings = False

    v_dIVT1 = np.linspace(0.0, -0.03, 4)
    dflt_dIVT1 = -0.0
    ts_time_power = 0.5

    ts = df.index.get_level_values('ts').unique()

    # adjustment. hedges can choose on which date. sell high IV is before release to center around underlying price.
    v_ts_pre_release = [i for i in ts if i.date() <= release_date]  # preparing on day of expiry
    # v_ts_pre_release = [i for i in ts if i.date() <= release_date - datetime.timedelta(days=1)]  # preparing options the night before

    # avoid before 10 am because rolling skew is still very noise
    v_ts_pre_release = [i for i in v_ts_pre_release if i.hour >= 10]

    v_ts_post_release = [i for i in ts if i.date() >= (release_date + datetime.timedelta(days=1))]
    # avoid after 4 pm because rolling skew is still very noise
    v_ts_post_release = [i for i in v_ts_post_release if i.hour >= 10]
    first_2_days = sorted(list(set([i.date() for i in v_ts_post_release])))[:2]
    v_ts_post_release = [i for i in v_ts_post_release if i.date() in first_2_days]
    ts_pre_release = v_ts_pre_release[-2]
    print(f'Release date: {release_date}. ts_pre_release: {ts_pre_release}')

    if v_ts_post_release:
        ts_post_release = v_ts_post_release[-1]
        print(f'TS Post Release - Min: {np.min(v_ts_post_release)} Max: {np.max(v_ts_post_release)}')

    df0 = df.loc[v_ts_pre_release]
    if v_ts_post_release:
        df1 = df.loc[v_ts_post_release]

    expiries = list(sorted(set(df0.index.get_level_values('expiry').values)))

    expiry_strike_pairs = list(sorted(set(zip(df0[df0['mid_iv'] != 0].index.get_level_values('expiry').values, df0[df0['mid_iv'] != 0].index.get_level_values('strike').values))))
    # expiry_strike_pairs_str = '\n'.join([f'{t[0].isoformat()} | {float(t[1])}' for t in expiry_strike_pairs])
    # print(f'''Expiry | strike pairs\n: {expiry_strike_pairs_str}''')

    s0 = df0.loc[ts_pre_release].iloc[0]['spot']
    s1 = df1.loc[ts_post_release].iloc[0]['spot'] if v_ts_post_release else np.nan
    print(f'''Spot0: {s0}; Spot1: {s1}; Stock jumped by dS: {round(s1 - s0, 1)} | {round(100 * (s1 / s0 - 1), 1)} %''')

    calcDate0 = ts_pre_release.date()
    calcDate1 = release_date + datetime.timedelta(days=1)

    exp_jump1 = sorted([i for i in expiries if i >= release_date])[0]
    exp_jump2 = sorted([i for i in expiries if i >= release_date])[1]
    exp_jumps = [exp_jump1, exp_jump2]  # get rid of specifying jump IVs. Anything that's elevated returns to a certain level.

    inf_vol = infinite_volatility(df)
    print(f'Max Expiry ATM Volatility: {inf_vol.mean()}')
    power_law_term_structure(df, exp_jump1)

    diffusion_iv = estimate_diffusion_iv(df, release_date)
    print(f'Nearest Expiry: {exp_jump1}, {sym.upper()}, ATM IV: {atm_iv(df0.loc[ts_pre_release], exp_jump1, s0)}, Diffusion IV: {diffusion_iv}')
    #####################

    ############## Spot price series
    figSpotTimeValue = make_subplots(rows=2, cols=1, subplot_titles=['Spot', 'Time Value'])
    figSpotTimeValue.add_trace(go.Scatter(x=df.index.get_level_values('ts'), y=df['spot'], mode='lines', name='Spot'), row=1, col=1)
    figSpotTimeValue.add_vline(x=datetime.datetime.fromisoformat(release_date.isoformat()) + datetime.timedelta(hours=16), line_dash='dash', line_color='red', row=1, col=1)
    figSpotTimeValue.update_layout(title=f'{sym} Spot', xaxis_title='Time', yaxis_title='Spot')
    ###############

    ############## Extrinsic value over time for near expiry - IS WRONG
    strikes = pd.Series(np.unique(df0.loc[(ts_pre_release, exp_jump1, slice(None), slice(None))].index.get_level_values('strike').astype(float)))
    strikes_atm = strikes.iloc[(strikes - s0).abs().sort_values().head(2).index].values
    for right, s_df in df.loc[slice(None), exp_jump1, strikes_atm, slice(None)].groupby('right'):
        strike = s_df.index.get_level_values('strike').astype(float)
        spot = s_df['spot']
        if right == 'call':
            intrinsic_value = spot - strike
            time_value = s_df['mid_price'] - intrinsic_value
        elif right == 'put':
            intrinsic_value = strike - spot
            time_value = s_df['mid_price'] - intrinsic_value
        else:
            raise ValueError('Right must be call or put')
        figSpotTimeValue.add_trace(go.Scatter(x=s_df.index.get_level_values('ts'), y=time_value, mode='markers', name=f'{right}', marker=dict(size=marker_size)), row=2, col=1)
        figSpotTimeValue.add_vline(x=datetime.datetime.fromisoformat(release_date.isoformat()) + datetime.timedelta(hours=16), line_dash='dash', line_color='red', row=2, col=1)

    show(figSpotTimeValue, f'{sym}_{release_date.strftime("%y%m%d")}_spot_time_value.html')
    ###############################################

    # Analysing IVs meant for hedging
    # Historically, how did the IVs change throughout the day? Entry & exit? For near, how much did time value in USD change? Theta?
    # Historically, how much did the IVs change after the release date? Feasible trade? Entry range, exit range... so just plot IV at ~low res of day.
    plot_dates = expiries  # release_date + datetime.timedelta(days=i) for i in [0, 30, 60, 90, 120, 150, 180, 270, 365, 500]]
    figIvOverTime = make_subplots(rows=len(plot_dates), cols=1, subplot_titles=[dt.isoformat() for dt in plot_dates])
    if v_ts_post_release:
        df_pre_post = df.sort_index().loc[(slice(pd.Timestamp(ts_pre_release.date()), pd.Timestamp(ts_post_release.date() + datetime.timedelta(days=2))))]
    else:
        df_pre_post = df.sort_index().loc[(slice(pd.Timestamp(ts_pre_release.date()), pd.Timestamp(ts_pre_release.date() + datetime.timedelta(days=2))))]
    for i, dt in enumerate(plot_dates):
        df_mny1_tenor_dt = iv_at_date_over_time(df_pre_post, dt, 1)
        for right, s_df in df_mny1_tenor_dt.groupby('right'):
            # release_date_iv = s_df[(s_df['iv']!=0) & (s_df['ts'].dt.date == release_date)]['iv'].mean()
            # pre_release_date_iv = s_df[(s_df['iv']!=0) & (s_df['ts'].dt.date < release_date)]['iv'].mean()
            # print(f'IV at {dt} | {right} | release: {release_date_iv} | pre_release: {pre_release_date_iv}')
            ix = s_df.index[(s_df['iv'] > s_df['iv'].quantile(0.05)) & (s_df['iv'] < s_df['iv'].quantile(0.95))]
            figIvOverTime.add_trace(go.Scatter(x=s_df.loc[ix, 'ts'], y=s_df.loc[ix, 'iv'], mode='markers', marker=dict(size=marker_size), name=f'{right}: {dt}'), row=i + 1, col=1)
            figIvOverTime.add_vline(x=datetime.datetime.fromisoformat(release_date.isoformat()) + datetime.timedelta(hours=16), line_dash='dash', line_color='red')
    figIvOverTime.update_layout(title=f'{sym} ATM IV over Time', xaxis_title='Time', yaxis_title='IV by expiry')
    show(figIvOverTime, f'''{sym}_{release_date.strftime('%y%m%d')}_iv_over_time.html''')

    # estimated_diffusion_iv   No feasible way estimating this diffusion way for later expiries. For csco, later expiries ATM IV fell significantly even below long term IV levels.
    # So now, just check plot this time atm iv over term and last release's one, then pick minimum tenor to use for hedging...

    option_universe = []
    for expiry, strike in expiry_strike_pairs:
        for right in ('call', 'put'):
            try:
                option_universe.append(Option(OptionContract('', sym, expiry, Decimal(strike), right), ts_pre_release.date(), s0,
                                              val_from_df(df0.loc[ts_pre_release], expiry, Decimal(strike), right, 'mid_iv')))
            except KeyError:
                pass
    print(f'Added {len(option_universe)} options to universe')

    # Stress test the portfolio params
    v_dS = np.linspace(0.8, 1.2, 21)
    v_dSpct = [100 * (dS - 1) for dS in v_dS]

    # Initialize portfolio of near short options
    strikes = pd.Series(np.unique(df0.loc[(ts_pre_release, exp_jump1, slice(None), slice(None))].index.get_level_values('strike').astype(float)))
    strikes_atm = strikes.iloc[(strikes - s0).abs().sort_values().head(2).index].values

    print(f'ATM strike: {strikes_atm}')
    pf = Portfolio()
    short_multiplier = 1
    for sec, q in {
        Option(OptionContract('', sym, exp_jump1, Decimal(strikes_atm[0]), 'call'), calcDate0, 0, 0): -1 * short_multiplier,
        Option(OptionContract('', sym, exp_jump1, Decimal(strikes_atm[0]), 'put'), calcDate0, 0, 0): -1 * short_multiplier,
        Option(OptionContract('', sym, exp_jump1, Decimal(strikes_atm[1]), 'call'), calcDate0, 0, 0): -1 * short_multiplier,
        Option(OptionContract('', sym, exp_jump1, Decimal(strikes_atm[1]), 'put'), calcDate0, 0, 0): -1 * short_multiplier,
    }.items():
        pf.add_holding(sec, q)

    implied_jump = implied_dS(np.sqrt(atm_iv(df0.loc[ts_pre_release], exp_jump1, s0) ** 2 - diffusion_iv ** 2))
    print(f'implied jump return: {100 * implied_jump :.2f} %; implied jump dS={implied_jump * s0 :.2f}')

    # Selecting hedges
    # Need a preference over far hedges if otherwise equal to near, equal gamma. Gamma far < gamma near. delta far flatter than delta near. IV far less sensitive to dS than near.
    # How much will IV(tenor) change?
    # If tenor + 2y is assumed constant (which it isnt necessarily), then jump iv may exponentially decay with respect to tenor, just like diffusion iv.
    # This can be calibrated... Then dIV(tenor) = dIV(jump) * exp(-tenor) + dIV(diffusion) * exp(-tenor)

    """Rewarding risk reduction around implied jump dS move more than large dS moves"""
    f_weight_dS = get_f_weight_dS(implied_jump * 3, plot=True)

    figures = []
    stress_scenarios: List[StressScenario] = []

    pf_i = {dIVT1: Portfolio(defaultdict(int, {**pf.holdings})) for dIVT1 in v_dIVT1}
    for i in range(ranking_iterations):
        figdSdIV = make_subplots(rows=3, cols=2, specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}]], subplot_titles=['dS dIV dPL', 'IV ATM Horizontal', 'IV Vertical'])

        # Selecting a portfolio for the presumed IV level change
        pf_iv = pf_i[dflt_dIVT1]
        if i > 0:
            ranked_hedges_dct = rank_hedges(
                pf_iv, option_universe, s0, calcDate0, calcDate1, exp_jumps, diffusion_iv, df0.loc[ts_pre_release],
                min_tenor=min_tenor, v_dS=v_dS, f_weight_ds=f_weight_dS, dIVT1=dflt_dIVT1, ts_time_power=ts_time_power)

            # print(f'Ranked hedges: {ranked_hedges_dct}')
            best_hedge = max(ranked_hedges_dct, key=ranked_hedges_dct.get)
            print(f'Best hedge: {best_hedge}: {ranked_hedges_dct[best_hedge]}')
            if ranked_hedges_dct[best_hedge] <= 0:
                print(f'Iteration={i}, dIVT1={dflt_dIVT1}. No more useful hedges.')
                # break
            pf_iv.add_holding(best_hedge, 1)  # enable shorting sth...

        # Stressing the portfolio for a few more dIV level changes
        for dIVT1 in v_dIVT1:
            print(f'Iteration: {i}, dIVT1: {dIVT1}')
            pf_iv = pf_i[dflt_dIVT1]

            deltaTotalPf0Total = pf_iv.deltaTotal(calcDate0, s0, df0.loc[ts_pre_release])
            # deltaTotalPf0StressedTotal = regress_stressed_delta_total(pf_iv, s0, calcDate0, calcDate1, exp_jumps, diffusion_iv, implied_jump * 2, df0.loc[ts_pre_release])
            print(f'Portfolio DeltaTotal={deltaTotalPf0Total:.2f}, Equity Position: {pf_iv[Equity(sym)]}')  # , deltaTotalPf0StressedTotal: {deltaTotalPf0StressedTotal}')

            # Stress testing short long pf
            v_dNLV = []
            for dS in v_dS:
                dNLV = stress_pf(pf_iv, s0, dS, calcDate0, calcDate1, exp_jumps, diffusion_iv, dIVT1=dIVT1, df0=df0.loc[ts_pre_release], ts_time_power=ts_time_power)
                stress_scenarios.append(StressScenario(pf_iv, dNLV, dS, dIVT1, ts_time_power, df0))
                v_dNLV.append(dNLV)
            figdSdIV.add_trace(go.Scatter(x=v_dSpct, y=v_dNLV, mode='markers', name=f'dIV: {str(dIVT1)}', marker=dict(size=marker_size)), row=1, col=1)

            if add_equity_holdings:
                pf_iv.add_holding(Equity(sym), -int(deltaTotalPf0Total))

                v_dNLV = []
                for dS in v_dS:
                    dNLV = stress_pf(pf_iv, s0, dS, calcDate0, calcDate1, exp_jumps, diffusion_iv, dIVT1=dIVT1, df0=df0.loc[ts_pre_release], ts_time_power=ts_time_power)
                    stress_scenarios.append(StressScenario(pf_iv, dNLV, dS, dIVT1, ts_time_power, df0))
                    v_dNLV.append(dNLV)
                figdSdIV.add_trace(go.Scatter(x=v_dSpct, y=v_dNLV, mode='markers', name=f'dIV + Equity: {str(dIVT1)}', marker=dict(size=marker_size)), row=1, col=1)

                pf_iv.remove_security(equity)

        pf_iv.print_entry_holdings(df0.loc[ts_pre_release], calcDate0)

        # Title subscript of holdings
        holdings_txt = ''
        for k, (sec, q) in enumerate(pf_iv.holdings.items()):
            holdings_txt += f'{sec} {q}, '
            if k > 0 and k % 6 == 0:
                holdings_txt += '<br>'
        figdSdIV.update_layout(title=f'{sym} dS dIV dPL Iteration {i}<br><sup>{holdings_txt}</sup>', xaxis_title=f'dS %', yaxis_title='dPL')

        # Plotting skews
        add_plot_iv(df0, ts_pre_release, pf_i[dflt_dIVT1], figdSdIV, 2, 2)
        add_plot_skew(df0, ts_pre_release, pf_i[dflt_dIVT1], v_dS, figdSdIV, 3, 2)

        add_plot_term_structure(df0, calcDate0, figdSdIV, 2, dIV_levels=v_dIVT1, ts_time_power=ts_time_power)
        # figdSdIV.update_layout(xaxis1_title=f'tenor [days]', yaxis1_title='IV')

        try:
            ### print skew for +/- 10% dS
            for dS in [0.9, 1.1]:
                for sec, q in pf_i[dflt_dIVT1].holdings.items():
                    print(f'{sec} {q} skew {dS:.2f}: {df0.loc[(ts_pre_release, sec.expiry, sec.optionContract.strike, sec.right)][f"div_skew_ds_{dS:.2f}"]}')
        except Exception as e:
            pass

        if v_ts_post_release:
            for dt, group in itertools.groupby(v_ts_post_release, lambda x: x.date()):
                v_ts = list(group)
                v_dSpct_actual = [100 * (df1.loc[ts]['spot'].iloc[0] / s0 - 1) for ts in v_ts]
                v_dPL_pf_actual = [simulate_pf_pnl(pf_i[dflt_dIVT1], df0.loc[ts_pre_release], df1.loc[ts], ts) for ts in v_ts]
                figdSdIV.add_trace(go.Scatter(x=v_dSpct_actual, y=v_dPL_pf_actual, mode='markers', name=f'{dt} v_dPL_pf_actual', marker=dict(size=marker_size)), row=1, col=1)

            add_plot_term_structure(df1, calcDate1, figdSdIV, 2, dIV_levels=(0,), ts_time_power=ts_time_power)

        figures.append(figdSdIV)
        show(figdSdIV, f'''{sym}_{release_date.strftime('%y%m%d')}_figdSdIV_iteration_{i}.html''')

    figIvOverTimeHoldings = go.Figure()
    if v_ts_post_release:
        df_pre_post = df.sort_index().loc[(slice(pd.Timestamp(ts_pre_release.date()), pd.Timestamp(ts_post_release.date() + datetime.timedelta(days=2))))]
    else:
        df_pre_post = df.sort_index().loc[(slice(pd.Timestamp(ts_pre_release.date()), pd.Timestamp(ts_pre_release.date() + datetime.timedelta(days=2))))]
    for i, (sec, q) in enumerate(pf_i[dflt_dIVT1].holdings.items()):
        if isinstance(sec, Equity):
            continue
        s_df = df_pre_post.loc[(slice(None), sec.expiry, sec.optionContract.strike, sec.right)]
        # ix = df.index[(df['iv'] > df['iv'].quantile(0.05)) & (df['iv'] < _df['iv'].quantile(0.95))]
        ix = s_df.index
        figIvOverTimeHoldings.add_trace(go.Scatter(x=ix, y=s_df.loc[ix, 'mid_iv'], mode='markers', marker=dict(size=marker_size), name=f'{str(sec)}'))
        figIvOverTimeHoldings.add_vline(x=datetime.datetime.fromisoformat(release_date.isoformat()) + datetime.timedelta(hours=16), line_dash='dash', line_color='red')
    figIvOverTimeHoldings.update_layout(title=f'Holdings IV over Time', xaxis_title='Time', yaxis_title='IV by expiry')
    show(figIvOverTimeHoldings, f'{sym}_{release_date.strftime("%y%m%d")}_iv_over_time_holdings.html')

    print(f'Done {sym}')


if __name__ == '__main__':
    run()
