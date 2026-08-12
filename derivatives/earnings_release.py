import itertools
import os
import numpy as np
import pandas as pd
import scipy.stats
import plotly.graph_objects as go

from collections import defaultdict
from pprint import pprint
from decimal import Decimal
from functools import lru_cache
from datetime import datetime, date, timedelta, time
from typing import List, Dict, Callable, Tuple, Union
from plotly.subplots import make_subplots
from dataclasses import dataclass
from itertools import combinations
from options.types.earnings_config import EarningsConfig
from options.types.holding import Holding
from derivatives.kalman import enrich_kalman, kalman_my_bid_col, kalman_my_ask_col, fit_kalman_states
from options.types.security import Security
from options.volatility.estimators.earnings_iv_drop_poly_regressor import EarningsIVDropPolyRegressorV3
from options.helper import spread_pc, security_from_symbol
from options.volatility.pf_opt_minlp_pyomo import derive_portfolio_milnp, PfMilnpResult
from shared.modules.logger import logger

from options.helper import val_from_df, moneyness_iv, atm_iv, year_quarter, enrich_atm_iv_by_right, get_moneyness, get_dividend_yield, get_moneyness_fwd_ln, get_tenor
from options.types.portfolio import Portfolio
from shared.paths import Paths
from shared.plotting import show
from shared.constants import EarningsPreSessionDates, model_nm_earnings_iv_drop_regressor, DiscountRateMarket
from options.types.equity import Equity
from options.types.option_frame import OptionFrame
from options.types.option_contract import OptionContract
from options.types.option import Option


class Scenario:
    worst = 'worst'
    best = 'best'
    estimated = 'estimated'
    mid = 'mid'
    fills = 'fills'


def next_release_date(sym: str, dt: date) -> date:
    release_dates = EarningsPreSessionDates(sym)
    return next((d for d in release_dates if d >= dt), None)


def get_m_spread_pc_credit(cfg: EarningsConfig, scoped_options: List[Option], calc_date1: date, s: float) -> np.ndarray:
    m_spread = []
    for ds_ret in cfg.v_ds_ret:
        m_spread.append([spread_pc(get_moneyness(float(o.strike), s * ds_ret), get_tenor(o.expiry, calc_date1)) for o in scoped_options])
    return np.array(m_spread)


def quote_transformations(df: pd.DataFrame):
    df = df[df['ask_iv'] >= df['bid_iv']]

    # Enrich ATM IV
    if 'atm_iv_by_right' not in df.columns:
        enrich_atm_iv_by_right(df, 'atm_iv_by_right')
    # df['d_iv_ratio'] = cfg.earnings_iv_drop_regressor.predict(df[['moneyness', 'tenor']])
    # df['mid_iv_1'] = df['mid_iv'] * df['d_iv_ratio']
    # df['d_iv_ts'] = df['atm_iv_by_right'] * d_iv_pct
    return df


def trade_transformations(df: pd.DataFrame, cfg: EarningsConfig):
    sym = cfg.sym
    rate = DiscountRateMarket
    dividend_yield = get_dividend_yield(sym)
    net_yield = rate - dividend_yield

    df['strike_flt'] = df.index.get_level_values('strike').astype(float)
    kalman_states = fit_kalman_states(df, cfg.release_date, net_yield)
    enrich_kalman(df, kalman_states, net_yield)
    return df


def get_earnings_release_pf(cfg: EarningsConfig) -> dict:
    """
        Refinement:
        - The ultimate short is exercised 1 day at EOD before release date. Also here can improve, research when the time value is high.
        - Generalize. also investigate portfolios earning the term structure IV crush.

        # Questions:
        # - What's the optimal T-x days to sell near expiry options? Understands how far values change. How intrinsic value changes as release date comes up.
        # - Optimal near/long hedge ratios? Risk / Pnl Trade off... relate risk space and cost to hedge?
        """
    if cfg.earnings_iv_drop_regressor is None:
        raise ValueError('Earnings IV Drop Regressor not set')

    sym = cfg.sym
    equity = Equity(sym)
    release_date = cfg.release_date
    holdings = [Holding(str(sec), q) for sec, q in cfg.portfolio.items()]

    option_frame = OptionFrame.load_frame(equity, cfg.resolution, cfg.seq_ret_threshold, year_quarter(release_date))
    df = quote_transformations(option_frame.df_options.sort_index())
    df = exclude_outlier_quotes(df, holdings, equity)

    option_frame.df_option_trades = trade_transformations(option_frame.df_option_trades, cfg)
    option_frame.df_option_trades = exclude_outlier_quotes(option_frame.df_option_trades, holdings, equity)

    ts = df.index.get_level_values('ts').unique()

    # adjustment. hedges can choose on which date. sell high IV is before release to center around underlying price.
    if release_date in [el.date() for el in df.index.get_level_values('ts').unique()]:
        v_ts_pre_release = [i for i in ts if i.date() == release_date]  # preparing on day of expiry
    else:
        v_ts_pre_release = [i for i in ts if i.date() <= release_date]  # preparing before release date
    # v_ts_pre_release = [i for i in ts if i.date() == (release_date - timedelta(days=1))]  # preparing options the night before

    # avoid before 10 am because rolling skew is still very noise
    v_ts_pre_release = [i for i in v_ts_pre_release if i.hour >= 10]

    v_ts_post_release = [i for i in ts if i.date() >= (release_date + timedelta(days=1))]
    # avoid after 4 pm because rolling skew is still very noise
    v_ts_post_release = [i for i in v_ts_post_release if i.hour >= 10]
    first_2_days = sorted(list(set([i.date() for i in v_ts_post_release])))[:2]
    v_ts_post_release = [i for i in v_ts_post_release if i.date() in first_2_days]
    # ts_pre_release = v_ts_pre_release[-1]
    ts_pre_release = v_ts_pre_release[1]
    # ts_pre_release = v_ts_pre_release[len(v_ts_pre_release)//2]
    print(f'Release date: {release_date}. ts_pre_release: {ts_pre_release}')

    ts_post_release = None
    if v_ts_post_release:
        ts_post_release = v_ts_post_release[0]
        # Need to any options that are not captured in that timestamp
        print(f'TS Post Release - Min: {np.min(v_ts_post_release)} Max: {np.max(v_ts_post_release)}')

    calc_date0 = ts_pre_release.date()
    calc_date1 = release_date + timedelta(days=1)

    # split frame into pre and post release
    df0 = df.loc[[ts_pre_release]]
    df1 = df.loc[v_ts_post_release] if v_ts_post_release else None

    # Spot 1 and Spot 2
    s0 = df0.loc[ts_pre_release].iloc[0]['spot']
    s1 = df1.loc[ts_post_release].iloc[0]['spot'] if v_ts_post_release else np.nan
    print(f'''Spot0: {s0}; Spot1: {s1}; Stock jumped by dS: {round(s1 - s0, 1)} | {round(100 * (s1 / s0 - 1), 1)} %''')

    expiries = list(sorted(set(df0.index.get_level_values('expiry').values)))
    exp_jump1 = sorted([i for i in expiries if i >= release_date])[0]
    pf = Portfolio(cfg.portfolio)

    option_universe, cache_iv0_bid, cache_iv0_ask, cache_iv1_bid, cache_iv1_ask = create_option_universe_iv_caches(sym, ts_pre_release, df0, s0, cfg.v_ds_ret, cfg, holdings=pf.get_holdings())

    # Picking Median Mid IV for every option
    # df0_median = get_df_median(df[df.index.get_level_values('ts').date == ts_pre_release.date()])
    # option_universe, cache_iv0_bid, cache_iv0_ask, cache_iv1_bid, cache_iv1_ask = create_option_universe_iv_caches(
    #     sym, ts_pre_release, df0_median, s0, cfg.v_ds_ret, cfg, holdings=pf.get_holdings(), df0_median=df0_median.droplevel(0))

    # inf_vol = infinite_volatility(df)
    # print(f'Max Expiry ATM Volatility: {inf_vol.mean()}')
    # if cfg.plot:
    #     power_law_term_structure(df, exp_jump1)

    # diffusion_iv = estimate_diffusion_iv(df, release_date)
    # print(f'Nearest Expiry: {exp_jump1}, {sym.upper()}, ATM IV: {atm_iv(df0.loc[ts_pre_release], exp_jump1, s0)}, Diffusion IV: {diffusion_iv}')
    # # This result seems too low, compared to IB emails...
    # implied_jump = implied_ds(np.sqrt(atm_iv(df0.loc[ts_pre_release], exp_jump1, s0) ** 2 - diffusion_iv ** 2))
    # print(f'implied jump return: {100 * implied_jump :.2f} %; implied jump dS={implied_jump * s0 :.2f}')

    # Spot price series1 internal value, eventually weights also...
    if cfg.plot:
        plot_spot(df, df0, exp_jump1, release_date, ts_pre_release, s0, cfg.sym)

    scoped_expiries = get_scoped_expiries(expiries)
    get_p0 = lambda o: val_from_df(df0.loc[ts_pre_release], o.expiry, o.optionContract.strike, o.right, 'mid_price')
    scoped_options = [o for o in option_universe if
                      option_in_pf_scope(o, cfg, calc_date0, s0, get_p0(o), scoped_expiries=scoped_expiries, pf=pf)]

    print(f'Scoped options: #{len(scoped_options)} out of # {len(option_universe)}')

    v_delta0_bid, v_delta0_ask, m_dnlv01_worst_buy, m_dnlv01_worst_sell, m_dnlv01_best_buy, m_dnlv01_best_sell = get_nlv_delta_matrix(s0, calc_date0, calc_date1, cfg, cache_iv0_bid, cache_iv0_ask, cache_iv1_bid, cache_iv1_ask, scoped_options)
    m_spread_pc_credit = get_m_spread_pc_credit(cfg, scoped_options, calc_date0, s0)
    m_dnlv01_estimated_buy, m_dnlv01_estimated_sell = get_estimate_m_dnlv_buy_sell(m_dnlv01_worst_buy, m_dnlv01_best_buy, m_dnlv01_worst_sell, m_dnlv01_best_sell, m_spread_pc_credit)
    v_delta0_mid = (v_delta0_bid + v_delta0_ask) / 2

    if cfg.run_solver:
        cfg_holdings_dct = {str(o): q for o, q in pf.holdings.items()}
        res: PfMilnpResult = derive_portfolio_milnp(
            scoped_options, m_dnlv01_buy=m_dnlv01_estimated_buy, m_dnlv01_sell=m_dnlv01_estimated_sell, cfg=cfg,
            f_weight_ds=get_f_weight_ds(), holdings=pf.get_holdings()
        )
        pf = Portfolio({security_from_symbol(s, calc_date1): q for s, q in res.holdings.items()})
        if cfg_holdings_dct:
            new_holdings = {o: q - cfg_holdings_dct.get(str(o), 0) for o, q in pf.holdings.items() if (q - cfg_holdings_dct.get(str(o), 0)) != 0}
            print(f'Added holdings by solver: {new_holdings}')

    presumed_fill_ivs = get_assumed_fill_iv(m_spread_pc_credit, pf.get_holdings(), scoped_options, cache_iv0_bid, cache_iv0_ask)
    pprint(f'presumed_fill_ivs: {presumed_fill_ivs}')

    stress_test_ds_worst = get_stress_test_ds(scoped_options, pf, m_dnlv01_worst_buy, m_dnlv01_worst_sell, v_delta0_mid, cfg, s0)
    stress_test_ds_estimated: StressTestDsResult = get_stress_test_ds(scoped_options, pf, m_dnlv01_estimated_buy, m_dnlv01_estimated_sell, v_delta0_mid, cfg, s0)
    stress_test_ds_best = get_stress_test_ds(scoped_options, pf, m_dnlv01_best_buy, m_dnlv01_best_sell, v_delta0_mid, cfg, s0)

    print(f'Portfolio w/o  Equity DeltaTotalPf0Total: {stress_test_ds_estimated.delta_total}, deltaTotalPf0Regressed: {stress_test_ds_estimated.delta_total_across_ds}')
    print('marginal_scaled_objective_by_holding')
    str_holdings = {str(k): v for k, v in pf.holdings.items()}
    for k, v in stress_test_ds_estimated.marginal_weighted_objective_by_holding.items():
        q_holding = str_holdings.get(k, 0)
        print(f'{k}: {(q_holding * v)/abs(q_holding):.2f}')
    print('-'*30)

    if cfg.plot:
        # Stressing the portfolio for a few more dIV level changes for plots
        plot_stressed_pnl(pf, stress_test_ds_estimated.delta_total_across_ds, release_date, ts_pre_release, v_ts_post_release, df1, s0, calc_date0, calc_date1, df0, cfg,
                          stress_test_ds_worst, stress_test_ds_estimated, stress_test_ds_best, option_frame)

        # plot_iv_over_time(pf, df, v_ts_post_release, ts_pre_release, ts_post_release, release_date, sym)

    print(f'Done earnings release {sym}')
    pprint_spread_usd_by_sym(m_dnlv01_best_buy, m_dnlv01_worst_buy, m_dnlv01_best_sell, m_dnlv01_worst_sell, pf, scoped_options)

    return pf.holdings


def get_f_weight_ds():
    """Rewarding risk reduction around implied jump dS move more than large dS moves"""
    # f_weight_ds = get_f_weight_ds_ret(implied_jump * 2, plot=cfg.plot)
    # f_weight_ds = get_f_weight_ds_ret(0.1, plot=cfg.plot)
    f_weight_ds = None
    return f_weight_ds


def pprint_spread_usd_by_sym(m_dnlv01_best_buy, m_dnlv01_worst_buy, m_dnlv01_best_sell, m_dnlv01_worst_sell, pf, scoped_options):
    m_dnlv01_spread_buy = m_dnlv01_best_buy - m_dnlv01_worst_buy
    m_dnlv01_spread_sell = m_dnlv01_best_sell - m_dnlv01_worst_sell

    v_spread_buy = np.maximum(np.array([1 if pf.portfolio.get(o, 0) > 0 else 0 for o in scoped_options]), 0) * m_dnlv01_spread_buy[10]
    v_spread_sell = np.minimum(np.array([-1 if pf.portfolio.get(o, 0) < 0 else 0 for o in scoped_options]), 0) * m_dnlv01_spread_sell[10]
    sym_q_buy = [(scoped_options[i], spread_usd) for i, spread_usd in enumerate(v_spread_buy) if spread_usd != 0]
    sym_q_sell = [(scoped_options[i], spread_usd) for i, spread_usd in enumerate(v_spread_sell) if spread_usd != 0]

    pprint(dict(sym_q_buy))
    pprint(dict(sym_q_sell))


def get_assumed_fill_iv(m_spread_pc_credit, pf: Portfolio, scoped_options: List[Option], cache_iv0_bid, cache_iv0_ask) -> Dict[Security, float]:
    scoped_options = [str(o) for o in scoped_options]
    cache_iv0_bid_str = {str(k): v for k, v in cache_iv0_bid.items()}
    cache_iv0_ask_str = {str(k): v for k, v in cache_iv0_ask.items()}
    v_iv0_ask = np.array([cache_iv0_bid_str.get(o, {1: 0})[1] for o in scoped_options])
    v_iv0_bid = np.array([cache_iv0_ask_str.get(o, {1: 0})[1] for o in scoped_options])
    v_spread_pc_credit = m_spread_pc_credit[m_spread_pc_credit.shape[0] // 2]
    spread_iv0 = v_iv0_ask - v_iv0_bid

    v_buy_iv = v_iv0_ask - spread_iv0 * v_spread_pc_credit
    v_sell_iv = v_iv0_bid + spread_iv0 * v_spread_pc_credit
    dct_buy_ivs = {h.symbol: v_buy_iv[scoped_options.index(h.symbol)] for h in pf if h.quantity > 0}
    dct_sell_ivs = {h.symbol: v_sell_iv[scoped_options.index(h.symbol)] for h in pf if h.quantity < 0}
    return {**dct_buy_ivs, **dct_sell_ivs}


def get_nlv_delta_matrix(s0: float, calc_date0: date, calc_date1: date, cfg: EarningsConfig,
                         cache_iv0_bid: Dict, cache_iv0_ask: Dict, cache_iv1_bid: Dict, cache_iv1_ask: Dict, scoped_options: List[Option]):
    nlv0_bid = np.array([nlv(o, s0, cache_iv0_bid[o][1], 1, calc_date0) for o in scoped_options])
    nlv0_ask = np.array([nlv(o, s0, cache_iv0_ask[o][1], 1, calc_date0) for o in scoped_options])

    dnlv01_worst_long = np.array([[nlv(o, s0 * ds_ret, cache_iv1_bid[o][ds_ret], 1, calc_date1) for o in scoped_options] - nlv0_ask for ds_ret in cfg.v_ds_ret])
    dnlv01_worst_short = np.array([[nlv(o, s0 * ds_ret, cache_iv1_ask[o][ds_ret], 1, calc_date1) for o in scoped_options] - nlv0_bid for ds_ret in cfg.v_ds_ret])

    dnlv01_best_long = np.array([[nlv(o, s0 * ds_ret, cache_iv1_ask[o][ds_ret], 1, calc_date1) for o in scoped_options] - nlv0_bid for ds_ret in cfg.v_ds_ret])
    dnlv01_best_short = np.array([[nlv(o, s0 * ds_ret, cache_iv1_bid[o][ds_ret], 1, calc_date1) for o in scoped_options] - nlv0_ask for ds_ret in cfg.v_ds_ret])

    v_delta0_bid = np.array([delta(o, s0, cache_iv0_bid[o][1], 1, calc_date0) for o in scoped_options])
    v_delta0_ask = np.array([delta(o, s0, cache_iv0_ask[o][1], 1, calc_date0) for o in scoped_options])

    return v_delta0_bid, v_delta0_ask, dnlv01_worst_long, dnlv01_worst_short, dnlv01_best_long, dnlv01_best_short


def get_estimate_m_dnlv_buy_sell(m_dnlv01_worst_buy, m_dnlv01_best_buy, m_dnlv01_worst_sell, m_dnlv01_best_sell, m_spread_pc_credit: np.ndarray):
    m_spread_buy = m_dnlv01_best_buy - m_dnlv01_worst_buy
    m_spread_sell = m_dnlv01_best_sell - m_dnlv01_worst_sell

    m_dnlv01_buy_estimated = m_dnlv01_worst_buy + m_spread_buy * m_spread_pc_credit
    m_dnlv01_sell_estimated = m_dnlv01_worst_sell + m_spread_sell * m_spread_pc_credit
    # return m_dnlv01_worst_buy, m_dnlv01_worst_sell
    return m_dnlv01_buy_estimated, m_dnlv01_sell_estimated


def get_scoped_expiries(expiries: List[date]) -> List[date]:
    # Drop some weeklies to reduce scope. Pick 2 closest expiries. Then last of each month
    scoped_expiries = expiries[:2]
    for i, exp in itertools.groupby(expiries, lambda x: x.month):
        months_last_exp = list(exp)[-1]
        if months_last_exp.month not in [dt.month for dt in scoped_expiries]:
            scoped_expiries.append(months_last_exp)
    print(f'Scoped expiries: #{len(scoped_expiries)} out of #{len(expiries)}, {scoped_expiries}')
    return scoped_expiries


def median(ps: pd.Series, skipna=True):
    _ps = ps[~ps.isna()] if skipna else ps
    if len(_ps) == 0:
        return np.nan
    return sorted(_ps)[len(_ps) // 2]


def get_df_median(df0):
    """ps0 where mid_iv == median"""
    ps_lst = []
    for expiry, s_df in df0.groupby('expiry'):
        for strike, s_df2 in s_df.groupby('strike'):
            for right, s_df3 in s_df2.groupby('right'):
                mid_iv = s_df3['mid_iv']
                mid_iv_median = median(mid_iv)
                df_median = s_df3.loc[mid_iv == mid_iv_median]
                if len(df_median) > 0:
                    ps_lst.append(df_median.iloc[0])
                else:
                    print('None el')
    df = pd.concat(ps_lst, axis=1).transpose()
    df.index = df.index.set_names(['ts', 'expiry', 'strike', 'right'])
    return df


def create_option_universe_iv_caches(sym, ts_pre_release, df0, s0, v_ds_ret, cfg, pf: Portfolio = None, df0_median=None):
    release_date = next_release_date(sym, ts_pre_release.date())
    calcDate0 = ts_pre_release.date()
    calcDate1 = release_date + timedelta(days=1)
    option_universe = set()

    df0_clean = df0[~df0['mid_price'].isna()]
    # 1st move into separate function
    # Here, should apply scope filters
    for ix in df0_clean.index:
        ts, expiry, strike, right = ix
        if expiry <= release_date:
            continue
        option_universe.add(get_option(sym, expiry, strike, right, calcDate0))
    print(f'Added {len(option_universe)} options to universe')

    # 2nd move into separate function. Consider using median IV to avoid outliers.
    df0_clean = df0_median if df0_median is not None else df0_clean.loc[ts_pre_release]

    cache_iv0_bid = {sec: {1: iv_as_of(sec, df0_clean, calcDate0, calcDate0, 1, cfg, 'bid')} for sec in option_universe.union({Equity(sym)})}
    cache_iv0_ask = {sec: {1: iv_as_of(sec, df0_clean, calcDate0, calcDate0, 1, cfg, 'ask')} for sec in option_universe.union({Equity(sym)})}
    cache_iv1_bid = {sec: {ds_ret: iv_as_of(sec, df0_clean, calcDate0, calcDate1, ds_ret, cfg, 'bid') for ds_ret in v_ds_ret} for sec in option_universe.union({Equity(sym)})}
    cache_iv1_ask = {sec: {ds_ret: iv_as_of(sec, df0_clean, calcDate0, calcDate1, ds_ret, cfg, 'ask') for ds_ret in v_ds_ret} for sec in option_universe.union({Equity(sym)})}

    # 3rd move into separate function
    # kick out options for which iv values are missing
    len(option_universe)
    rm_option = {o for o in option_universe if pd.isna(pd.Series(cache_iv1_bid[o].values())).sum() > 0}
    rm_option.union({o for o in option_universe if pd.isna(pd.Series(cache_iv1_ask[o].values())).sum() > 0})
    rm_option.union({o for o in option_universe if pd.isna(pd.Series(cache_iv0_bid[o].values())).sum() > 0})
    rm_option.union({o for o in option_universe if pd.isna(pd.Series(cache_iv0_ask[o].values())).sum() > 0})
    if pf:
        rm_option = [o for o in rm_option if o not in pf.holdings.keys()]
    len(rm_option)

    print(f'Kicking out {len(rm_option)} options from option universe due to missing IV values: {rm_option}')
    for o in rm_option:
        if o in cache_iv0_bid:
            cache_iv0_bid.pop(o)
        if o in cache_iv0_ask:
            cache_iv0_ask.pop(o)
        if o in cache_iv1_bid:
            cache_iv1_bid.pop(o)
        if o in cache_iv1_ask:
            cache_iv1_ask.pop(o)
        if str(o) in option_universe:
            option_universe.remove(o)

    return option_universe, cache_iv0_bid, cache_iv0_ask, cache_iv1_bid, cache_iv1_ask


def title_stressed_pnl_plot(pf, deltaTotalPf0, ts_pre_release):
    # Title subscript of holdings
    sym = pf.underlying
    holdings_txt = ''
    for k, (sec, q) in enumerate(pf.portfolio.items()):
        holdings_txt += f'{sec} {q}, '
        if k > 0 and k % 6 == 0:
            holdings_txt += '<br>'
    holdings_txt += f' DeltaTotalPf0Total: {deltaTotalPf0:.2f}'
    return f'{sym} dS dIV dPL ts0: {ts_pre_release}<br><sup>{holdings_txt}</sup>'


def plot_stressed_pnl(pf: Portfolio, deltaTotalPf0, release_date, ts_pre_release, v_ts_post_release, df1, s0, calcDate0, calcDate1, df0, cfg: EarningsConfig,
                          stress_test_ds_worst, stress_test_ds_estimated, stress_test_ds_best, option_frame: OptionFrame, marker_size=4):
    sym = pf.underlying
    figdSdIV = make_subplots(rows=3, cols=2, specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}]],
                             subplot_titles=['dS dIV dPL', 'IV ATM Horizontal', 'IV Vertical (K/S)',
                                             'Simulated PnL (mid->mid) vs PnL Estimated (mid->estimated bid/ask); vertical dash',
                                             'Simulated IV (mid->mid); Arrows vs IV Estimated (mid->estimated bid/ask); Vertical dash'])

    # Traces using the iv crush model.
    x = [100*(x-1) for x in stress_test_ds_worst.ds_dnlv.keys()]
    y_worst = [dnlv + s0 * (ds_ret-1) * -stress_test_ds_worst.delta_total_across_ds for ds_ret, dnlv in stress_test_ds_worst.ds_dnlv.items()]
    y_estimated = [dnlv + s0 * (ds_ret-1) * -stress_test_ds_estimated.delta_total_across_ds for ds_ret, dnlv in stress_test_ds_estimated.ds_dnlv.items()]
    y_best = [dnlv + s0 * (ds_ret-1) * -stress_test_ds_best.delta_total_across_ds for ds_ret, dnlv in stress_test_ds_best.ds_dnlv.items()]
    figdSdIV.add_trace(go.Scatter(x=x, y=y_worst, mode='markers', name=f'dIV: model worst fill', marker=dict(size=marker_size)), row=1, col=1)
    figdSdIV.add_trace(go.Scatter(x=x, y=y_estimated, mode='markers', name=f'dIV: model estimated fill', marker=dict(size=marker_size)), row=1, col=1)
    figdSdIV.add_trace(go.Scatter(x=x, y=y_best, mode='markers', name=f'dIV: model best fill', marker=dict(size=marker_size)), row=1, col=1)

    title = title_stressed_pnl_plot(pf, deltaTotalPf0, ts_pre_release.isoformat())
    figdSdIV.update_layout(title=title, xaxis_title=f'dS %', yaxis_title='dPL')

    # Plotting skews
    add_plot_iv(df0, ts_pre_release, pf, figdSdIV, 2, 2)
    # add_plot_skew(df0, ts_pre_release, pf, figdSdIV, 3, 2)
    add_plot_term_structure(df0, calcDate0, figdSdIV, 2, cfg, dIV_levels=cfg.v_dIVT1, ts_time_power=cfg.ts_time_power)
    if df1 is not None:
        add_plot_term_structure(df1, calcDate1, figdSdIV, 2, cfg, dIV_levels=(None,), ts_time_power=cfg.ts_time_power, name_prefix='Predicted ')

    if v_ts_post_release:
        # Adding equity
        pf.add_holding(option_frame.equity, -int(stress_test_ds_estimated.delta_total_across_ds))

        df_trade = option_frame.df_option_trades
        v_ts_release_date_trade = np.unique([ts for ts in df_trade.index.get_level_values('ts') if ts.date() == release_date])
        v_ts_release_date_quote = np.unique([ts for ts in df0.index.get_level_values('ts') if ts.date() == release_date])
        df0_trade = df_trade.loc[v_ts_release_date_trade]
        df0_quote = df0.loc[v_ts_release_date_quote]
        for dt, group in itertools.groupby(v_ts_post_release, lambda x: x.date()):
            v_ts_quote = list(group)
            v_ts_trade = np.unique([ts for ts in df_trade.index.get_level_values('ts') if ts.date() == dt])
            df1_quote = df1.loc[v_ts_quote]
            df1_trade = df_trade.loc[v_ts_trade]

            # print('new' + '-' * 80)
            # dct_dlnv = simulate_pf_pnl_new(pf, df0_quote, df0_trade, df1_quote, df1_trade, scenario=Scenario.mid)
            # x = []
            # y = []
            # p0 = df0_quote['spot'].iloc[-1]
            # for ts, v_dlnv in dct_dlnv.items():
            #     get_v_ts = lambda df, ts: df.index.get_level_values('ts')[(ts < df.index.get_level_values('ts')) &  (df.index.get_level_values('ts') < ts + timedelta(hours=1))]
            #     if len(get_v_ts(df1_trade, ts)) > 0:
            #         x += [100 * (df1_trade.loc[get_v_ts(df1_trade, ts), 'spot'].iloc[-1] / p0 - 1)] * len(v_dlnv)
            #     elif len(get_v_ts(df1_quote, ts)) > 0:
            #         x += [100 * (df1_quote.loc[get_v_ts(df1_quote, ts), 'spot'].iloc[-1] / p0 - 1)] * len(v_dlnv)
            #     else:
            #         continue
            #     y += list(v_dlnv)
            # figdSdIV.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'{dt} v_dPL_pf_actual NEW', marker=dict(size=2)), row=1, col=1)

            try:
                print('old' + '-' * 80)
                v_dnlv = simulate_pf_pnl(pf, df0_quote, df0_trade, df1_quote, df1_trade, scenario=Scenario.worst)
                x = 100 * (np.random.choice(df1_trade['spot'], len(v_dnlv)) / np.random.choice(df0_trade['spot'], len(v_dnlv)) - 1)
                figdSdIV.add_trace(go.Scatter(x=x, y=v_dnlv, mode='markers', name=f'{dt} v_dPL_pf_actual MID', marker=dict(size=2)), row=1, col=1)
            except Exception as e:
                print(e)

            try:
                print('KALMAN' + '-'*80)
                v_dnlv = simulate_pf_pnl_kalman(pf, df0_quote, df0_trade, df1_quote, df1_trade, scenario=Scenario.worst)
                x = 100 * (np.random.choice(df1_trade['spot'], len(v_dnlv)) / np.random.choice(df0_trade['spot'], len(v_dnlv)) - 1)
                figdSdIV.add_trace(go.Scatter(x=x, y=v_dnlv, mode='markers', name=f'{dt} v_dPL_pf_actual KALMAN', marker=dict(size=2)), row=1, col=1)
            except Exception as e:
                print(e)

            try:
                print('KALMAN afternoon' + '-' * 80)
                v_dnlv = simulate_pf_pnl_kalman(pf, df0_quote, df0_trade, df1_quote.loc[ts_scoped(df1_quote)], df1_trade.loc[ts_scoped(df1_trade)], scenario=Scenario.worst)
                x = 100 * (np.random.choice(df1_trade['spot'], len(v_dnlv)) / np.random.choice(df0_trade['spot'], len(v_dnlv)) - 1)
                figdSdIV.add_trace(go.Scatter(x=x, y=v_dnlv, mode='markers', name=f'{dt} v_dPL_pf_actual KALMAN Afternoon', marker=dict(size=2)), row=1, col=1)
            except Exception as e:
                print(e)

        # show(figdSdIV, f'''{sym}_{release_date.strftime('%y%m%d')}_figdSdIV_iteration.html''')

        add_plot_term_structure(df1, calcDate1, figdSdIV, 2, cfg, dIV_levels=(0,), ts_time_power=cfg.ts_time_power, name_prefix='Realized ')

        # For each position in pf, plot the realized and estimated IVs and PnLs
        iv_traces = []
        pnl_traces = []
        for sec, q in pf.holdings.items():
            if isinstance(sec, Equity):
                continue
            elif isinstance(sec, Option):
                s1 = df1.loc[v_ts_post_release].iloc[0]['spot'].mean()
                tenor_ = get_tenor(sec.expiry, ts_pre_release.date())

                iv0 = val_from_df(df0.loc[ts_pre_release], sec.expiry, sec.optionContract.strike, sec.right, 'mid_iv')  # not correct. ignore df. bad values
                iv1_estimated_bid, iv1_estimated_ask = estimate_iv1(df0.loc[ts_pre_release], sec, s0, s1, tenor_, cfg=cfg, calc_date0=calcDate0)
                iv1 = df1.loc[(v_ts_post_release, sec.expiry, sec.optionContract.strike, sec.right), 'mid_iv'].mean()

                price0 = val_from_df(df0.loc[ts_pre_release], sec.expiry, sec.optionContract.strike, sec.right, 'mid_price')
                estimated_price1_bid = sec.npv(iv1_estimated_bid, s1, ts_pre_release.date())
                estimated_price1_ask = sec.npv(iv1_estimated_ask, s1, ts_pre_release.date())
                price1 = df1.loc[(v_ts_post_release, sec.expiry, sec.optionContract.strike, sec.right), 'mid_price'].mean()

                moneyness0 = get_moneyness(sec.strike, s0)
                moneyness1 = get_moneyness(sec.strike, s1)
                iv_traces.append(go.Scatter(x=[moneyness0, moneyness1], y=[iv0, iv1], mode='markers+lines', name=f'{q} {sec} IV0->IV1',
                                            marker=dict(size=8, symbol="arrow-bar-up", angleref="previous")))
                iv_traces.append(go.Scatter(x=[moneyness1]*2, y=[iv1_estimated_bid, iv1_estimated_ask], name=f'{q} {sec} IV1EstBid->IV1EstAsk',
                                            mode='markers+lines', marker=dict(size=6, symbol="circle-open-dot"), line=dict(width=3, dash='dash')))

                pnl_traces.append(go.Scatter(x=[moneyness1], y=[(price1 - price0) * 100 * q], mode='markers', name=f'{q} {sec} PnL', marker=dict(size=8, symbol="diamond-open-dot")))
                pnl_traces.append(go.Scatter(x=[moneyness1]*2, y=(np.array([estimated_price1_bid, estimated_price1_ask]) - price0) * 100 * q,
                                             mode='markers+lines', name=f'{q} {sec} PnLEstBid<->PnLEstAsk', marker=dict(size=6, symbol="circle-open-dot"), line=dict(width=3, dash='dash')))

        for t in pnl_traces:
            figdSdIV.add_trace(t, row=3, col=1)

        for t in iv_traces:
            figdSdIV.add_trace(t, row=3, col=2)

    show(figdSdIV, f'''{sym}_{release_date.strftime('%y%m%d')}_figdSdIV_iteration.html''')


def median_iv(df, expiry, strike, right, col_nm='fill_iv'):
    return df.loc[(slice(None), expiry, strike, right), col_nm].median()


def kalman_report(df, pf: Portfolio):
    df['ts'] = df.index.get_level_values('ts')
    y_col_nm = 'fill_iv'
    kalman_mean_col = 'fill_iv_kalman_mean'
    kalman_up_col = 'fill_iv_kalman_up'
    kalman_low_col = 'fill_iv_kalman_low'
    kalman_my_ask_col = 'fill_iv_kalman_my_ask'
    kalman_my_bid_col = 'fill_iv_kalman_my_bid'
    print(df.shape)
    # sec, q = next(iter(pf.items()))
    for sec, q in pf.items():
        print(f'{sec} {q}')
        if isinstance(sec, Equity):
            continue
        else:
            df_o = df.loc[(slice(None), sec.expiry, sec.optionContract.strike, sec.right)]
            df_o = df_o[~df_o['ts'].duplicated()]

            # df_o[[kalman_my_bid_col, y_col_nm, kalman_my_ask_col]]
            if q > 0:
                print((df_o[kalman_my_bid_col] > df_o[y_col_nm]).sum())
            else:
                print((df_o[kalman_my_ask_col] < df_o[y_col_nm]).sum())


def plot_spot(df, df0, exp_jump1, release_date, ts_pre_release, s0, sym, marker_size=4):
    figSpotTimeValue = make_subplots(rows=2, cols=1, subplot_titles=['Spot', 'Time Value'])
    figSpotTimeValue.add_trace(go.Scatter(x=df.index.get_level_values('ts'), y=df['spot'], mode='lines', name='Spot'), row=1, col=1)
    figSpotTimeValue.add_vline(x=datetime.fromisoformat(release_date.isoformat()) + timedelta(hours=16), line_dash='dash', line_color='red', row=1, col=1)
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
        figSpotTimeValue.add_vline(x=datetime.fromisoformat(release_date.isoformat()) + timedelta(hours=16), line_dash='dash', line_color='red', row=2, col=1)

    show(figSpotTimeValue, f'{sym}_{release_date.strftime("%y%m%d")}_spot_time_value.html')
    ###############################################

    # Analysing IVs meant for hedging
    # Historically, how did the IVs change throughout the day? Entry & exit? For near, how much did time value in USD change? Theta?
    # Historically, how much did the IVs change after the release date? Feasible trade? Entry range, exit range... so just plot IV at ~low res of day.
    # expiries = list(sorted(set(df0.index.get_level_values('expiry').values)))
    # plot_dates = expiries  # release_date + timedelta(days=i) for i in [0, 30, 60, 90, 120, 150, 180, 270, 365, 500]]
    # figIvOverTime = make_subplots(rows=len(plot_dates), cols=1, subplot_titles=[dt.isoformat() for dt in plot_dates])
    # if v_ts_post_release:
    #     df_pre_post = df.sort_index().loc[(slice(pd.Timestamp(ts_pre_release.date()), pd.Timestamp(ts_post_release.date() + timedelta(days=2))))]
    # else:
    #     df_pre_post = df.sort_index().loc[(slice(pd.Timestamp(ts_pre_release.date()), pd.Timestamp(ts_pre_release.date() + timedelta(days=2))))]
    # for i, dt in enumerate(plot_dates):
    #     df_mny1_tenor_dt = iv_at_date_over_time(df_pre_post, dt, 1)
    #     for right, s_df in df_mny1_tenor_dt.groupby('right'):
    #         ix = s_df.index[(s_df['iv'] > s_df['iv'].quantile(0.05)) & (s_df['iv'] < s_df['iv'].quantile(0.95))]
    #         figIvOverTime.add_trace(go.Scatter(x=s_df.loc[ix, 'ts'], y=s_df.loc[ix, 'iv'], mode='markers', marker=dict(size=marker_size), name=f'{right}: {dt}'), row=i + 1, col=1)
    #         figIvOverTime.add_vline(x=datetime.fromisoformat(release_date.isoformat()) + timedelta(hours=16), line_dash='dash', line_color='red')
    # figIvOverTime.update_layout(title=f'{sym} ATM IV over Time', xaxis_title='Time', yaxis_title='IV by expiry')
    # show(figIvOverTime, f'''{sym}_{release_date.strftime('%y%m%d')}_iv_over_time.html''')


def plot_iv_over_time(pf, df, v_ts_post_release, ts_pre_release, ts_post_release, release_date, sym, marker_size=4):
    figIvOverTimeHoldings = go.Figure()
    if v_ts_post_release:
        df_pre_post = df.sort_index().loc[(slice(pd.Timestamp(ts_pre_release.date()), pd.Timestamp(ts_post_release.date() + timedelta(days=2))))]
    else:
        df_pre_post = df.sort_index().loc[(slice(pd.Timestamp(ts_pre_release.date()), pd.Timestamp(ts_pre_release.date() + timedelta(days=2))))]
    for i, (sec, q) in enumerate(pf.portfolio.items()):
        if isinstance(sec, Equity):
            continue
        s_df = df_pre_post.loc[(slice(None), sec.expiry, sec.optionContract.strike, sec.right)]
        ix = s_df.index
        figIvOverTimeHoldings.add_trace(go.Scatter(x=ix, y=s_df.loc[ix, 'mid_iv'], mode='markers+lines', marker=dict(size=3), name=f'{str(sec)}'))
        figIvOverTimeHoldings.add_vline(x=datetime.fromisoformat(release_date.isoformat()) + timedelta(hours=16), line_dash='dash', line_color='red')
    figIvOverTimeHoldings.update_layout(title=f'Holdings IV over Time', xaxis_title='Time', yaxis_title='IV by expiry')
    show(figIvOverTimeHoldings, f'{sym}_{release_date.strftime("%y%m%d")}_iv_over_time_holdings.html')


def is_early_stopping_criteria_met(dct_dS_dNLV):
    """If all dS scenarios for selected dIV level drop are positive and most extreme dS are more positive than dS-1, so curving into positive pnl"""
    ds_ret = sorted(dct_dS_dNLV.keys())
    ds_min = ds_ret[0]
    ds_min_1 = ds_ret[1]
    ds_max = ds_ret[-1]
    ds_max_1 = ds_ret[-2]

    return all(dNLV > 0 for dNLV in dct_dS_dNLV.values()) and dct_dS_dNLV[ds_max] > dct_dS_dNLV[ds_max_1] and dct_dS_dNLV[ds_min] > dct_dS_dNLV[ds_min_1]


def implied_ds(dIV: float) -> float:
    return dIV / np.sqrt(365)


def get_f_weight_ds_ret(implied_jump, plot=False) -> Callable:
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


def derive_delta_from_regressing_min_npvs(dct_dS_dNLV, s0):
    v_ds_ret = list(dct_dS_dNLV.keys())
    neg_ds = [dnpv for ds, dnpv in dct_dS_dNLV.items() if 1 > ds > 0.85]
    ix_neg = np.argmin(neg_ds)
    ix_pos_ds = [ds for ds, dnpv in dct_dS_dNLV.items() if 1 < ds < 1.15]
    pos_ds = [dnpv for ds, dnpv in dct_dS_dNLV.items() if ds > 1]
    ix_pos = np.argmin(pos_ds)
    try:
        ds = ix_pos_ds[ix_pos] * s0 - v_ds_ret[ix_neg] * s0
    except IndexError as e:
        print(f'ix_pos_ds: {ix_pos_ds} ix_pos: {ix_pos} pos_ds: {pos_ds} ix_neg: {ix_neg} neg_ds: {neg_ds}')
        return 0
    delta = slope = (pos_ds[ix_pos] - neg_ds[ix_neg]) / ds
    return delta


def option_in_pf_scope(o: Option, cfg: EarningsConfig, calcDate0, s0, p0, scoped_expiries=None, exclude: List[Option]=None, pf: Portfolio=None):
    if pf and str(o) in [str(h) for h in pf.holdings.keys()]:
        return True

    if exclude and o in exclude:
        return False

    # Want large tenors as their IV decreases less after release date.
    if scoped_expiries and o.expiry not in scoped_expiries:
        return False

    if get_tenor(o.expiry, calcDate0) < cfg.min_tenor:
        return False

    if get_tenor(o.expiry, calcDate0) > cfg.max_tenor:
        return False

    moneyness = float(o.strike) / s0
    if (cfg.moneyness_limits and
            (moneyness < cfg.moneyness_limits[0] or moneyness > cfg.moneyness_limits[1])):
        return False

    iv0 = o.iv(p0, s0, calcDate0)
    if iv0 == 0 or np.isnan(iv0):
        return False

    if cfg.abs_delta_limits:
        delta = abs(o.delta(iv0, s0, calcDate0))
        if delta < cfg.abs_delta_limits[0] or delta > cfg.abs_delta_limits[1]:
            return False

    return True


def get_v_iv_vega(df_trade, df_quote, o: Option, q: int, sub_sample_size: int, calc_date: date, quote_col_nm, trade_col_nm='fill_iv', scenario: Scenario|str = Scenario.worst):
    try:
        v_ix = df_trade.loc[(slice(None), o.expiry, o.optionContract.strike, o.right)].index
        v_ix = np.random.choice(v_ix, sub_sample_size) if len(v_ix) > sub_sample_size else v_ix
        v_iv_vega = np.array([(ps[trade_col_nm], o.vega(ps[trade_col_nm], ps['spot'], calc_date)) for ix, ps in df_trade.loc[(v_ix, o.expiry, o.optionContract.strike, o.right)].iterrows()])
        if len(v_iv_vega) == 0:
            raise ValueError(f'No trade data for {o} in df0_trade')
    except (KeyError, ValueError) as e:
        print(f'No trade data for {o}: {q} in df0_trade. Using {scenario} quote data instead. {e}')
        v_ix = df_quote.loc[(slice(None), o.expiry, o.optionContract.strike, o.right)].index
        v_ix = np.random.choice(v_ix, sub_sample_size) if len(v_ix) > sub_sample_size else v_ix
        v_iv_vega = np.array([(ps[quote_col_nm], o.vega(ps[quote_col_nm], ps['spot'], calc_date)) for ix, ps in df_quote.loc[(v_ix, o.expiry, o.optionContract.strike, o.right)].iterrows()])
    return v_iv_vega


def get_ts_sample(df: pd.DataFrame, o: Option):
    v_ix_ts = df.loc[(slice(None), o.expiry, o.optionContract.strike, o.right)].index
    ts_sample = []
    for h in range(9, 16):
        v_ix = v_ix_ts[v_ix_ts.get_level_values('ts').hour == h]
        if len(v_ix) > 0:
            ts_sample.append(max(v_ix))
    return ts_sample


def get_v_nlv_sample(df_trade, df_quote, o: Option, q: int, calc_date: date, quote_col_nm, trade_col_nm='fill_iv', scenario: str | Scenario = Scenario.worst) -> Tuple[np.ndarray, List[datetime]]:
    v_nlv = []
    ts_sample = []
    for h in range(9, 16):
        try:
            v_ix_ts = df_trade.loc[(slice(None), o.expiry, o.optionContract.strike, o.right)].index
            v_ix = v_ix_ts[v_ix_ts.get_level_values('ts').hour == h]
            if len(v_ix) <= 0:
                raise ValueError(f'No trade data for {o} in df0_trade')
            ps = df_trade.loc[(max(v_ix), o.expiry, o.optionContract.strike, o.right)]
            v_nlv.append(100 * q * o.npv(ps[trade_col_nm], ps['spot'], calc_date))
            ts_sample.append(max(v_ix))

        except (KeyError, ValueError) as e:
            print(f'No trade data for {o}: {q} in df0_trade. Using {scenario} quote data instead. {e}')
            v_ix_ts = df_quote.loc[(slice(None), o.expiry, o.optionContract.strike, o.right)].index
            v_ix = v_ix_ts[v_ix_ts.get_level_values('ts').hour == h]
            if len(v_ix) > 0:
                ps = df_quote.loc[(max(v_ix), o.expiry, o.optionContract.strike, o.right)]
                v_nlv.append(100 * q * o.npv(ps[quote_col_nm], ps['spot'], calc_date))
                ts_sample.append(max(v_ix))

    return v_nlv, ts_sample


def simulate_pf_pnl_new(pf: Portfolio, df0_quote, df0_trade, df1_quote, df1_trade, scenario=Scenario.worst) -> Dict[datetime, np.ndarray]:
    """
    Sample a snapshot in every hour on T0 and T+1.
    Overnight equity at 15:55 and 9:31.
    The pf delta at 9:31 determines the delta for the remainder of T+1 snaps.
    Assuming zero transaction or hedging costs.
    """
    calc_date0 = df0_quote.index.get_level_values('ts')[0].date()
    calc_date1 = df1_quote.index.get_level_values('ts')[0].date()
    n_sample = len(range(9, 16))
    dct_dlnv = defaultdict(lambda: np.zeros(n_sample))
    dct_keys = pd.Series(None, index=[datetime(calc_date1.year, calc_date1.month, calc_date1.day) + timedelta(hours=i) for i in range(9, 16)])
    p0 = df0_quote['spot'].iloc[-1]

    for o, q in pf.items():
        if isinstance(o, Equity):
            q = next(iter([q for sec, q in pf.items() if isinstance(sec, Equity)]), 0)
            for h in range(9, 16):
                ts = datetime(calc_date1.year, calc_date1.month, calc_date1.day) + timedelta(hours=h)
                v_ix = df1_trade.index.get_level_values('ts')
                p1 = df1_trade.loc[v_ix[v_ix > ts], 'spot'].iloc[0]
                print(f'Overnight equity Quantity: {q}, dS: {p1 - p0}')
                dct_dlnv[ts] += q * (p1 - p0)

            p1 = df1_quote['spot'].iloc[0]
            print(f'Overnight equity Quantity: {q}, dS: {p1 - p0}')
            dct_dlnv = {ts: v_dlnv + q * (p1 - p0) for ts, v_dlnv in dct_dlnv.items()}

        elif isinstance(o, Option):
            col_nm = 'ask_iv' if q > 0 else 'bid_iv' if scenario == Scenario.worst else 'mid_iv'
            v_nlv0, _ = get_v_nlv_sample(df0_trade, df0_quote, o, q, calc_date0, quote_col_nm=col_nm, trade_col_nm='fill_iv', scenario=scenario)
            # v_nlv0 = v_nlv0[~np.isnan(v_nlv0)]
            if len(v_nlv0) == 0:
                print(f'Excluded {o} from simulation. No data for {o}: {q}.')
                continue

            col_nm = 'bid_iv' if q > 0 else 'ask_iv' if scenario == Scenario.worst else 'mid_iv'
            v_nlv1, ts_sample1 = get_v_nlv_sample(df1_trade, df1_quote, o, q, calc_date1, quote_col_nm=col_nm, trade_col_nm='fill_iv', scenario=scenario)
            ts_sample1 = [datetime(ts.year, ts.month, ts.day) + timedelta(hours=ts.hour) for ts in ts_sample1]
            ps_nlv1 = dct_keys.copy()
            ps_nlv1[ts_sample1] = v_nlv1
            ps_nlv1 = ps_nlv1.ffill().bfill()
            # v_nlv1 = v_nlv1[~np.isnan(v_nlv1)]
            # if len(v_nlv1) == 0:
            #     print(f'No data for v_nlv1 {o}: {q}. Ignoring this. one. Assume 0.')
            #     nlv1 = 0
            for ts, nlv1 in ps_nlv1.items():
                dct_dlnv[ts] += np.random.choice(np.array([nlv1 - nlv0 for nlv0, nlv1 in itertools.product(v_nlv0, [nlv1])]), n_sample)

    # Add equity hedge from T+!:09:31 to ts_sample1
    # pf_delta_sod = pf_delta_since(pf, df1_trade, df1_quote, df1_quote.index.get_level_values('ts')[0], p0)

    # for ts, v_dnlv in dct_dlnv.items():
    #     pf_delta_ts = pf_delta_since(pf, df1_trade, df1_quote, ts, p0)
    #     p2 = df1_trade.loc[ts, 'spot'].iloc[0] if ts in df1_trade.index.get_level_values('ts') else df1_quote.loc[ts, 'spot'].iloc[0]
    #     ds = (p2 - p1)
    #     v_dnlv += -(pf_delta_sod + pf_delta_ts)/2 * ds

    return dct_dlnv


def pf_delta_since(pf, df_trade, df_quote, since_ts, p1):
    calc_date1 = since_ts.date()
    pf_delta_sod = 0
    for o, q in pf.items():
        if isinstance(o, Option):
            try:
                ix_ts = df_trade.index.get_level_values('ts')
                ix_ts = ix_ts[ix_ts > since_ts]
                first_iv = df_trade.loc[(ix_ts, o.expiry, o.optionContract.strike, o.right), 'fill_iv'].iloc[0]
            except Exception as e:
                ix_ts = df_quote.index.get_level_values('ts')
                ix_ts = ix_ts[ix_ts > since_ts]
                first_iv = df_quote.loc[(ix_ts, o.expiry, o.optionContract.strike, o.right), 'mid_iv']
                first_iv = first_iv.iloc[0] if len(first_iv) else 0.3
            delta = o.delta(first_iv, p1, calc_date1)
            if np.isnan(delta):
                print(f'No data for {o}: {q}. Ignoring this. one. Delta=0')
                delta = 0
            pf_delta_sod += 100 * q * delta
    return pf_delta_sod


def simulate_pf_pnl(pf: Portfolio, df0_quote, df0_trade, df1_quote, df1_trade, sample_size=500, scenario=Scenario.worst) -> np.ndarray:
    """
    Sample a snapshot in every hour on T0 and T+1.
    Overnight equity at 15:55 and 9:31.
    The pf delta at 9:31 determines the delta for the remainder of T+1 snaps.
    Assuming zero transaction or hedging costs.
    """
    calc_date0 = df0_quote.index.get_level_values('ts')[0].date()
    calc_date1 = df1_quote.index.get_level_values('ts')[0].date()
    sub_sample_size = int(np.ceil(sample_size**0.5))

    v_dlnv = np.zeros(sample_size)
    for sec, q in pf.items():
        if isinstance(sec, Equity):
            # The equity jump. Before and after, assuming perfect, costless hedging.
            p0 = df0_quote['spot'].iloc[-1]
            p1 = df1_quote['spot'].iloc[0]
            v_dlnv += q * (p1 - p0)
        else:
            o = sec

            col_nm = 'ask_iv' if q > 0 else 'bid_iv' if scenario == Scenario.worst else 'mid_iv'
            v_iv_vega0 = get_v_iv_vega(df0_trade, df0_quote, o, q, sub_sample_size, calc_date0, quote_col_nm=col_nm, trade_col_nm='fill_iv', scenario=scenario)
            v_iv_vega0 = np.array([(iv, vega) for iv, vega in v_iv_vega0 if not any((np.isnan(v) for v in (iv, vega)))])
            if len(v_iv_vega0) == 0:
                print(f'No data for v_iv_vega0 {o}: {q}. Assume intrinsic value. 0 IV => 0 vega.')
                v_iv_vega0 = np.array([(0, 0)])

            col_nm = 'bid_iv' if q > 0 else 'ask_iv' if scenario == Scenario.worst else 'mid_iv'
            v_iv_vega1 = get_v_iv_vega(df1_trade, df1_quote, o, q, sub_sample_size, calc_date1, quote_col_nm=col_nm, trade_col_nm='fill_iv', scenario=scenario)
            v_iv_vega1 = np.array([(iv, vega) for iv, vega in v_iv_vega1 if not any((np.isnan(v) for v in (iv, vega)))])
            if len(v_iv_vega1) == 0:
                print(f'No data for v_iv_vega1 {o}: {q}. Ignoring this. one. Assume intrinsic value. 0 IV => 0 vega.')
                v_iv_vega1 = np.array([(0, 0)])

            tmp = [100*q * (iv1 - iv0) * (vega0 + vega1) / 2 for (iv0, vega0), (iv1, vega1) in itertools.product(v_iv_vega0, v_iv_vega1) if not any((np.isnan(v) for v in (iv1, iv0, vega0, vega1)))]
            if len(tmp) == 0:
                print(f'No data for {o}: {q}. Ignoring this. one. Bad. Should assume cannot buy or sell, hence market value simply.')
                continue
            v_dlnv += np.random.choice(tmp, sample_size)

    return v_dlnv


ts_scoped = lambda df: [ts for ts in df.index.get_level_values('ts').unique() if ts.time() > time(13, 30)]


def simulate_pf_pnl_kalman(pf: Portfolio, df0_quote, df0_trade, df1_quote, df1_trade, sample_size=500, scenario=Scenario.worst) -> np.ndarray:
    """
    Don't use mid prices, but actual fills. If none available, worst quote.

    For each day and for each security, n prices are sampled.
    There'd be n**2 dnlv combinations.

    Plot actual trade fills during that period, instead of limit. My fill estimates have to match actual pass fills!
    Simulate better and worse fill depending on liquidity.
    Worst/Best/Mid
    """
    col_nm_fill_iv = 'fill_iv'
    calc_date0 = df0_quote.index.get_level_values('ts')[0].date()
    calc_date1 = df1_quote.index.get_level_values('ts')[0].date()
    sub_sample_size = int(np.ceil(sample_size**0.5))

    v_dlnv = np.zeros(sample_size)
    for sec, q in pf.items():
        if isinstance(sec, Equity):
            p0 = df0_quote['spot'].iloc[-1]
            p1 = df1_quote['spot'].iloc[0]
            v_dlnv += q * (p1 - p0)
        else:
            o = sec

            # Buying at t0 if q > 0
            quote_iv_col_nm = 'ask_iv' if q > 0 else 'bid_iv' if scenario == Scenario.worst else 'mid_iv'
            iv_col_nm = kalman_my_bid_col if q > 0 else kalman_my_ask_col
            _df0_trade = df0_trade[~df0_trade[iv_col_nm].isna()]
            if q > 0:
                _df0_trade = _df0_trade[_df0_trade[kalman_my_bid_col] >= _df0_trade[col_nm_fill_iv]]
            else:
                _df0_trade = _df0_trade[_df0_trade[kalman_my_ask_col] <= _df0_trade[col_nm_fill_iv]]
            v_iv_vega0 = get_v_iv_vega(_df0_trade, df0_quote, o, q, sub_sample_size, calc_date0, quote_col_nm=quote_iv_col_nm, trade_col_nm=iv_col_nm, scenario=scenario)
            v_iv_vega0 = np.array([(iv, vega) for iv, vega in v_iv_vega0 if not any((np.isnan(v) for v in (iv, vega)))])
            if len(v_iv_vega0) == 0:
                print(f'No data for v_iv_vega0 {o}: {q}. Assume intrinsic value. 0 IV => 0 vega.')
                v_iv_vega0 = np.array([(0, 0)])

            # Selling at t0 if q > 0
            quote_iv_col_nm = 'ask_iv' if q < 0 else 'bid_iv' if scenario == Scenario.worst else 'mid_iv'
            iv_col_nm = kalman_my_bid_col if q < 0 else kalman_my_ask_col
            _df1_trade = df1_trade[~df1_trade[iv_col_nm].isna()]
            if q < 0:
                _df1_trade = _df1_trade[_df1_trade[kalman_my_bid_col] >= _df1_trade[col_nm_fill_iv]]
            else:
                _df1_trade = _df1_trade[_df1_trade[kalman_my_ask_col] <= _df1_trade[col_nm_fill_iv]]
            v_iv_vega1 = get_v_iv_vega(df1_trade, df1_quote, o, q, sub_sample_size, calc_date1, quote_col_nm=quote_iv_col_nm, trade_col_nm=iv_col_nm, scenario=scenario)
            v_iv_vega1 = np.array([(iv, vega) for iv, vega in v_iv_vega1 if not any((np.isnan(v) for v in (iv, vega)))])
            if len(v_iv_vega1) == 0:
                print(f'No data for v_iv_vega1 {o}: {q}. Ignoring this. one. Assume intrinsic value. 0 IV => 0 vega.')
                v_iv_vega1 = np.array([(0, 0)])

            tmp = [100 * q * (iv1 - iv0) * (vega0 + vega1) / 2 for (iv0, vega0), (iv1, vega1) in itertools.product(v_iv_vega0, v_iv_vega1) if
                   not any((np.isnan(v) for v in (iv1, iv0, vega0, vega1)))]
            if len(tmp) == 0:
                print(f'No data for {o}: {q}. Ignoring this. one. Bad. Should assume cannot buy or sell, hence market value simply.')
                continue
            v_dlnv += np.random.choice(tmp, sample_size)

    return v_dlnv


@dataclass
class IVByTime:
    ts: datetime
    iv: float
    moneyness: float
    tenor: float
    right: str


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
                fwd_s = s * np.exp(net_yield * get_tenor(expiry, date.today()))
                iv = moneyness_iv(ss_df.droplevel(-1), right_moneyness, expiry, fwd_s)
                v_atexpiry_iv.append(iv)
            atexpiry_iv = np.interp((dt - at_the_date_expiries[0]).days, [0, (at_the_date_expiries[1] - at_the_date_expiries[0]).days], v_atexpiry_iv)
            tenor_ = get_tenor(dt, ts.date())
            results.append(IVByTime(ts, atexpiry_iv, right_moneyness, tenor_, right))
    return pd.DataFrame(results)


def stress_pf(pf: Portfolio, s0, ds_ret: float, calcDate0, calcDate1: date, df0, cfg: EarningsConfig, d_iv=None, scenario: str = Scenario.worst) -> float:
    """
    Then in addition, also change dIV.
    No equity hedging here
    """
    s1 = s0 * ds_ret

    nlv0 = 0
    nlv1 = 0
    for sec, q in pf.items():
        if isinstance(sec, Equity):
            nlv0 += q * s0
            nlv1 += q * s1
        else:
            o = sec
            ps_o = df0.loc[(o.expiry, o.optionContract.strike, o.right)]

            p0_worst = ps_o['bid_close'] if q < 0 else ps_o['ask_close']
            p0_best = ps_o['ask_close'] if q < 0 else ps_o['bid_close']

            iv0_worst_entry = o.iv(p0_worst, s0, calcDate0)
            iv0_best_entry = o.iv(p0_best, s0, calcDate0)

            if np.isnan(iv0_worst_entry) or np.isnan(iv0_best_entry):
                raise ValueError(f'IV0 is NaN for {o} worst/best {p0_worst}/{p0_best} {s0} {calcDate0}')

            if scenario == Scenario.worst:
                nlv0 += p0_worst * q * 100
            elif scenario == Scenario.best:
                nlv0 += p0_best * q * 100
            else:
                raise ValueError(f'Unknown scenario: {scenario}')

            if d_iv is not None:
                if scenario == Scenario.worst:
                    iv1 = iv0_best_entry + d_iv  # change of direction.
                elif scenario == Scenario.best:
                    iv1 = iv0_worst_entry + d_iv
                else:
                    raise ValueError(f'Unknown scenario: {scenario}')
            else:
                # reversed side as transaction goes opposite way. quantity is -q
                if scenario == Scenario.worst:
                    side = 'bid' if -q < 0 else 'ask'
                elif scenario == Scenario.best:
                    side = 'ask' if -q < 0 else 'bid'
                else:
                    raise ValueError(f'Unknown scenario: {scenario}')
                iv1 = estimate_iv1_side(df0, o, s0, s1, ps_o['tenor'], cfg, side, calcDate0)

            if np.isnan(iv1):
                p1 = o.npv(0, s1, calcDate1)
                logger.warning(f'stress_pf: IV1 is NaN for {o} {iv0_worst_entry}/{iv0_best_entry} {iv1} {ds_ret}. Assuming IV0, Resulting price is ')
            else:
                p1 = o.npv(iv1, s1, calcDate1)

            nlv1 += p1 * q * 100

    return nlv1 - nlv0

def npv(sec: Union[Option, Equity], s0, iv: float, calc_date: date) -> float:
    return sec.npv(iv, s0, calc_date) if isinstance(sec, Option) else s0


def nlv(sec: Union[Option, Equity], s0, iv: float, q, calc_date: date) -> float:
    if iv is None or np.isnan(iv):
        return None

    if isinstance(sec, Option):
        return sec.npv(iv, s0, calc_date) * q * 100
    else:
        return q * s0


def delta(sec: Union[Option, Equity], s0, iv: float, q, calc_date: date) -> float:
    if isinstance(sec, Option):
        return sec.delta(iv, s0, calc_date) * q * 100
    else:
        return q * s0


def iv_as_of(o: Option, df0, calc_date0: date, as_of: date, ds_ret, cfg: EarningsConfig, side):
    if isinstance(o, Equity):
        return 0

    ps_o = df0.loc[(o.expiry, o.optionContract.strike, o.right)]
    s0 = ps_o['spot']
    s1 = s0 * ds_ret

    p0 = ps_o['bid_close'] if side == 'bid' else ps_o['ask_close']

    if np.isnan(p0) or np.isnan(s0):
        return np.nan

    iv0 = o.iv(p0, s0, calc_date0)

    if np.isnan(iv0):
        logger.error(ValueError(f'IV0 is 0 or NaN for {o} {p0} {s0} {calc_date0}'))
        return np.nan

    if calc_date0 == as_of:
        return iv0

    return estimate_iv1_side(df0, o, s0, s1, ps_o['tenor'], cfg, side, calc_date0)


@lru_cache(maxsize=None)
def get_option(underlying: str, expiry: date, strike: Decimal, right: str, calc_date0: date) -> Option:
    return Option(OptionContract('', underlying, expiry, strike, right), calc_date0)


def a_bx_cx2(x, a, b, c):
    return a + b * x + c * x ** 2


def estimate_iv1_side(df0: pd.DataFrame, o: Option, s0, s1, tenor_: float, cfg: EarningsConfig, side, calc_date0):
    col_price = f'{side}_close'
    moneyness1 = float(o.strike) / s1

    x1 = pd.DataFrame([[get_moneyness_fwd_ln(Equity(cfg.sym), float(o.strike), s1, tenor_), tenor_]], columns=['moneyness_fwd_ln', 'tenor'])
    d_iv_ratio = cfg.earnings_iv_drop_regressor.predict(x1)[0][0]

    dft = df0.loc[(o.expiry, slice(None), o.right)]
    dft = dft[dft[col_price].notna()]
    # iv0 = df0.loc[(o.expiry, o.optionContract.strike, o.right), 'mid_iv']

    virtual_strike0 = moneyness1 * s0

    strikes = list(sorted(dft.index.get_level_values('strike').unique()))

    # interpolate the nearest IVs.
    ix_upper_strike = np.argmax(strikes > virtual_strike0) if (strikes > virtual_strike0).sum() > 0 else -1
    ix_lower_strike = ix_upper_strike - 1 if (strikes > virtual_strike0).sum() > 0 else -1
    lower_strike = strikes[ix_lower_strike]
    upper_strike = strikes[ix_upper_strike]

    price_lower = dft.loc[lower_strike, col_price]
    o_lower = get_option(o.underlying_symbol, o.expiry, lower_strike, o.right, calc_date0)
    iv_lower = o_lower.iv(price_lower, s0, calc_date0)

    price_upper = dft.loc[upper_strike, col_price]
    o_upper = get_option(o.underlying_symbol, o.expiry, upper_strike, o.right, calc_date0)
    iv_upper = o_upper.iv(price_upper, s0, calc_date0)

    virtual_iv0 = np.interp(virtual_strike0, [lower_strike, upper_strike], [iv_lower, iv_upper])

    iv1 = virtual_iv0 * (1 + d_iv_ratio)

    return iv1


def estimate_iv1(df0: pd.DataFrame, o: Option, s0, s1, tenor_: float, cfg: EarningsConfig, calc_date0) -> Tuple[float, float]:
    """
    For extreme moneyness contracts, best to assume I pay spread due to low liquidity.
    """

    iv_bid1 = estimate_iv1_side(df0, o, s0, s1, tenor_, cfg, 'bid', calc_date0)
    iv_ask1 = estimate_iv1_side(df0, o, s0, s1, tenor_, cfg, 'ask', calc_date0)

    return iv_bid1, iv_ask1


@lru_cache(maxsize=1000)
def d_iv_power_law(d_iv_tenor_1, expiry: date, calc_date: date, ts_time_power=0.5):
    tenor_ = get_tenor(expiry, calc_date)
    if tenor_ <= 0:
        return d_iv_tenor_1
    return d_iv_tenor_1 / tenor_ ** ts_time_power


def infinite_volatility(df) -> pd.Series:
    """
    Vola infinity could be something along max expiry sample 3-2 months before reporting date. Averaged + error bounds.
    Also sample the usual term structure power law. Average + err bounds.
    From avg. power law and infite vola, calc expected post reporting date IVs. Issue, questionable whether surface returns immediately to this state. Check previous earnings…
    Assume for each month how much level of expiry is gonna drop.
    """

    @dataclass
    class Point:
        ts: datetime
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


def power_law_term_structure(df, min_expiry, min_tenor_near=0.0, marker_size=3):
    """
    Question: Do the term structure of IVs follow a power law?
    Here plotting z:
    (far_atm_iv - near_atm_iv) * sqrt(t_far * t_near) / sqrt(t_far - t_near)
    Currently only for the 2 tenors, nearest expiry and ~1Y.

    But that's not answering a TS power law question...
    """
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
            tenor_near = get_tenor(exp_near, ts.date())
            tenor_far = get_tenor(exp_far, ts.date())

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

    fig = make_subplots(rows=4, cols=1, subplot_titles=['ATM IV', 'TS over time', 'Unscaled z: (far_atm_iv - near_atm_iv) * sqrt(t_far * t_near) / sqrt(t_far - t_near)', 'Scaled z: z / z(T3mT1y)'])

    # ATM IVs by expiry
    for exp_far, ss_df in res_df.groupby('exp_far'):
        fig.add_trace(go.Scatter(x=ss_df.index, y=ss_df['far_atm_iv'], mode='lines+markers', name=f'{exp_far}', marker=dict(size=marker_size)), row=1, col=1)

    # TS over time
    for ts, s_df in df.groupby(level='ts'):
        s0 = s_df.iloc[0]['spot']
        x = []
        y = []
        for expiry, ss_df in s_df.groupby('expiry'):
            iv = atm_iv(ss_df.loc[ts], expiry, s0)
            if iv <= 0 or np.isnan(iv):
                continue
            tenor_ = get_tenor(expiry, ts.date())
            x.append(tenor_)
            y.append(iv)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f'IV Term Structure @ {ts.date()}', marker=dict(size=marker_size)), row=2, col=1)

    # Unscaled
    for ix, s_df in res_df.groupby(['exp_near', 'exp_far']):
        fig.add_trace(go.Scatter(x=s_df.index, y=s_df['z'], mode='lines+markers', name=f'{ix[0]}-{ix[1]}', marker=dict(size=marker_size)), row=3, col=1)

    # Scaled
    # exp_T3m = res_df['exp_near'].iloc[(res_df['tenor_near'] - 0.25).abs().argmin()]
    exp_T3m = min_expiry
    exp_T1y = res_df['exp_far'].iloc[(res_df['tenor_far'] - 1).abs().argmin()]
    ps_scale = res_df.reset_index().set_index(['exp_near', 'exp_far']).loc[exp_T3m, exp_T1y][['z', 'ts']].set_index('ts')['z']
    for ts, s_df in res_df.groupby(level='ts'):
        res_df.loc[ts, 'z'] = s_df['z'] / ps_scale.loc[ts]

    for ix, s_df in res_df.groupby(['exp_near', 'exp_far']):
        fig.add_trace(go.Scatter(x=s_df.index, y=s_df['z'], mode='lines+markers', name=f'far: {ix[1]}-near: {ix[0]}', marker=dict(size=marker_size)), row=4, col=1)

    show(fig, 'power_law_term_structure.html')

    return res_df


def estimate_diffusion_iv(df, release_date):
    ts = df.index.get_level_values('ts').unique()
    expiries_df = list(sorted(set(df.index.get_level_values('expiry').values)))

    # This calc requires about 14 days worth of data if weeklies are traded.
    # 7 days to avoid typical ramp of expiring options
    diffusion_iv = None
    if [i for i in expiries_df if i < release_date]:
        previous_non_jump_expiry = sorted([i for i in expiries_df if i < release_date])[-1]
        v_ts_pre_releaseTm7 = [i for i in ts if i.date() == previous_non_jump_expiry - timedelta(days=7)]
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


def add_plot_term_structure(df, calcDate, fig, row, cfg: EarningsConfig, dIV_levels=(0, -0.01), ts_time_power=0.5, name_prefix=''):
    for dIV_level in dIV_levels:
        x = []
        y = []
        for expiry, ss_df in df.groupby('expiry'):
            tenor_ = get_tenor(expiry, calcDate)
            ts = ss_df.index.get_level_values('ts').unique()[0]
            atm_iv_val = atm_iv(ss_df.loc[ts], expiry, ss_df.iloc[0]['spot'])
            if atm_iv_val <= 0 or np.isnan(atm_iv_val):
                continue

            if dIV_level is None:  # predicted
                df_mny_t = pd.DataFrame([[get_moneyness_fwd_ln(Equity(cfg.sym), ss_df.iloc[0]['spot'], ss_df.iloc[0]['spot'], tenor_), tenor_]], columns=['moneyness_fwd_ln', 'tenor'])
                d_iv_ratio = cfg.earnings_iv_drop_regressor.predict(df_mny_t)[0][0]
                dIV_tenor = atm_iv_val * d_iv_ratio
                # dIV_tenor = ss_df.loc[(ts, expiry, slice(None), slice(None)), 'd_iv_ratio'].unique().mean() * atm_iv_val
            else:  # fixed simulation
                dIV_tenor = d_iv_power_law(dIV_level, expiry, calcDate, ts_time_power=ts_time_power)
            iv = atm_iv_val + dIV_tenor
            if iv > 5:
                pass

            x.append((expiry - df.index.get_level_values('ts').min().date()).days)
            y.append(iv)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f'{name_prefix}IV Term Structure @ {calcDate} dIV={dIV_level}', marker=dict(size=3)), row=row, col=1)


def add_plot_iv(df, ts, pf, fig, row, col, marker_size_iv=3, marker_size_holdings=8):
    calcDate = ts.date()
    holding_expiries = list(set([sec.expiry for sec in pf.portfolio.keys() if isinstance(sec, Option)]))

    for expiry in holding_expiries:
        s_df = df.loc[(ts, expiry, slice(None), slice(None))]

        # Plot IVs
        for right, ss_df in s_df.groupby(level='right'):
            sample_df = ss_df[ss_df['mid_iv'].fillna(0) != 0]
            fig.add_trace(go.Scatter(x=sample_df['moneyness'], y=sample_df['mid_iv'], mode='lines+markers', name=f'IV {expiry} {right}', marker=dict(size=marker_size_iv)), row=row,
                          col=col)

    right_symbol_map = {'call': 'x-dot', 'put': 'circle-dot'}
    for right, s_df in df.loc[ts].groupby(level='right'):
        holdings = {sec: q for sec, q in pf.portfolio.items() if isinstance(sec, Option) and sec.right == right}
        s = s_df.iloc[0]['spot']
        x_moneyness = [float(sec.optionContract.strike) / s for sec in holdings.keys()]
        y_iv = [sec.iv(s_df.loc[(sec.expiry, sec.optionContract.strike, sec.right), 'mid_price'], s, calcDate) for sec in holdings.keys()]
        # if 0 in y_iv:
        #     raise ValueError(f'Zero IV for {right} {ts} {x_moneyness} {y_iv}')

        fig.add_trace(go.Scatter(
            x=x_moneyness, y=y_iv, mode='markers', name=f'IV Holdings {right}',
            text=[f'{str(sec)}: {q}' for sec, q in holdings.items()],
            marker=dict(size=marker_size_holdings, symbol=right_symbol_map[right])), row=row, col=col)


def get_exp_d_iv_skew_regressed_curvature(df, ds):
    # Average all skews up to a K close to -ds
    df[f'exp_d_iv_skew_regressed_curvature_{ds:.2f}'] = None
    for expiry, s_df in df.groupby(level='expiry'):
        for right, ss_df in s_df.groupby(level='right'):
            strikes = ss_df.index.get_level_values('strike')
            expected_d_iv = []
            for strike in strikes:
                final_strike = float(strike) - ds
                strikes_to_avg = strikes[(float(strike) <= strikes.astype(float)) & (strikes.astype(float) <= final_strike)] if final_strike > float(strike) \
                    else strikes[(final_strike <= strikes.astype(float)) & (strikes.astype(float) <= float(strike))]
                expected_d_iv.append(ss_df.loc[(expiry, strikes_to_avg, right), 'mid_iv_curvature_regressed'].mean() * -ds)

            df.loc[(expiry, slice(None), right), 'exp_d_iv_skew_regressed_curvature'] = expected_d_iv
    return df[f'exp_d_iv_skew_regressed_curvature_{ds:.2f}']


def add_plot_skew(df, ts, pf, fig, row, col, marker_size=4):
    holding_expiries = list(set([sec.expiry for sec in pf.portfolio.keys() if isinstance(sec, Option)]))
    s = df.loc[ts].iloc[0]['spot']
    s_df = df.loc[ts]
    for expiry in holding_expiries:
        try:
            ss_df = s_df.loc[expiry]
        except KeyError as e:
            print(f'KeyError: {e} {expiry} {s_df.index.get_level_values("expiry")}')
            continue
        for right, sss_df in ss_df.groupby(level='right'):
            skew_col_nm = 'mid_iv_curvature_regressed'
            v_skew = sss_df.loc[(slice(None), right), skew_col_nm]
            fig.add_trace(go.Scatter(x=sss_df.index.get_level_values('strike').astype(float)/s, y=v_skew, mode='markers', name=f'Skew {expiry} {right}', marker=dict(size=marker_size)), row=row, col=col)


def run():
    """
        Contrast 2 strategies:
        - 1: Sell near expiry options, buy far expiry options as hedge. Earn near IV crush, earn smaller/larger than expected ds. Pay far expiry IV crush.
        - TBD 2: Earn far expiry IV crush.

        To contrast, need an estimation of how much the IV surface is gonna change.
        """
    tickers = 'BA'
    for take in [-2]:
        for sym in tickers.split(','):
            release_date = EarningsPreSessionDates(sym)[take]
            print(release_date)

            holdings = {Option(OptionContract.from_ib_symbol(s), release_date): q for s, q in {
                 # "DAL   240726C00048500": -1,
            }.items()}
            # holdings[Equity(sym)] = 39

            path_model = os.path.join(Paths.path_models, model_nm_earnings_iv_drop_regressor)
            earnings_iv_drop_regressor = EarningsIVDropPolyRegressorV3().load_model(path_model)

            cfg = EarningsConfig(sym, release_date, plot=True, plot_last=True, earnings_iv_drop_regressor=earnings_iv_drop_regressor,
                                 moneyness_limits=(0.8, 1.2), abs_delta_limits=(0.1, 0.9), seq_ret_threshold=0.002,
                                 min_tenor=0.0, max_tenor=1, add_equity_holdings=True, portfolio=holdings, run_solver=True,
                                 n_contracts=20
                                 )
            get_earnings_release_pf(cfg)


if __name__ == '__main__':
    from connector.api_minlp.common import get_stress_test_ds, StressTestDsResult, exclude_outlier_quotes

    run()
