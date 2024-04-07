import itertools
import os

import numpy as np
import pandas as pd
import scipy.stats
import plotly.graph_objects as go

from functools import lru_cache
from datetime import datetime, date, timedelta
from typing import List, Dict, Callable, Tuple, Union
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from itertools import combinations
from options.volatility.estimators.earnings_iv_drop_regressor import EarningsIVDropRegressor
from options.volatility.pf_opt_minlp_pyomo import derive_portfolio_milnp
from shared.modules.logger import logger

from options.helper import val_from_df, moneyness_iv, atm_iv, year_quarter, tenor, enrich_atm_iv
from options.typess.iv_surface import col_nm_mean_regressed_skew_ds
from options.typess.portfolio import Portfolio
from shared.paths import Paths
from shared.plotting import show
from shared.constants import EarningsPreSessionDates
from options.typess.enums import Resolution
from options.typess.equity import Equity
from options.typess.option_frame import OptionFrame
from options.typess.option_contract import OptionContract
from options.typess.option import Option
from shared.utils.decorators import time_it


@dataclass
class EarningsConfig:
    sym: str
    take: int = -1
    early_stopping: bool = False
    plot: bool = True
    plot_last: bool = True
    ranking_iterations: int = 20
    max_scoped_options: int = 400
    resolution: Resolution = Resolution.minute
    seq_ret_threshold: float = 0.005
    min_tenor: float = 0.5
    moneyness_limits: Tuple[float, float] = (0.6, 1.4)
    abs_delta_limits: Tuple[float, float] = (0.05, 0.95)
    use_skew: bool = False
    add_equity_holdings: bool = True
    v_dIVT1: np.array = field(default_factory=lambda: np.linspace(0.0, -0.03, 4))
    earnings_iv_drop_regressor: EarningsIVDropRegressor = None
    ts_time_power: float = 0.5
    v_ds_ret: np.array = field(default_factory=lambda: np.linspace(0.8, 1.2, 21))
    short_q_scale: int = 1


def next_release_date(sym: str, dt: date) -> date:
    release_dates = EarningsPreSessionDates(sym)
    return next((d for d in release_dates if d >= dt), None)


def get_earnings_release_pf(cfg: EarningsConfig) -> dict:
    """
        Refinement:
        - The ultimate short is exercised 1 day at EOD before release date. Also here can improve, research when the time value is high.
        Todo: Check how much the extrinsic value decreases towards release date. Want to short/long more than 1 D in advance?? IV goes up. Leaving money on the table?

        - Generalize. also investigate portfolios earning the term structure IV crush.

        # Questions:
        # - What's the optimal T-x days to sell near expiry options? Understands how far values change. How intrinsic value changes as release date comes up.
        # - Optimal near/long hedge ratios? Risk / Pnl Trade off... relate risk space and cost to hedge?
        """
    if cfg.earnings_iv_drop_regressor is None:
        raise ValueError('Earnings IV Drop Regressor not set')

    sym = cfg.sym
    equity = Equity(sym)
    release_date = EarningsPreSessionDates(sym)[cfg.take]

    option_frame = OptionFrame.load_frame(equity, cfg.resolution, cfg.seq_ret_threshold, year_quarter(release_date))
    df = option_frame.df_options.sort_index()

    # Enrich ATM IV
    enrich_atm_iv(df)
    d_iv_pct = cfg.earnings_iv_drop_regressor.predict(df)
    df['d_iv_ts'] = df['atm_iv'] * d_iv_pct

    ts = df.index.get_level_values('ts').unique()

    # adjustment. hedges can choose on which date. sell high IV is before release to center around underlying price.
    # v_ts_pre_release = [i for i in ts if i.date() <= release_date]  # preparing on day of expiry
    v_ts_pre_release = [i for i in ts if i.date() <= (release_date - timedelta(days=1))]  # preparing options the night before

    # avoid before 10 am because rolling skew is still very noise
    v_ts_pre_release = [i for i in v_ts_pre_release if i.hour >= 10]

    v_ts_post_release = [i for i in ts if i.date() >= (release_date + timedelta(days=1))]
    # avoid after 4 pm because rolling skew is still very noise
    v_ts_post_release = [i for i in v_ts_post_release if i.hour >= 10]
    first_2_days = sorted(list(set([i.date() for i in v_ts_post_release])))[:2]
    v_ts_post_release = [i for i in v_ts_post_release if i.date() in first_2_days]
    ts_pre_release = v_ts_pre_release[0]
    print(f'Release date: {release_date}. ts_pre_release: {ts_pre_release}')

    if v_ts_post_release:
        ts_post_release = v_ts_post_release[-1]
        print(f'TS Post Release - Min: {np.min(v_ts_post_release)} Max: {np.max(v_ts_post_release)}')

    calcDate0 = ts_pre_release.date()
    calcDate1 = release_date + timedelta(days=1)

    # split frame into pre and post release
    df0 = df.loc[[ts_pre_release]]
    if v_ts_post_release:
        df1 = df.loc[v_ts_post_release]

    # Spot 1 and Spot 2
    s0 = df0.loc[ts_pre_release].iloc[0]['spot']
    s1 = df1.loc[ts_post_release].iloc[0]['spot'] if v_ts_post_release else np.nan
    print(f'''Spot0: {s0}; Spot1: {s1}; Stock jumped by dS: {round(s1 - s0, 1)} | {round(100 * (s1 / s0 - 1), 1)} %''')
    v_ds_pct = [100 * (dS - 1) for dS in cfg.v_ds_ret]

    option_universe, cache_iv0, cache_iv1 = create_option_universe_iv_caches(sym, ts_pre_release, df0, s0, cfg.v_ds_ret, cfg.use_skew)
    expiries = list(sorted(set(df0.index.get_level_values('expiry').values)))
    exp_jump1 = sorted([i for i in expiries if i >= release_date])[0]
    exp_jumps = [exp_jump1]  # get rid of specifying jump IVs. Anything that's elevated returns to a certain level.

    inf_vol = infinite_volatility(df)
    print(f'Max Expiry ATM Volatility: {inf_vol.mean()}')
    # if cfg.plot:
    #     power_law_term_structure(df, exp_jump1)

    diffusion_iv = estimate_diffusion_iv(df, release_date)
    print(f'Nearest Expiry: {exp_jump1}, {sym.upper()}, ATM IV: {atm_iv(df0.loc[ts_pre_release], exp_jump1, s0)}, Diffusion IV: {diffusion_iv}')

    # Spot price series1 internal value, eventually weights also...
    if cfg.plot:
        plot_spot(df, df0, exp_jump1, ts_post_release, release_date, ts_pre_release, v_ts_post_release, s0, cfg.sym)

    # estimated_diffusion_iv   No feasible way estimating this diffusion way for later expiries. For csco, later expiries ATM IV fell significantly even below long term IV levels.
    # So now, just check plot this time atm iv over term and last release's one, then pick minimum tenor to use for hedging...

    pf = initialize_portfolio(df0, ts_pre_release, sym, exp_jump1, s0, cfg, option_universe)

    # This result seems too low, compared to IB emails...
    implied_jump = implied_ds(np.sqrt(atm_iv(df0.loc[ts_pre_release], exp_jump1, s0) ** 2 - diffusion_iv ** 2))
    print(f'implied jump return: {100 * implied_jump :.2f} %; implied jump dS={implied_jump * s0 :.2f}')

    scoped_expiries = get_scoped_expiries(expiries)
    get_p0 = lambda o: val_from_df(df0.loc[ts_pre_release], o.expiry, o.optionContract.strike, o.right, 'mid_price')
    scoped_options = [o for o in option_universe.values() if
                      option_in_pf_scope(o, cfg.min_tenor, cfg.moneyness_limits, calcDate0, s0, get_p0(o), cfg.abs_delta_limits, scoped_expiries=scoped_expiries, pf=pf)]
    print(f'Scoped options: #{len(scoped_options)} out of # {len(option_universe)}')
    # scoped_options = descope_options(scoped_options, s0, cfg.max_scoped_options)

    nlv0, v_delta0, m_dnlv01 = get_nlv_delta_matrix(s0, calcDate0, calcDate1, cfg, cache_iv0, cache_iv1, scoped_options)

    # Derive portfolio with iterative combinations OLD
    # scoped_option_names = [str(o) for o in scoped_options]
    # nlv01 = pd.DataFrame([np.array([nlv(o, s0 * ds_ret, cache_iv1[o][ds_ret], 1, calcDate1) for o in scoped_options]) - nlv0 for ds_ret in cfg.v_ds_ret], index=cfg.v_ds_ret, columns=scoped_option_names)
    # pf, iteration = derive_portolio_with_iterative_combinations(pf, equity, s0, df0, df1, v_ts_post_release, scoped_options, scoped_option_names, v_ds_pct, release_date, ts_pre_release, calcDate0, calcDate1, exp_jumps, diffusion_iv, cfg, f_weight_dS, cache_iv1, nlv01, v_delta0)

    """Rewarding risk reduction around implied jump dS move more than large dS moves"""
    # f_weight_ds = get_f_weight_ds_ret(implied_jump * 2, plot=cfg.plot)
    f_weight_ds = get_f_weight_ds_ret(0.1, plot=cfg.plot)

    iteration = cfg.ranking_iterations
    pf, _, inst = derive_portfolio_milnp(scoped_options, m_dnlv01, v_delta0, cfg, f_weight_ds=f_weight_ds, pf=pf)

    deltaTotalPf0, deltaTotalPf0Regressed = handle_equity_additions(pf, calcDate0, s0, df0, ts_pre_release, calcDate1, cfg)

    pf.print_entry_holdings(df0.loc[ts_pre_release], calcDate0)

    # Stressing the portfolio for a few more dIV level changes for plots
    if cfg.plot:
        plot_stressed_pnl(pf, cfg.v_dIVT1, v_ds_pct, deltaTotalPf0, iteration, release_date, ts_pre_release, v_ts_post_release, df1, s0, calcDate0, calcDate1, exp_jumps,
                          diffusion_iv, cfg.use_skew, df0, cfg.ts_time_power, cfg.v_ds_ret)
        # plot another 4 combos. where strike of calls and puts and varied +1/-1.
        # The solver should return alterative scenarios, possible with constraints just removing what's been selected... Problem, might come up with incompatible portfolio combos.
        # Could further constrain to

    if cfg.plot:
        plot_iv_over_time(pf, df, v_ts_post_release, ts_pre_release, ts_post_release, release_date, sym)

    print(f'Done earnings release {sym}')
    if cfg.plot_last and not cfg.plot:
        deltaTotalPf0 = pf.deltaTotal(calcDate0, s0, df0.loc[ts_pre_release])
        plot_stressed_pnl(pf, cfg.v_dIVT1, v_ds_pct, deltaTotalPf0, iteration, release_date, ts_pre_release, v_ts_post_release, df1, s0, calcDate0, calcDate1, exp_jumps,
                          diffusion_iv, cfg.use_skew, df0, cfg.ts_time_power, cfg.v_ds_ret)
    return pf.holdings


def get_nlv_delta_matrix(s0, calcDate0, calcDate1, cfg, cache_iv0, cache_iv1, scoped_options):
    nlv0 = np.array([nlv(o, s0, cache_iv0[o][1], 1, calcDate0) for o in scoped_options])
    v_delta0 = np.array([delta(o, s0, cache_iv0[o][1], 1, calcDate0) for o in scoped_options])
    m_dnlv01 = np.array([[nlv(o, s0 * ds_ret, cache_iv1[o][ds_ret], 1, calcDate1) for o in scoped_options] - nlv0 for ds_ret in cfg.v_ds_ret])
    return nlv0, v_delta0, m_dnlv01


def get_scoped_expiries(expiries: List[date]) -> List[date]:
    # Drop some weeklies to reduce scope. Pick 2 closest expiries. Then last of each month
    scoped_expiries = expiries[:2]
    for i, exp in itertools.groupby(expiries, lambda x: x.month):
        months_last_exp = list(exp)[-1]
        if months_last_exp.month not in [dt.month for dt in scoped_expiries]:
            scoped_expiries.append(months_last_exp)
    print(f'Scoped expiries: #{len(scoped_expiries)} out of #{len(expiries)}, {scoped_expiries}')
    return scoped_expiries


def handle_equity_additions(pf, calcDate0, s0, df0, ts_pre_release, calcDate1, cfg):
    s_ret_dNLV = {}
    for ds_ret in cfg.v_ds_ret:
        dNLV = stress_pf(pf, s0, ds_ret, calcDate0, calcDate1, cfg.use_skew, df0=df0.loc[ts_pre_release])
        s_ret_dNLV[ds_ret] = dNLV

    deltaTotalPf0 = pf.deltaTotal(calcDate0, s0, df0.loc[ts_pre_release])
    deltaTotalPf0Regressed = derive_delta_from_regressing_min_npvs(s_ret_dNLV, s0)
    print(f'Portfolio w/o  Equity DeltaTotalPf0Total: {deltaTotalPf0}, deltaTotalPf0Regressed: {deltaTotalPf0Regressed}')
    if cfg.add_equity_holdings:
        pf.add_holding(Equity(cfg.sym), -int(deltaTotalPf0Regressed))

    if cfg.add_equity_holdings:
        s_ret_dNLV = {}
        for ds_ret in cfg.v_ds_ret:
            dNLV = stress_pf(pf, s0, ds_ret, calcDate0, calcDate1, cfg.use_skew, df0=df0.loc[ts_pre_release])
            s_ret_dNLV[ds_ret] = dNLV
        deltaTotalPf0 = pf.deltaTotal(calcDate0, s0, df0.loc[ts_pre_release])
        deltaTotalPf0Regressed = derive_delta_from_regressing_min_npvs(s_ret_dNLV, s0)
        print(f'Portfolio with Equity DeltaTotalPf0Total: {deltaTotalPf0}, deltaTotalPf0Regressed: {deltaTotalPf0Regressed}')
    return deltaTotalPf0, deltaTotalPf0Regressed


def initialize_portfolio(df0, ts_pre_release, sym, exp_jump1, s0, cfg, option_universe):
    strikes = pd.Series(np.unique(df0.loc[(ts_pre_release, exp_jump1, slice(None), slice(None))].index.get_level_values('strike').astype(float)))
    strikes_atm = strikes.iloc[(strikes - s0).abs().sort_values().head(2).index].values

    print(f'ATM strike: {strikes_atm}')
    pf = Portfolio()
    for sec, q in {
        # option_universe[OptionContract('', sym, exp_jump1, Decimal(strikes_atm[0]), 'call').ib_symbol()]: -1 * cfg.short_q_scale,
        # option_universe[OptionContract('', sym, exp_jump1, Decimal(strikes_atm[0]), 'put').ib_symbol()]: -1 * cfg.short_q_scale,
        # option_universe[OptionContract('', sym, exp_jump1, Decimal(strikes_atm[1]), 'call').ib_symbol()]: -1 * cfg.short_q_scale,
        # option_universe[OptionContract('', sym, exp_jump1, Decimal(strikes_atm[1]), 'put').ib_symbol()]: -1 * cfg.short_q_scale,
        #
        **{option_universe[s]: q for s, q in {
            # 'DELL  231208P00079000': 3, 'DELL  231201P00085000': -3, 'DELL  231229P00076000': 9
            # 'DELL  231208C00071000': 1, 'DELL  231201P00077000': -2, 'DELL  231201P00073000': -3, 'DELL  231201P00085000': -3, 'DELL  231229P00076000': 10, 'DELL  231201C00075000': -1
            # 'DELL  231201C00065000': 6, 'DELL  231201C00066000': -7, 'DELL  231201C00074000': -2, 'DELL  231229C00070000': 9
            # 'DELL  231201P00073000':-2, 'DELL  231201P00085000':-3, 'DELL  231222C00079000':1, 'DELL  231201C00078000':-1, 'DELL  231229P00076000':9, 'DELL  231201P00076000':-2, 'DELL  231229C00070000':1, 'DELL  231201C00074000':-1,
            # 'DELL  231215P00078000': 2, 'DELL  231201C00075000': -1, 'DELL  231208P00075000': 1, 'DELL  231201C00078000': -1, 'DELL  231201C00079000': -1, 'DELL  231215C00067500': 1, 'DELL  231229C00071000': 1, 'DELL  231215C00076000': 1, 'DELL  231201P00073000': -2, 'DELL  231201P00075000': -1, 'DELL  231222C00075000': 1, 'DELL  240105P00074000': 1
            # 'DELL  231208C00071000':1, 'DELL  231201P00077000':-2, 'DELL  231201P00073000':-3, 'DELL  231201P00085000':-3, 'DELL  231229P00076000':10, 'DELL  231201C00075000':-1
            }.items()}
    }.items():
        pf.add_holding(sec, q)
    # 'DELL': 378,
    # pf.add_holding(Equity(sym), -46)
    return pf


def create_option_universe_iv_caches(sym, ts_pre_release, df0, s0, v_ds_ret, use_skew, holdings: Dict[str, int] = None):
    release_date = next_release_date(sym, ts_pre_release.date())
    calcDate0 = ts_pre_release.date()
    calcDate1 = release_date + timedelta(days=1)
    option_universe = {}
    for ix in df0.index:
        ts, expiry, strike, right = ix
        if expiry < release_date:
            continue
        o = Option(OptionContract('', sym, expiry, strike, right), calcDate0, s0, 0)
        option_universe[str(o)] = o
    print(f'Added {len(option_universe)} options to universe')
    f_iv_fallback = lambda df, sec: df0.loc[(ts_pre_release, sec.expiry, sec.strike, sec.right), 'mid_iv'] if isinstance(sec, Option) else 0

    cache_iv0 = {sec: {1: iv_as_of(sec, df0.loc[ts_pre_release], calcDate0, calcDate0, 1, use_skew, f_iv_fallback(df0, sec))} for sec in
                 list(option_universe.values()) + [Equity(sym)]}
    cache_iv1 = {sec: {ds_ret: iv_as_of(sec, df0.loc[ts_pre_release], calcDate0, calcDate1, ds_ret, use_skew) for ds_ret in v_ds_ret} for sec in
                 list(option_universe.values()) + [Equity(sym)]}

    # kick out options for which iv values are missing
    len(option_universe)
    rm_option = set([o for o in option_universe.values() if pd.isna(pd.Series(cache_iv1[o].values())).sum() > 0])
    rm_option.union(set([o for o in option_universe.values() if pd.isna(pd.Series(cache_iv0[o].values())).sum() > 0]))
    if holdings:
        rm_option = [o for o in rm_option if str(o) not in holdings]
    len(rm_option)

    print(f'Kicking out {len(rm_option)} options from option universe due to missing IV values: {rm_option}')
    for o in rm_option:
        if o in cache_iv0:
            cache_iv0.pop(o)
        if o in cache_iv1:
            cache_iv1.pop(o)
        if str(o) in option_universe:
            del option_universe[str(o)]

    return option_universe, cache_iv0, cache_iv1


def derive_portolio_with_iterative_combinations(pf, equity, s0, df0, df1, v_ts_post_release, scoped_options, scoped_option_names, v_ds_pct, release_date, ts_pre_release, calcDate0, calcDate1, exp_jumps, diffusion_iv, cfg, f_weight_dS, cache_iv1, nlv01, delta0):
    for iteration in range(cfg.ranking_iterations):
        print(f'Iteration: {iteration}')

        if iteration > 0:
            pf.remove_security(equity)
            nlv_pf0_no_additions = sum([nlv(sec, s0, iv_as_of(sec, df0.loc[ts_pre_release], calcDate0, calcDate0, 1, cfg.use_skew, exp_jumps, diffusion_iv), q, calcDate0) for sec, q in pf.holdings.items()])
            pf_delta_total = pf.deltaTotal(calcDate0, s0, df0.loc[ts_pre_release])
            ranked_hedges_dct = rank_hedges(pf, pf_delta_total, nlv01, delta0, s0, nlv_pf0_no_additions, calcDate1, cache_iv1, cfg.v_ds_ret, f_weight_dS)

            best_hedge_str, best_q = best_hedge_combo = max(ranked_hedges_dct, key=ranked_hedges_dct.get)
            best_hedge = scoped_options[scoped_option_names.index(best_hedge_str)]
            print(f'Best hedge: {best_hedge} Q: {best_q}: {ranked_hedges_dct[best_hedge_combo]}')
            # if ranked_hedges_dct[best_hedge] <= 0:
            #     break
            pf.add_holding(best_hedge, best_q)

        dct_dS_dNLV = {}
        for ds_ret in cfg.v_ds_ret:
            dNLV = stress_pf(pf, s0, ds_ret, calcDate0, calcDate1, cfg.use_skew, df0=df0.loc[ts_pre_release])
            dct_dS_dNLV[ds_ret] = dNLV

        deltaTotalPf0 = pf.deltaTotal(calcDate0, s0, df0.loc[ts_pre_release])
        deltaTotalPf0Regressed = derive_delta_from_regressing_min_npvs(dct_dS_dNLV, s0)
        print(f'Portfolio DeltaTotalPf0Total: {deltaTotalPf0}, deltaTotalPf0Regressed: {deltaTotalPf0Regressed}')
        if cfg.add_equity_holdings:
            pf.add_holding(Equity(cfg.sym), -int(deltaTotalPf0Regressed))
            dct_dS_dNLV = {}
            for ds_ret in cfg.v_ds_ret:
                dNLV = stress_pf(pf, s0, ds_ret, calcDate0, calcDate1, cfg.use_skew, df0=df0.loc[ts_pre_release])
                dct_dS_dNLV[ds_ret] = dNLV

        pf.print_entry_holdings(df0.loc[ts_pre_release], calcDate0)

        # Stressing the portfolio for a few more dIV level changes for plots
        if cfg.plot:
            plot_stressed_pnl(pf, cfg.v_dIVT1, v_ds_pct, deltaTotalPf0, iteration, release_date, ts_pre_release, v_ts_post_release, df1, s0, calcDate0, calcDate1, exp_jumps, diffusion_iv, cfg.use_skew, df0, cfg.ts_time_power, cfg.v_ds_ret)

        early_stopping_criteria_met = is_early_stopping_criteria_met(dct_dS_dNLV)
        if cfg.early_stopping and early_stopping_criteria_met:
            if cfg.plot_last:
                plot_stressed_pnl(pf, cfg.v_dIVT1, v_ds_pct, deltaTotalPf0, iteration, release_date, ts_pre_release, v_ts_post_release, df1, s0, calcDate0, calcDate1, exp_jumps, diffusion_iv, cfg.use_skew, df0, cfg.ts_time_power, cfg.v_ds_ret)
            return pf.holdings, iteration
    return pf, iteration


def title_stressed_pnl_plot(pf, release_date, iteration, deltaTotalPf0):
    # Title subscript of holdings
    sym = pf.underlying
    holdings_txt = ''
    for k, (sec, q) in enumerate(pf.holdings.items()):
        holdings_txt += f'{sec} {q}, '
        if k > 0 and k % 6 == 0:
            holdings_txt += '<br>'
    holdings_txt += f' DeltaTotalPf0Total: {deltaTotalPf0:.2f}'
    return f'{sym} dS dIV dPL Iteration {iteration} release_date: {release_date}<br><sup>{holdings_txt}</sup>'


def plot_stressed_pnl(pf: Portfolio, v_dIVT1, v_ds_pct, deltaTotalPf0, iteration,  release_date, ts_pre_release, v_ts_post_release, df1, s0, calcDate0, calcDate1, exp_jumps, diffusion_iv, use_skew, df0, ts_time_power, v_ds_ret, marker_size=4):
    sym = pf.underlying
    figdSdIV = make_subplots(rows=3, cols=2, specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}]], subplot_titles=['dS dIV dPL', 'IV ATM Horizontal', 'IV Vertical'])

    for dIVT1 in list(v_dIVT1) + [None]:
        v_dNLV = []
        for dS in v_ds_ret:
            dNLV = stress_pf(pf, s0, dS, calcDate0, calcDate1, use_skew, df0=df0.loc[ts_pre_release], d_iv_tenor=dIVT1)
            v_dNLV.append(dNLV)
        figdSdIV.add_trace(go.Scatter(x=v_ds_pct, y=v_dNLV, mode='markers', name=f'dIV: {str(dIVT1)}', marker=dict(size=marker_size)), row=1, col=1)

        # if add_equity_holdings:
        #     pf.add_holding(Equity(sym), -int(deltaTotalPf0Total))
        #
        #     v_dNLV = []
        #     for dS in v_dS:
        #         dNLV = stress_pf(pf, s0, dS, calcDate0, calcDate1, exp_jumps, diffusion_iv, df0=df0, ts_time_power=ts_time_power)
        #         v_dNLV.append(dNLV)
        #     figdSdIV.add_trace(go.Scatter(x=v_dSpct, y=v_dNLV, mode='markers', name=f'dIV + Equity: {str(dIVT1)}', marker=dict(size=marker_size)), row=1, col=1)
        title = title_stressed_pnl_plot(pf, release_date, iteration, deltaTotalPf0)
        figdSdIV.update_layout(title=title, xaxis_title=f'dS %', yaxis_title='dPL')

        # Plotting skews
        add_plot_iv(df0, ts_pre_release, pf, figdSdIV, 2, 2)
        add_plot_skew(df0, ts_pre_release, pf, figdSdIV, 3, 2)
        add_plot_term_structure(df0, calcDate0, figdSdIV, 2, dIV_levels=v_dIVT1, ts_time_power=ts_time_power)

    if v_ts_post_release:
        for dt, group in itertools.groupby(v_ts_post_release, lambda x: x.date()):
            v_ts = list(group)
            v_dSpct_actual = [100 * (df1.loc[ts]['spot'].iloc[0] / s0 - 1) for ts in v_ts]
            v_dPL_pf_actual = [simulate_pf_pnl(pf, df0.loc[ts_pre_release], df1.loc[ts], ts) for ts in v_ts]
            figdSdIV.add_trace(go.Scatter(x=v_dSpct_actual, y=v_dPL_pf_actual, mode='markers', name=f'{dt} v_dPL_pf_actual', marker=dict(size=marker_size)), row=1, col=1)

        add_plot_term_structure(df1, calcDate1, figdSdIV, 2, dIV_levels=(0,), ts_time_power=ts_time_power)

    show(figdSdIV, f'''{sym}_{release_date.strftime('%y%m%d')}_figdSdIV_iteration_{iteration}.html''')


def plot_spot(df, df0, exp_jump1, ts_post_release, release_date, ts_pre_release, v_ts_post_release, s0, sym, marker_size=4):
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
    expiries = list(sorted(set(df0.index.get_level_values('expiry').values)))
    plot_dates = expiries  # release_date + timedelta(days=i) for i in [0, 30, 60, 90, 120, 150, 180, 270, 365, 500]]
    figIvOverTime = make_subplots(rows=len(plot_dates), cols=1, subplot_titles=[dt.isoformat() for dt in plot_dates])
    if v_ts_post_release:
        df_pre_post = df.sort_index().loc[(slice(pd.Timestamp(ts_pre_release.date()), pd.Timestamp(ts_post_release.date() + timedelta(days=2))))]
    else:
        df_pre_post = df.sort_index().loc[(slice(pd.Timestamp(ts_pre_release.date()), pd.Timestamp(ts_pre_release.date() + timedelta(days=2))))]
    for i, dt in enumerate(plot_dates):
        df_mny1_tenor_dt = iv_at_date_over_time(df_pre_post, dt, 1)
        for right, s_df in df_mny1_tenor_dt.groupby('right'):
            ix = s_df.index[(s_df['iv'] > s_df['iv'].quantile(0.05)) & (s_df['iv'] < s_df['iv'].quantile(0.95))]
            figIvOverTime.add_trace(go.Scatter(x=s_df.loc[ix, 'ts'], y=s_df.loc[ix, 'iv'], mode='markers', marker=dict(size=marker_size), name=f'{right}: {dt}'), row=i + 1, col=1)
            figIvOverTime.add_vline(x=datetime.fromisoformat(release_date.isoformat()) + timedelta(hours=16), line_dash='dash', line_color='red')
    figIvOverTime.update_layout(title=f'{sym} ATM IV over Time', xaxis_title='Time', yaxis_title='IV by expiry')
    show(figIvOverTime, f'''{sym}_{release_date.strftime('%y%m%d')}_iv_over_time.html''')


def plot_iv_over_time(pf, df, v_ts_post_release, ts_pre_release, ts_post_release, release_date, sym, marker_size=4):
    figIvOverTimeHoldings = go.Figure()
    if v_ts_post_release:
        df_pre_post = df.sort_index().loc[(slice(pd.Timestamp(ts_pre_release.date()), pd.Timestamp(ts_post_release.date() + timedelta(days=2))))]
    else:
        df_pre_post = df.sort_index().loc[(slice(pd.Timestamp(ts_pre_release.date()), pd.Timestamp(ts_pre_release.date() + timedelta(days=2))))]
    for i, (sec, q) in enumerate(pf.holdings.items()):
        if isinstance(sec, Equity):
            continue
        s_df = df_pre_post.loc[(slice(None), sec.expiry, sec.optionContract.strike, sec.right)]
        ix = s_df.index
        figIvOverTimeHoldings.add_trace(go.Scatter(x=ix, y=s_df.loc[ix, 'mid_iv'], mode='markers', marker=dict(size=marker_size), name=f'{str(sec)}'))
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


def option_in_pf_scope(o: Option, min_tenor, moneyness_limits, calcDate0, s0, p0, abs_delta_limits=(), scoped_expiries=None, exclude: List[Option]=None, pf: Portfolio=None):
    if pf and o in pf.holdings:
        return True

    if exclude and o in exclude:
        return False

    # Want large tenors as their IV decreases less after release date.
    if scoped_expiries and o.expiry not in scoped_expiries:
        return False

    if tenor(o.expiry, calcDate0) < min_tenor:
        return False

    moneyness = float(o.strike) / s0
    if moneyness < moneyness_limits[0] or moneyness > moneyness_limits[1]:
        return False

    iv0 = o.iv(p0, s0, calcDate0)
    if iv0 == 0 or np.isnan(iv0):
        return False

    if abs_delta_limits:
        delta = abs(o.delta(iv0, s0, calcDate0))
        if delta < abs_delta_limits[0] or delta > abs_delta_limits[1]:
            return False

    return True


@time_it
def rank_hedges(pf0: Portfolio, pf_delta_total: float, nlv01: pd.DataFrame, delta0: np.array, s0, nlv_pf0_no_additions, calcDate1: date,
                cache_iv1: Dict[Union[Option, Equity], Dict[float, float]],
                v_ds_ret=np.linspace(0.8, 1.2, 21),
                f_weight_ds: Callable[[float], float] = lambda x: 1,
                f_weight_risk: Callable[[float], float] = lambda risk: 1 if risk < 0 else 0.5,
                ) -> Dict[Tuple[str, int], float]:
    """
    Very inefficient at finding the optimal combinations. Currently performs iterative brute force search by considering all possible combinations of 2 options to be added
    to the portfolio. This may not find combination where one needs, eg, a central call/put near short hedged with multiple long far calls/puts.
    Works if portfolio is initialized with preset options. Moving from iteratively finding the single best options to using a combination already improved simulation results.

    numba.core.errors.TypingError: Failed in nopython mode pipeline (step: Literal propagation)
    Failed in literal_propagation_subpipeline mode pipeline (step: performs partial type inference)
    Untyped global name 'nlv': Cannot determine Numba type of <class 'function'>
    File "earnings_release.py", line 442:
    def rank_hedges(pf0: Portfolio, nlv01: pd.DataFrame, s0, nlv_pf0_no_additions, calcDate1: date,
        <source elided>
        v_nlv_pf1_no_additions = pd.Series([sum([nlv(sec, s0 * ds_ret, cache_iv1[sec][ds_ret], q, calcDate1) for sec, q in pf0.holdings.items()]) for ds_ret in v_ds_ret], index=v_ds_ret)
        ^

    This error may have been caused by the following argument(s):
    - argument 0: Cannot determine Numba type of <class 'options.typess.portfolio.Portfolio'>
    - argument 1: Cannot determine Numba type of <class 'pandas.core.frame.DataFrame'>
    - argument 4: Cannot determine Numba type of <class 'datetime.date'>
    - argument 5: Cannot determine Numba type of <class 'dict'>
    - argument 7: Cannot determine Numba type of <class 'function'>
    - argument 8: Cannot determine Numba type of <class 'function'>
    """
    assert delta0.shape[0] == nlv01.shape[1]
    n_ds = nlv01.shape[0]
    n_options = nlv01.shape[1]
    n_combinations = 2

    v_nlv_pf1_no_additions = pd.Series([sum([nlv(sec, s0 * ds_ret, cache_iv1[sec][ds_ret], q, calcDate1) for sec, q in pf0.holdings.items()]) for ds_ret in v_ds_ret], index=v_ds_ret)
    v_dnlv_pf01_no_additions = v_nlv_pf1_no_additions - nlv_pf0_no_additions
    dim_quantity_axis = 2 ** n_combinations
    m_q = np.array(list(itertools.product([1, -1], repeat=n_combinations))).T
    # m_q = np.array([[1], [1]])  # Basic, 2 combos, only longing
    n_q = m_q.shape[-1]

    # m -> matrix of all combinations. each cell is sum of NLVs of respective option for respective ds_ret
    # Initializing empty matrix
    m = np.empty(tuple([n_q, n_ds] + [n_options] * n_combinations))  # n_ds_ret; n_options; n_options
    assert m.shape == tuple([n_q, n_ds] + [n_options] * n_combinations)

    m_delta = np.empty(tuple([n_q] + [n_options] * n_combinations))

    # Filling with combination 1
    for k, q in enumerate(m_q[0]):
        m_delta[k] = q * delta0
        for i in range(n_ds):
            m[k, i] = q * nlv01.values[i]

    assert m[0][1][0].sum() == m[0][1][1].sum()
    assert m[0][0][1].sum() == m[0][0][2].sum()
    assert m[1][1][1].sum() == m[1][1][2].sum()

    # Transforming to combinations(2) by summing same, but transformed/rotated.
    for k, q in enumerate(m_q[1]):
        for j in range(n_options):
            for c in range(n_combinations-1):
                ix = tuple([slice(None)] + [j] * (c+1))
                m_delta[k][ix] += q * delta0

        for i in range(n_ds):
            for j in range(n_options):
                for c in range(n_combinations - 1):
                    ix = tuple([slice(None)] + [j] * (c + 1))
                    m[k][i][ix] += q * nlv01.values[i]

    assert np.sum(m[tuple([0, 0] + [0] * n_combinations)]) == nlv01.values[0, 0] * n_combinations
    assert np.sum(m[tuple([0, 1] + [1] * n_combinations)]) == nlv01.values[1, 1] * n_combinations

    # Across all ds_ret dimensions add (pf1-pf0) from each cell
    for k in range(m.shape[0]):
        for i in range(n_ds):
            m[k, i] += v_dnlv_pf01_no_additions.iloc[i]

    # Want to remove combinations that increase absolute delta. Get delta for each combination... Sum with portfolio delta...
    for k in range(n_q):
        m_delta_exclude = m_delta[k] * pf_delta_total > 0
        m[k] = np.where(m_delta_exclude, -np.inf, m[k])

    # Would not want to reduce quantity of existing portfolio, hence setting corresponding cells to -inf
    for o, q in pf0.holdings.items():
        if isinstance(o, Equity):
            continue
        if str(o) not in nlv01.columns:
            continue
        i = nlv01.columns.get_loc(str(o))
        for k in [i for i, qq in enumerate(m_q[0]) if abs(qq + q) < abs(qq)]:
            for ix in itertools.permutations(tuple([i] + [slice(None)] * (n_combinations - 1))):
                # m[k, :][ix] = -np.inf
                pass

    # Apply ds_ret weights
    v_f_weight_ds = np.array([f_weight_ds(ds_ret) for ds_ret in v_ds_ret])
    for k in range(m.shape[0]):
        for i in range(n_ds):
            m[k, i] *= v_f_weight_ds[i]

    # Apply risk weight to every cell in array.
    m = np.where(m > 0, m*0.5, m)

    # Now sum across ds_ret dimension to get a net risk
    risk_by_combination = np.sum(m, axis=1)
    assert risk_by_combination.shape == tuple([n_q] + [n_options] * n_combinations)

    ix_best_combo = np.unravel_index(np.argmax(risk_by_combination), risk_by_combination.shape)

    qi = lambda ix_m, i: m_q[ix_best_combo[0]][i]
    option_nm_i = lambda i: nlv01.columns[ix_best_combo[i]]
    ix_i = lambda i: [ix_best_combo[i]] * n_combinations
    d_nlv_i = lambda i: risk_by_combination[tuple([ix_best_combo[0]] + ix_i(i))] / n_combinations

    return {(option_nm_i(i), qi(ix_best_combo[0], i)): d_nlv_i(i) for i in range(n_combinations)}


def simulate_pf_pnl(pf, df0_esr, df1, ts):
    nlv0 = 0
    nlv1 = 0
    for sec, q in pf.items():
        if isinstance(sec, Equity):
            p0 = df0_esr.iloc[0]['spot']
            p1 = df1.iloc[0]['spot']
            nlv0 += q * p0
            nlv1 += q * p1
            continue

        o = sec
        p0 = val_from_df(df0_esr, o.expiry, o.optionContract.strike, o.right, 'mid_price')

        try:
            p1 = val_from_df(df1, o.expiry, o.optionContract.strike, o.right, 'mid_price')
            if np.isnan(p0) or np.isnan(p1):
                print(f'Error Simulate_pf_pnl at {ts}: NaN price for {o} {p0} {p1}')
                continue

            nlv0 += p0 * q * 100
            nlv1 += p1 * q * 100

        except KeyError as e:
            # Expired option - intrinsic value only
            spot = val_from_df(df0_esr, slice(None), slice(None), o.right, 'spot')[0]
            p1 = spot - float(o.strike) if o.right == 'call' else float(o.strike) - spot
            p1 = max(p1, 0)
            nlv0 += p0 * q * 100
            nlv1 += p1 * q * 100
            # print(f'simulate_pf_pnl: KeyError for {o}')

    return nlv1 - nlv0


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
                fwd_s = s * np.exp(net_yield * tenor(expiry, date.today()))
                iv = moneyness_iv(ss_df.droplevel(-1), right_moneyness, expiry, fwd_s)
                v_atexpiry_iv.append(iv)
            atexpiry_iv = np.interp((dt - at_the_date_expiries[0]).days, [0, (at_the_date_expiries[1] - at_the_date_expiries[0]).days], v_atexpiry_iv)
            tenor_ = tenor(dt, ts.date())
            results.append(IVByTime(ts, atexpiry_iv, right_moneyness, tenor_, right))
    return pd.DataFrame(results)


def estimate_iv_change():
    """
    factors: tenor, infinite vola, nearest neighbor (market) vola, term structure, fair ts? (power law), skew, fair skew?,
    """


def stress_pf(pf: Portfolio, s0, ds_ret: float, calcDate0, calcDate1: date, use_skew, df0, d_iv_tenor=None) -> float:
    """
    At the very least, return pf nlv after ds_ret.
    Then in addition, also change dIV.
    No equity hedging here

    Refactoring to a pandas free object than can bt jit compiled and run in GPU. Numpy
    np array with 4d: [expiry or tenor, strike or mnyness, right, ds_ret].
    For _0 and _1
    Metrics are: iv1, p, s, delta, skew, dIV, dIV_skew, dIV_tenor, npv
    """
    s1 = s0 * ds_ret
    ds = (s1 - s0)
    nlv0 = 0
    nlv1 = 0
    for sec, q in pf.items():
        if isinstance(sec, Equity):
            nlv0 += q * s0
            nlv1 += q * s1
        else:
            o = sec
            ps_o = df0.loc[(o.expiry, o.optionContract.strike, o.right)]

            p0 = ps_o['mid_price']
            iv0 = o.iv(p0, s0, calcDate0)
            if iv0 == 0 or np.isnan(iv0):
                raise ValueError(f'IV0 is 0 or NaN for {o} {p0} {s0} {calcDate0}')

            nlv0 += o.npv(iv0, s0, calcDate0) * q * 100

            # if o.expiry in jump_expiries:  # shouldn't rely on me defining what drops. every expiry will drop. gotta estimate a iv_srf_1(s)
            #     iv1 = iv0 + iv0 * ps_o[f'd_iv_ts']
            # else:
            #     # square root of time rule starting from a presume IV level move for tenor 1.
            if use_skew:
                skew_col_nm = col_nm_mean_regressed_skew_ds(ds_ret)
                try:
                    skew = df0.loc[(sec.expiry, sec.optionContract.strike, sec.right)][skew_col_nm]
                except KeyError:
                    print(f'KeyError: No Skew for {o.expiry} {o.optionContract.strike} {o.right} {(ds_ret):.2f}')
                    skew = 0
                skew = 0 if np.isnan(skew) else skew
                dIV_skew = skew * -ds
            else:
                dIV_skew = 0

            dIV_tenor = d_iv_tenor if d_iv_tenor else ps_o[f'd_iv_ts']
            # Even better: identify which expiries are currently off the expected power law, and modify ts_time_power factor anticipating their correction. Rather than power law
            # might rather compare normalized term structures.

            iv1 = iv0 + dIV_tenor + dIV_skew

            nlv1 += o.npv(iv1, s1, calcDate1) * q * 100
    return nlv1 - nlv0


def npv(sec: Union[Option, Equity], s0, iv: float, calc_date: date) -> float:
    return sec.npv(iv, s0, calc_date) if isinstance(sec, Option) else s0


def nlv(sec: Union[Option, Equity], s0, iv: float, q, calc_date: date) -> float:
    if not iv:
        return None
    if isinstance(sec, Option):
        # try:
        return sec.npv(iv, s0, calc_date) * q * 100
        # except RuntimeError as e:
        #     print(f'Error in npv for {sec} {s0} {iv} {calc_date}')
        #     return np.nan
    else:
        return q * s0


def delta(sec: Union[Option, Equity], s0, iv: float, q, calc_date: date) -> float:
    if isinstance(sec, Option):
        return sec.delta(iv, s0, calc_date) * q * 100
    else:
        return q * s0


def iv_as_of(o: Option, df0, calc_date0: date, as_of: date, ds_ret, use_skew: bool, iv_fallback=None, jump_expiries=None, diffusion_iv=None):
    if isinstance(o, Equity):
        return 0
    ps_o = df0.loc[(o.expiry, o.optionContract.strike, o.right)]
    s0 = ps_o['spot']
    s1 = s0 * ds_ret
    ds = (s1 - s0)

    p0 = ps_o['mid_price']
    iv0 = o.iv(p0, s0, calc_date0)
    if iv0 == 0 or np.isnan(iv0):
        if iv_fallback != 0 and iv_fallback is not None:
            return iv_fallback
        logger.error(ValueError(f'IV0 is 0 or NaN for {o} {p0} {s0} {calc_date0}'))
        return np.nan

    if calc_date0 == as_of:
        return iv0

    if jump_expiries and o.expiry in jump_expiries:  # shouldn't rely on me defining what drops. every expiry will drop. gotta estimate a iv_srf_1(s)
        iv1 = diffusion_iv if diffusion_iv else iv0 + ps_o[f'd_iv_ts']
        # This clause is to be deleted and delegated to the dIV model currently feeding ps_o[f'd_iv_ts']
    else:
        if use_skew:
            skew_col_nm = col_nm_mean_regressed_skew_ds(ds_ret)
            try:
                skew = ps_o[skew_col_nm]
            except KeyError:
                print(f'KeyError: No Skew for {o.expiry} {o.optionContract.strike} {o.right} {(ds_ret):.2f}')
                skew = 0
            skew = 0 if np.isnan(skew) else skew
            dIV_skew = skew * -ds
        else:
            dIV_skew = 0

        dIV_tenor = ps_o[f'd_iv_ts']

        iv1 = iv0 + dIV_tenor + dIV_skew
    return iv1


@lru_cache(maxsize=1000)
def d_iv_power_law(d_iv_tenor_1, expiry: date, calcDate: date, ts_time_power=0.5):
    tenor_ = tenor(expiry, calcDate)
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
            tenor_near = tenor(exp_near, ts.date())
            tenor_far = tenor(exp_far, ts.date())

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
            tenor_ = tenor(expiry, ts.date())
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


def add_plot_iv(df, ts, pf, fig, row, col, marker_size_iv=3, marker_size_holdings=8):
    calcDate = ts.date()
    holding_expiries = list(set([sec.expiry for sec in pf.holdings.keys() if isinstance(sec, Option)]))

    for expiry in holding_expiries:
        s_df = df.loc[(ts, expiry, slice(None), slice(None))]

        # Plot IVs
        for right, ss_df in s_df.groupby(level='right'):
            sample_df = ss_df[ss_df['mid_iv'].fillna(0) != 0]
            fig.add_trace(go.Scatter(x=sample_df['moneyness'], y=sample_df['mid_iv'], mode='lines+markers', name=f'IV {expiry} {right}', marker=dict(size=marker_size_iv)), row=row,
                          col=col)

    right_symbol_map = {'call': 'x-dot', 'put': 'circle-dot'}
    for right, s_df in df.loc[ts].groupby(level='right'):
        holdings = {sec: q for sec, q in pf.holdings.items() if isinstance(sec, Option) and sec.right == right}
        s = s_df.iloc[0]['spot']
        x_moneyness = [float(sec.optionContract.strike) / s for sec in holdings.keys()]
        y_iv = [sec.iv(s_df.loc[(sec.expiry, sec.optionContract.strike, sec.right), 'mid_price'], s, calcDate) for sec in holdings.keys()]
        if 0 in y_iv:
            raise ValueError(f'Zero IV for {right} {ts} {x_moneyness} {y_iv}')

        fig.add_trace(go.Scatter(
            x=x_moneyness, y=y_iv, mode='markers', name=f'IV Holdings {right}',
            text=[f'{str(sec)}: {q}' for sec, q in holdings.items()],
            marker=dict(size=marker_size_holdings, symbol=right_symbol_map[right])), row=row, col=col)


def descope_options(scoped_options, s0, n=400):
    """
    enforce 400 by removing options from the expiries containing most contracts. always remove option with greatest log(moneness) until 400 is reached.
    """
    i = len(scoped_options)
    while i > n:
        dct_exp_n = {exp: len(list(ops)) for exp, ops in itertools.groupby(scoped_options, lambda x: x.expiry)}
        exp_reduce = max(dct_exp_n, key=dct_exp_n.get)
        rm_o = sorted([o for o in scoped_options if o.expiry == exp_reduce], key=lambda x: abs(np.log(float(x.strike) / s0)))[-1]
        scoped_options.remove(rm_o)
        i -= 1
    return scoped_options


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
    holding_expiries = list(set([sec.expiry for sec in pf.holdings.keys() if isinstance(sec, Option)]))
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
    sym = 'TGT'
    # sym = 'WDAY'
    # sym = 'PATH'
    # sym = 'DELL'
    path_model = os.path.join(Paths.path_models, f'earnings_iv_drop_regressor_2024-03-20b.json')
    earnings_iv_drop_regressor = EarningsIVDropRegressor().load_model(path_model)

    cfg = EarningsConfig(sym, take=-2, plot=True, plot_last=True, earnings_iv_drop_regressor=earnings_iv_drop_regressor,
                         moneyness_limits=(0.7, 1.3), abs_delta_limits=(0.1, 0.9),
                         min_tenor=0.0, use_skew=True, add_equity_holdings=True)
    get_earnings_release_pf(cfg)


if __name__ == '__main__':
    run()
