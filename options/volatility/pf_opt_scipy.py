import os
from pprint import pprint

import numpy as np
import pandas as pd
from gekko import GEKKO

from collections import defaultdict
from datetime import timedelta
from decimal import Decimal
from scipy.optimize import minimize

from options.helper import year_quarter, enrich_atm_iv, val_from_df
from options.typess.equity import Equity
from options.typess.option import Option
from options.typess.option_contract import OptionContract
from options.typess.option_frame import OptionFrame
from options.typess.portfolio import Portfolio
from options.volatility.earnings_release import EarningsConfig, iv_as_of, option_in_pf_scope, nlv, delta, get_f_weight_ds_ret
from options.volatility.estimators.earnings_iv_drop_regressor import EarningsIVDropRegressor
from shared.constants import EarningsPreSessionDates
from shared.paths import Paths
from collections import OrderedDict


def sample_optimization():
    def sample_fobj(x):
        x0, x1, x2 = x
        return -1 * x0 + 5 * x1 + x2 ** 2

    x0 = np.array([1, 1, 1])

    def con(x):
        print(x[0])
        return x[0] + x[1]

    constraints = [
        {'type': 'eq', 'fun': con},
    ]
    val = minimize(sample_fobj, x0, args=(), method='SLSQP', bounds=None, constraints=constraints, tol=None, callback=None, options=dict(maxiter=100))
    print(val)
    print(sample_fobj(val.x))


if __name__ == '__main__':
    sym = 'TGT'
    path_model = os.path.join(Paths.path_models, f'earnings_iv_drop_regressor_2024-03-20b.json')
    earnings_iv_drop_regressor = EarningsIVDropRegressor().load_model(path_model)

    cfg = EarningsConfig(sym, take=-2, ranking_iterations=20, early_stopping=False, plot=True, plot_last=True, earnings_iv_drop_regressor=earnings_iv_drop_regressor,
                         moneyness_limits=(0.7, 1.3), min_tenor=0.0, use_skew=False)
    equity = Equity(sym)
    release_date = EarningsPreSessionDates(sym)[cfg.take]

    option_frame = OptionFrame.load_frame(equity, cfg.resolution, cfg.seq_ret_threshold, year_quarter(release_date))
    df = option_frame.df_options.sort_index()

    ts = df.index.get_level_values('ts').unique()
    v_ts_pre_release = [i for i in ts if i.date() <= release_date]  # preparing on day of expiry
    v_ts_pre_release = [i for i in v_ts_pre_release if i.hour >= 10]

    ts_pre_release = v_ts_pre_release[-2]
    calcDate0 = ts_pre_release.date()
    calcDate1 = release_date + timedelta(days=1)
    df0 = df.loc[[ts_pre_release]]

    # Enrich ATM IV
    enrich_atm_iv(df0)
    d_iv_pct = cfg.earnings_iv_drop_regressor.predict(df0)
    df0['d_iv_ts'] = df0['atm_iv'] * d_iv_pct

    s0 = df0.loc[ts_pre_release].iloc[0]['spot']
    print(f'''Spot0: {s0}; Spot1''')

    # Create option universe and enrich frame with option objects
    option_universe = {}
    for ix in df0.index:
        ts, expiry, strike, right = ix
        if expiry < release_date:
            continue
        o = Option(OptionContract('', sym, expiry, strike, right), ts_pre_release.date(), s0, 0)
        option_universe[str(o)] = o
    print(f'Added {len(option_universe)} options to universe')

    expiries = list(sorted(set(df0.index.get_level_values('expiry').values)))

    exp_jump1 = sorted([i for i in expiries if i >= release_date])[0]
    exp_jumps = [exp_jump1]

    v_ds_pct = [100 * (dS - 1) for dS in cfg.v_ds_ret]

    # Initialize portfolio of near short options
    strikes = pd.Series(np.unique(df0.loc[(ts_pre_release, exp_jump1, slice(None), slice(None))].index.get_level_values('strike').astype(float)))
    strikes_atm = strikes.iloc[(strikes - s0).abs().sort_values().head(2).index].values

    print(f'ATM strike: {strikes_atm}')
    pf = Portfolio()
    for sec, q in {
        option_universe[OptionContract('', sym, exp_jump1, Decimal(strikes_atm[0]), 'call').ib_symbol()]: -1 * cfg.short_q_scale,
        option_universe[OptionContract('', sym, exp_jump1, Decimal(strikes_atm[0]), 'put').ib_symbol()]: -1 * cfg.short_q_scale,
        option_universe[OptionContract('', sym, exp_jump1, Decimal(strikes_atm[1]), 'call').ib_symbol()]: -1 * cfg.short_q_scale,
        option_universe[OptionContract('', sym, exp_jump1, Decimal(strikes_atm[1]), 'put').ib_symbol()]: -1 * cfg.short_q_scale,
    }.items():
        pf.add_holding(sec, q)

    pf = Portfolio(defaultdict(int, {**pf.holdings}))

    cache_iv0 = {sec: {1: iv_as_of(sec, df0.loc[ts_pre_release], calcDate0, calcDate0, exp_jumps, None, 1, cfg.use_skew)} for sec in
                 list(option_universe.values()) + [Equity(sym)]}
    cache_iv1 = {sec: {ds_ret: iv_as_of(sec, df0.loc[ts_pre_release], calcDate0, calcDate1, exp_jumps, None, ds_ret, cfg.use_skew) for ds_ret in cfg.v_ds_ret} for sec in
                 list(option_universe.values()) + [Equity(sym)]}
    # kick out options for which iv values are missing
    len(option_universe)
    rm_option = set([o for o in option_universe.values() if pd.isna(pd.Series(cache_iv1[o].values())).sum() > 0])
    rm_option.union(set([o for o in option_universe.values() if pd.isna(pd.Series(cache_iv0[o].values())).sum() > 0]))
    len(rm_option)
    print(f'Kicking out {len(rm_option)} options from option universe due to missing IV values: {rm_option}')
    for o in rm_option:
        if o in cache_iv0:
            cache_iv0.pop(o)
        if o in cache_iv1:
            cache_iv1.pop(o)
        if str(o) in option_universe:
            del option_universe[str(o)]

    get_p0 = lambda o: val_from_df(df0.loc[ts_pre_release], o.expiry, o.optionContract.strike, o.right, 'mid_price')
    scoped_options = [o for o in option_universe.values() if option_in_pf_scope(o, cfg.min_tenor, cfg.moneyness_limits, calcDate0, s0, get_p0(o))]

    # drop some weeklies to reduce scope
    scoped_expiries = [expiries[0]] + expiries[4:]
    scoped_options = [o for o in scoped_options if o.optionContract.expiry in scoped_expiries]

    scoped_option_names = [str(o) for o in scoped_options]
    nlv0 = np.array([nlv(o, s0, cache_iv0[o][1], 1, calcDate0) for o in scoped_options])
    v_delta0 = np.array([delta(o, s0, cache_iv0[o][1], 1, calcDate0) for o in scoped_options])
    ps_delta0 = pd.Series(v_delta0, index=scoped_option_names)
    df_nlv01 = pd.DataFrame([np.array([nlv(o, s0 * ds_ret, cache_iv1[o][ds_ret], 1, calcDate1) for o in scoped_options]) - nlv0 for ds_ret in cfg.v_ds_ret], index=cfg.v_ds_ret,
                         columns=scoped_option_names)
    df_nlv01 = df_nlv01.T.sum(axis=1)
    m_nlv01 = np.array([[nlv(o, s0 * ds_ret, cache_iv1[o][ds_ret], 1, calcDate1) for o in scoped_options] - nlv0 for ds_ret in cfg.v_ds_ret])

    """
        Define an objective function   d NLV = -f(T, K, right)
        Constraints: Right: 0/1 ; K: discrete, T: discrete
        Minimize sum(d NLV) for n options.
        Parameters: n, option universe, pf0

        SLSQP.
        Variables: Assuming we want to find an optimal pf consisting of 20 options. The NLV of each call/put can be expressed as some
        differentiable function of strike K.
        The the solver could received l-expiries (eg 10) * 2-right variables * 3-quantity (1, 0, -1). Starting guess is strike close to moneyness 1.
        So the objective function takes in 60 variables and returns the sum of the NLV of the portfolio.
        Constraints:
            Total abs(quantity) <= : parameter
            -20 < pf_delta < 20, skip at beginning
        Starting condition:
            pf0
            option universe

        Variables: Vector of length of universe. Each cell is a quantity of the option, 0, -1, 1. Constraint: abs(sum(v_var) <= 20
        m_universe[v_selected]*v_quantity. Sum along ds_return axis (potentially skip). just return sum!

        possible combinations of 20 options. trillions and trillions.

        Perhaps can simplify by removing strike category: vector of possible (expiry, right) pairs. Value of variable is not quantity, delta [-1 -> 1] encoding strike and quantity
    """

    # Apply ds_ret weights
    f_weight_ds = get_f_weight_ds_ret(0.1, plot=False)
    n_ds = len(cfg.v_ds_ret)
    v_f_weight_ds = np.array([f_weight_ds(ds_ret) for ds_ret in cfg.v_ds_ret])
    for k in range(m_nlv01.shape[0]):
        for i in range(n_ds):
            m_nlv01[k, i] *= v_f_weight_ds[i]
    m_nlv01_summed = np.sum(m_nlv01, axis=0)

    max_n_strikes = 100 * 2

    exp_right_pairs = []
    dct_exp_right2option = OrderedDict()
    for o in scoped_options:
        pair = (o.optionContract.expiry, o.right)
        if pair not in dct_exp_right2option:
            dct_exp_right2option[pair] = []
        if pair not in exp_right_pairs:
            exp_right_pairs.append(pair)
        dct_exp_right2option[pair].append(o)
    dct_exp_right2option_delta = {k: [ps_delta0.loc[str(o)] for o in v] for k, v in dct_exp_right2option.items()}
    dct_exp_right2option_nlv = {k: [df_nlv01.loc[str(o)] for o in v] for k, v in dct_exp_right2option.items()}

    x0 = np.zeros(len(exp_right_pairs))
    m_option2delta_edge2nlv = np.empty((len(x0), max_n_strikes, 2))
    m_option2delta_edge2nlv.fill(np.inf)

    for i, (ix, v_deltas) in enumerate(dct_exp_right2option_delta.items()):
        expiry, right = ix
        v_nlv = dct_exp_right2option_nlv[ix]
        if right == 'call':
            # Here, very discontinuous. Imagine how
            v_deltas = list(-1*np.array(v_deltas)) + v_deltas[::-1]
            v_nlv = list(-1*np.array(v_nlv)) + v_nlv[::-1]
        else:
            v_deltas = v_deltas[::-1] + list(-1*np.array(v_deltas))
            v_nlv = v_nlv[::-1] + list(-1*np.array(v_nlv))
        m_option2delta_edge2nlv[i, :len(v_deltas), 0] = v_deltas
        m_option2delta_edge2nlv[i, :len(v_deltas), 1] = v_nlv

    def map_x2nlv(x, m_delta_edges2nlv, ix_x):
        print(x)
        ix = np.argmin(np.abs(m_delta_edges2nlv[:, :, 0] - x[:, np.newaxis]), axis=1)
        # Need to interpolate here somehow, given that optimization rather fails without
        return m_delta_edges2nlv[ix_x, ix, 1]

    # map from delta to strike, quantity mapped to an NLV, essentially map delta to NLV            ... or just equity for very, very high abs delta / low gamma
    # for each exp, right -> define delta edge values for each strike/quantity. write some vectorized func mapping float value to an index in the array. np.argmax/min should do it.
    # From that index, get the NLV value. Then sum over all options. return -sum.
    # m_delta_edges - first index level maps to expiry, right pair. second index level maps to the delta edge values.. third index differeniates between delta and nlv

    def fobj(x: np.ndarray, m_option2delta_edge2nlv) -> float:
        """
        Maximize: sum(NLV) / (Slope * some_factor_to_calibrate) across ds_ret vector.
        The variables are the quantities of options.
        Constraint:
            - Each quantity cannot be more than x or less than -x
            - Abs sum of quantities cannot exceed x
            - must be integer: given by algorithm
        """
        ix_x = list(range(len(x)))
        # Each value in a variable x corresponds to a nlv value. Map and sum. Constraints handled separately
        v_nlv = np.apply_along_axis(map_x2nlv, 0, x, m_option2delta_edge2nlv, ix_x)
        if np.any(np.isnan(v_nlv)):
            raise ValueError('NaN in NLV')
        return -np.sum(v_nlv)

    n_options = len(scoped_options)
    n_options = 400   # This and 21 ds_ret is the limit currently. More breaks when searching to maximize the greatest loss.
    n_contracts = 20
    g_scoped_options = scoped_options[:n_options]

    m = GEKKO(remote=False)
    m.options.SOLVER = 1  # APOPT is an MINLP solver

    # optional solver settings with APOPT
    m.solver_options = [
        'minlp_maximum_iterations 500', \
        # minlp iterations with integer solution
        'minlp_max_iter_with_int_sol 500', \
        # treat minlp as nlp
        # 'minlp_as_nlp 0', \
        # nlp sub-problem max iterations
        # 'nlp_maximum_iterations 50', \
        # 1 = depth first, 2 = breadth first
        # 'minlp_branch_method 1', \
        # maximum deviation from whole number
        'minlp_integer_tol 0.1', \
        # covergence tolerance
        'minlp_gap_tol 0.01'
    ]

    # Constants
    ds_per_side = m_nlv01.shape[0] // 2

    # Parameters
    g_m_nlv01 = m_nlv01[:, :n_options]
    g_v_delta0 = v_delta0[:n_options]

    # Variables
    v_var = m.Array(m.Var, n_options, value=0, lb=-n_contracts, ub=n_contracts, integer=True)

    # Intermediates
    t = [m.Intermediate(v_var @ g_m_nlv01[i]) for i in range(g_m_nlv01.shape[0])]
    neg_ds = m.Intermediate(sum(t[5:ds_per_side]))
    pos_ds = m.Intermediate(sum(t[-ds_per_side:-5]))

    mn = t[0]  # min
    for i in range(1, len(t)):
        mn = m.min3(mn, t[i])

    # Constraints
    m.Equation(m.sum([m.abs2(v) for v in v_var]) <= n_contracts)

    # Objectives.
    # NLV
    m.Maximize(m.sum(t))  # Multiple object is 10x faster than breaking it up into multiple objectives
    # m.Maximize(m.sum(sum(v_var * g_m_nlv01)))  # Multiple object is 10x faster than breaking it up into multiple objectives
    # for i in range(g_m_nlv01.shape[0]):
    #     m.Maximize(m.sum(v_var * g_m_nlv01[i]))

    # Directional risk. Minimizing delta total across ds_ret
    risk_adj_power = 2
    m.Minimize(abs(neg_ds - pos_ds) ** risk_adj_power)
    # m.Minimize(abs(v_var @ g_v_delta0) ** risk_adj_power)
    # for i in range(g_v_delta0.shape[0]):
    #     m.Minimize(m.abs2(v_var[i] * g_v_delta0[i]) ** risk_adj_power)

    # Max Loss
    weight_max_min_nlv = 10
    # m.Maximize(weight_max_min_nlv * m.min2(v_nlv_by_ds))
    m.Maximize(weight_max_min_nlv * mn)

    # initialize with IPOPT finding approximate solution
    # m.options.SOLVER = 3
    # m.solve(disp=False)

    # solve with APOPT for integer constraint
    m.options.SOLVER = 1
    m.solve(disp=True)

    print(v_var)
    print('Objective: ', m.options.objfcnval)
    obj_m_risk_adj = -m.options.objfcnval - np.array([v.value[0] for v in v_var]) @ g_v_delta0
    print(f'-object - risk adjustment: {obj_m_risk_adj}')

    sol_dnlv_total = np.sum(np.sum(np.array([v.value[0] for v in v_var]) * g_m_nlv01, axis=1))
    print(f'dNLV Total: {sol_dnlv_total}')

    dnlv_by_ds = np.sum(np.array([v.value[0] for v in v_var]) * g_m_nlv01, axis=1)
    print(f'dLNLV by ds: {pd.Series(dnlv_by_ds, index=cfg.v_ds_ret)}')

    sol_delta_total = np.array([v.value[0] for v in v_var]) @ g_v_delta0
    print(f'Solution: Delta total: {sol_delta_total}, Weighted: {abs(sol_delta_total)**risk_adj_power}')

    # possible vals:
    # print('Delta total going fully long each contract')
    # print([np.eye(n_options, n_options)[i]*n_contracts @ g_v_delta0 for i in range(n_options)])

    pprint({str(g_scoped_options[i]): v.value[0] for i, v in enumerate(v_var) if v.value[0] != 0})
    m.cleanup()
    print('Done.')
