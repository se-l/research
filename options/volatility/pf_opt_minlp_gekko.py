import itertools
import os
import numpy as np
import pandas as pd

from pprint import pprint
from gekko import GEKKO
from datetime import timedelta

from options.helper import year_quarter, enrich_atm_iv, val_from_df
from options.typess.equity import Equity
from options.typess.option_frame import OptionFrame
from options.typess.portfolio import Portfolio
from options.volatility.estimators.earnings_iv_drop_regressor import EarningsIVDropRegressor
from shared.constants import EarningsPreSessionDates
from shared.paths import Paths


def derive_portfolio_milnp(scoped_options, m_dnlv01, v_delta0, cfg, n_contracts=20, pf=None) -> Portfolio:
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
    if pf:
        n_contracts = max(n_contracts, sum([abs(q) for q in pf.holdings.values()]))
    n_options = len(scoped_options)
    # n_options = 30   # 400 and 21 ds_ret is the limit currently. More breaks when searching to maximize the greatest loss.
    g_scoped_options = scoped_options[:n_options]

    m = GEKKO(remote=False)
    m.options.SOLVER = 1  # APOPT is an MINLP solver
    m.options.IMODE = 3  # Steady state optimization

    # optional solver settings with APOPT
    m.solver_options = [
        'minlp_maximum_iterations 3000', \
        # minlp iterations with integer solution
        'minlp_max_iter_with_int_sol 3000', \
        # treat minlp as nlp
        'minlp_as_nlp 0', \
        # nlp sub-problem max iterations
        'nlp_maximum_iterations 500', \
        # 1 = depth first, 2 = breadth first
        'minlp_branch_method 1', \
        # maximum deviation from whole number
        'minlp_integer_tol 0.1', \
        # covergence tolerance
        'minlp_gap_tol 0.01'
    ]

    # Constants
    ix_neg_ds = [i for i, v in enumerate(cfg.v_ds_ret) if v < 1]
    ix_pos_ds = [i for i, v in enumerate(cfg.v_ds_ret) if v > 1]

    # Parameters
    g_m_dnlv01 = m_dnlv01[:, :n_options]
    g_v_delta0 = v_delta0[:n_options]

    # Variables
    v_var = m.Array(m.Var, n_options, value=0, lb=-n_contracts, ub=n_contracts, integer=True)

    # Intermediates
    if True:
        # 766 sec
        t = [m.Intermediate(v_var @ g_m_dnlv01[i]) for i in range(g_m_dnlv01.shape[0])]

        # They are ds weighted
        neg_ds = m.Intermediate(sum([t[i] for i in ix_neg_ds]))
        pos_ds = m.Intermediate(sum([t[i] for i in ix_pos_ds]))

        mn = t[0]  # min
        for i in range(1, len(t)):
            mn = m.min3(mn, t[i])
    # else:
    #     # 781 sec
    #     neg_ds = 0
    #     pos_ds = 0
    #     t = []
    #     for i in range(g_m_dnlv01.shape[0]):
    #         t.append(m.Intermediate(v_var @ g_m_dnlv01[i]))
    #         mn = m.min3(mn, t[i]) if i > 0 else t[i]
    #         neg_ds += t[i] if i in ix_neg_ds else 0
    #         pos_ds += t[i] if i in ix_pos_ds else 0

    # Constraints
    m.Equation(m.sum([m.abs2(v) for v in v_var]) <= n_contracts)
    if pf:  # If any existing portfolio is provided, add constraints
        assert sum([abs(q) for q in pf.holdings.values()]) <= n_contracts
        for sec, q in pf.holdings.items():
            print(g_scoped_options.index(sec))
            var = v_var[g_scoped_options.index(sec)]
            var.value = q
            m.Equation(var == q)

    # Objectives.
    # Net NLV ( dS weights applied
    m.Maximize(m.sum(t))  # Multiple object is 10x faster than breaking it up into multiple objectives
    # m.Maximize(m.sum(sum(v_var * g_m_dnlv01)))  # Multiple object is 10x faster than breaking it up into multiple objectives
    # for i in range(g_m_dnlv01.shape[0]):
    #     m.Maximize(m.sum(v_var * g_m_dnlv01[i]))

    # Directional risk. Minimizing delta total across ds_ret
    risk_adj_power = 1
    m.Minimize(abs(neg_ds - pos_ds) ** risk_adj_power)

    # Max Loss. It's a variant of applying risk weights incentivizing to minimize the greatest loss
    weight_max_min_nlv = 25
    m.Maximize(weight_max_min_nlv * mn)

    # solve with APOPT for integer constraint
    m.options.SOLVER = 1
    print(f'Solving for #Options: {n_options}, #Contracts: {n_contracts}')
    m.solve(disp=True)
    # try:
    #     m.solve(disp=True)
    # except Exception as e:
    #     print(e)
    #     m.cleanup()
    #     return Portfolio()

    print(f'Objective: {m.options.objfcnval}')

    sol_dnlv_total = np.sum(np.sum(np.array([v.value[0] for v in v_var]) * g_m_dnlv01, axis=1))
    print(f'dNLV Total dS weighted: {sol_dnlv_total}')

    dnlv_by_ds = np.sum(np.array([v.value[0] for v in v_var]) * g_m_dnlv01, axis=1)
    print(f'dLNLV by ds: {pd.Series(dnlv_by_ds, index=cfg.v_ds_ret)}')

    sol_delta_total = np.array([v.value[0] for v in v_var]) @ g_v_delta0
    print(f'Solution: Delta total: {sol_delta_total}, Weighted: {abs(sol_delta_total) ** risk_adj_power}')

    # Possible vals:
    # print('Delta total going fully long each contract')
    # print([np.eye(n_options, n_options)[i]*n_contracts @ g_v_delta0 for i in range(n_options)])

    holdings = {g_scoped_options[i]: v.value[0] for i, v in enumerate(v_var) if v.value[0] != 0}
    pprint(holdings)
    m.cleanup()

    return Portfolio(holdings)


def run():
    sym = 'PANW'
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

    option_universe, cache_iv0, cache_iv1 = create_option_universe_iv_caches(sym, ts_pre_release, df0, release_date, s0, cfg, calcDate0, calcDate1)
    expiries = list(sorted(set(df0.index.get_level_values('expiry').values)))
    exp_jump1 = sorted([i for i in expiries if i >= release_date])[0]

    pf = initialize_portfolio(df0, ts_pre_release, sym, exp_jump1, s0, cfg, option_universe)

    scoped_expiries = get_scoped_expiries(expiries)
    get_p0 = lambda o: val_from_df(df0.loc[ts_pre_release], o.expiry, o.optionContract.strike, o.right, 'mid_price')
    scoped_options = [o for o in option_universe.values() if option_in_pf_scope(o, cfg.min_tenor, cfg.moneyness_limits, calcDate0, s0, get_p0(o), cfg.abs_delta_limits, scoped_expiries=scoped_expiries)]
    print(f'Scoped options: #{len(scoped_options)} out of # {len(option_universe)}')
    scoped_options = descope_options(scoped_options, s0, cfg.max_scoped_options)

    nlv0 = np.array([nlv(o, s0, cache_iv0[o][1], 1, calcDate0) for o in scoped_options])
    v_delta0 = np.array([delta(o, s0, cache_iv0[o][1], 1, calcDate0) for o in scoped_options])
    m_dnlv01 = np.array([[nlv(o, s0 * ds_ret, cache_iv1[o][ds_ret], 1, calcDate1) for o in scoped_options] - nlv0 for ds_ret in cfg.v_ds_ret])
    apply_ds_ret_weights(m_dnlv01, get_f_weight_ds_ret(0.1, plot=False), cfg)

    derive_portfolio_milnp(scoped_options, m_dnlv01, v_delta0, cfg, pf=pf)


if __name__ == '__main__':
    from options.volatility.earnings_release import EarningsConfig, option_in_pf_scope, nlv, delta, get_f_weight_ds_ret, create_option_universe_iv_caches, \
    initialize_portfolio, apply_ds_ret_weights, get_scoped_expiries, descope_options

    run()
    print('Done.')
