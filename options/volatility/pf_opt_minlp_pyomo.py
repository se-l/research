import os
import time
import numpy as np
import pandas as pd
import cloudpickle
import psutil
import pyomo.environ as pyo
import signal

from multiprocessing import Process, Queue
from typing import Dict, Tuple, List
from pprint import pprint
from pyomo.environ import ConcreteModel, SolverFactory, Constraint
from datetime import timedelta

from options.helper import year_quarter, enrich_atm_iv, val_from_df, apply_ds_ret_weights
from options.typess.equity import Equity
from options.typess.option import Option
from options.typess.option_frame import OptionFrame
from options.typess.portfolio import Portfolio
from options.volatility.estimators.earnings_iv_drop_regressor import EarningsIVDropRegressor
from shared.constants import EarningsPreSessionDates
from shared.modules.logger import logger
from shared.paths import Paths


def derive_portfolio_milnp(
        scoped_options, m_dnlv01, v_delta0, cfg,
        n_contracts=20,
        risk_adj_power=1,
        weight_max_min_nlv=25,
        f_weight_ds=None,
        pf=None,
        tee=True
) -> Tuple[Portfolio, ConcreteModel, ConcreteModel]:
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

    m = ConcreteModel()

    # Constants
    ix_neg_ds = [i for i, v in enumerate(cfg.v_ds_ret) if v < 1]
    ix_pos_ds = [i for i, v in enumerate(cfg.v_ds_ret) if v > 1]

    # Parameters
    m.scoped_options_str = [str(o) for o in scoped_options]
    m.g_m_dnlv01 = m_dnlv01[:, :n_options]
    m.g_m_dnlv01_ds_weighted = apply_ds_ret_weights(m.g_m_dnlv01, f_weight_ds, cfg)
    g_v_delta0 = v_delta0[:n_options]

    # Variables
    m.set_var = pyo.Set(initialize=range(n_options))

    m.v_var_p = pyo.Var(m.set_var, initialize=0, bounds=(0, n_contracts), domain=pyo.IntegerSet)
    m.v_var_n = pyo.Var(m.set_var, initialize=0, bounds=(-n_contracts, 0), domain=pyo.IntegerSet)
    m.v_var_abs = [m.v_var_p[i] - m.v_var_n[i] for i in range(n_options)]
    m.v_var_net = [m.v_var_p[i] + m.v_var_n[i] for i in range(n_options)]

    m.mn = pyo.Var(initialize=0)
    m.var_lr_diff = pyo.Var(initialize=0, bounds=(0, None))
    m.var_lr_diff_bin = pyo.Var(initialize=0, domain=pyo.Binary)

    # Intermediates
    m.t = [m.v_var_net @ m.g_m_dnlv01_ds_weighted[i] for i in range(m.g_m_dnlv01_ds_weighted.shape[0])]

    # They are ds weighted
    m.neg_ds = sum([m.t[i] for i in ix_neg_ds])
    m.pos_ds = sum([m.t[i] for i in ix_pos_ds])
    m.lr_diff = m.neg_ds - m.pos_ds
    m.lr_diff_sign = 2 * m.var_lr_diff_bin - 1

    # Constraints
    m.c_max_abs_positions = Constraint(expr=sum([m.v_var_abs[i] for i in range(n_options)]) <= n_contracts)

    m.cons_max_loss = pyo.ConstraintList()
    for i in range(m.g_m_dnlv01_ds_weighted.shape[0]):
        m.cons_max_loss.add(expr=m.mn <= m.t[i])

    m.cons_risk_balance = pyo.ConstraintList()
    m.cons_risk_balance.add(expr=m.lr_diff * m.lr_diff_sign >= 1)
    m.cons_risk_balance.add(expr=m.lr_diff * m.lr_diff_sign <= m.var_lr_diff)

    if pf:  # If any existing portfolio is provided, add constraints
        m.cons_pf = pyo.ConstraintList()
        assert sum([abs(q) for q in pf.holdings.values()]) <= n_contracts
        for sec, q in pf.holdings.items():
            try:
                print(m.scoped_options_str.index(str(sec)))
            except ValueError:
                raise ValueError(f'Option {sec} not in scoped options.')
            i = m.scoped_options_str.index(str(sec))
            if q < 0:
                m.cons_pf.add(expr=(m.v_var_p[i] + m.v_var_n[i]) <= q)
            elif q > 0:
                m.cons_pf.add(expr=(m.v_var_p[i] + m.v_var_n[i]) >= q)

    m.cons_exclude = pyo.ConstraintList()

    # Objectives.
    # should make the weight dS dependent? and remove dS weighting from the input matrix?
    # Requires a little more refactoring. Would slow down, means 1 more vector sum.
    # Pass unweighted to this function, Weigh once for whoever needs it once. In report, contrast weighted and unweighted.

    m.obj = pyo.Objective(expr=sum(m.t) + m.mn * weight_max_min_nlv - m.var_lr_diff, sense=pyo.maximize)

    print(f'Solving for #Options: {n_options}, #Contracts: {n_contracts}')
    solver = SolverFactory('mindtpy')
    obj_val_max = -np.inf

    # Now keep on solving until all paths have been explored and performance is < 90% of best
    # Need starting instance to create paths, then some recursive function to create new instances
    # Each solver needs to run on its own core. Cannot pickle, not pyomo nor ql objects - would need to re-init whole matrix etc, then just past constraints....
    # like the str name of an option... Would certainly speed up stuff. returns some performance metrics, like the pyomo json result.
    # For a rester, each processes caches. Waiting 5 mins for solve to come back is too long to work on BTs...
    # pre-calc or cache on disk

    inst = m.create_instance()
    # https://pyomo.readthedocs.io/en/stable/contributed_packages/mindtpy.html#mindtpy-implementation-and-optional-arguments
    res = solver.solve(inst,
                       nlp_solver='ipopt',
                       # nlp_solver_args={'timelimit': 650, 'options': {'max_iter': 100}},
                       # mip_solver='glpk',
                       mip_solver='cbc',
                       mip_solver_args={'options': {'ratio': 0.005, 'sec': 100}},
                       iteration_limit=10,
                       tee=tee,
                       )
    # print(res)
    obj_val = pyo.value(inst.obj)
    obj_val_max = max(obj_val_max, obj_val)
    # inst.display()

    report_model_instance(inst, m.scoped_options_str, n_options, m.g_m_dnlv01, g_v_delta0, cfg, ix_neg_ds, ix_pos_ds, risk_adj_power=risk_adj_power)
    holdings = get_instance_holdings(inst, scoped_options)

    return Portfolio(holdings), m, inst


def task(num: int, inst: bytes, queue: Queue):
    signal.set_wakeup_fd(-1)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    pid = os.getpid()
    print(f"{num} {pid}")
    queue.put((pid, pp_solve(inst)))


def get_sub_portfolios(m: ConcreteModel, holdings: Dict[Option, int], scoped_options, min_obj_val: float = None) -> Tuple[List[Portfolio], List[ConcreteModel]]:
    """not yet recursive..."""
    t0 = time.time()
    # with multiprocessing.Pool(8) as pool:
    #     sub_instances = pool.map(pp_solve, )

    queue = Queue()
    processes = []
    for i, b in enumerate(get_sub_problems(m, holdings, m.scoped_options_str)):
        p = Process(target=task, args=(i, b, queue))
        p.start()
        processes.append(p)

    sub_instances = []
    while len(processes) > 0:
        if queue.qsize() > 0:
            pid, res = queue.get()
            sub_instances.append(res)
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                logger.info(f'Terminated PID: {pid}')
            except psutil.NoSuchProcess:
                pass
            p = next(iter([p for p in processes if p.pid == pid]), None)
            if p and not p.is_alive():
                p.join()
                logger.info(f'Joined PID: {pid}')
            processes = [p for p in processes if p.pid != pid]
        for p in processes:
            if not p.is_alive():
                p.join()
                logger.info(f'Joined PID: {p.pid}')
                processes.remove(p)

    # sub_instances = []
    # for b in get_sub_problems(m, holdings, m.scoped_options_str):
    #     try:
    #         sub_instances.append(pp_solve(b))
    #     except Exception as e:
    #         logger.error(f'Error in pp_solve: {e}')

    sub_instances = [cloudpickle.loads(b) for b in sub_instances if b]
    logger.info(f'get_sub_portfolios: Done MP Time: {time.time() - t0}s')
    # Done in 3 min. With websockets, can explore recursively, adding more portfolios over time.

    portfolios = []
    s_holdings = set()
    for s_inst in sub_instances:
        obj_val = pyo.value(s_inst.obj)
        if min_obj_val:
            logger.info(f'Sub obj: {pyo.value(s_inst.obj)}; % of min_obj_val: {100 * (pyo.value(s_inst.obj) / min_obj_val):.2f}%')
        if min_obj_val and obj_val < min_obj_val:
            logger.info(f'Skipping. sub obj value too low: {pyo.value(s_inst.obj)}')
            continue
        portfolios.append(Portfolio(get_instance_holdings(s_inst, scoped_options)))

        logger.info(get_instance_holdings(s_inst, scoped_options))
        s_holdings = s_holdings.union(set(get_instance_holdings(s_inst, scoped_options).keys()))
    logger.info(f'# Viable additional options: {len(s_holdings) - len(holdings)}')
    return portfolios, sub_instances


def pp_solve(b) -> bytes | None:
    try:
        inst = cloudpickle.loads(b)
        solver = SolverFactory('mindtpy')
        solver.solve(inst,
                     nlp_solver='ipopt',  # nlp_solver_args={'timelimit': 650, 'options': {'max_iter': 100}},
                     mip_solver='cbc',
                     mip_solver_args={'options': {'ratio': 0.005, 'sec': 100}},
                     iteration_limit=10,
                     tee=True,
                     )
        return cloudpickle.dumps(inst)
    except Exception as e:
        logger.error(f'Error in pp_solve: {e}')
        return None


def get_sub_problems(m, holdings, scoped_options_str: List[str]) -> bytes:
    inst = m.create_instance()
    for o in holdings.keys():
        inst.c4 = Constraint(expr=inst.v_var_abs[scoped_options_str.index(str(o))] == 0)
        yield cloudpickle.dumps(inst)


def serialize_instance(inst):
    with open('D:\\inst.pkl', mode='wb') as file:
        cloudpickle.dump(inst, file)
    # with open('D:\\inst.pkl', mode='rb') as file:
    #     inst = cloudpickle.load(file)
    return inst


def report_model_instance(inst, g_scoped_options, n_options, g_m_dnlv01, g_v_delta0, cfg, ix_neg_ds, ix_pos_ds, risk_adj_power=1):
    vars = [inst.v_var_p[i].value + inst.v_var_n[i].value for i in range(n_options)]

    neg_ds = pyo.value(inst.neg_ds)
    pos_ds = pyo.value(inst.pos_ds)
    print(f'var_lr_diff: {pyo.value(inst.var_lr_diff)}; lr_diff: {pyo.value(inst.lr_diff)}; lr_diff_sign: {pyo.value(inst.lr_diff_sign)}; <1 ds: {neg_ds}; ds>1: {pos_ds}; n-p: {neg_ds-pos_ds}')

    nlv_by_ds = np.sum(np.array(vars) * g_m_dnlv01, axis=1)
    sol_dnlv_total = np.sum(nlv_by_ds)
    print(f'dNLV Total dS weighted: {sol_dnlv_total}')

    ls_diff = sum(nlv_by_ds[ix_neg_ds]) - sum(nlv_by_ds[ix_pos_ds])
    print(f'LS Diff: {ls_diff}')

    dnlv_by_ds = np.sum(np.array(vars) * g_m_dnlv01, axis=1)
    print(f'dLNLV by ds: \n{pd.Series(dnlv_by_ds, index=cfg.v_ds_ret)}')

    sol_delta_total = np.array(vars) @ g_v_delta0
    print(f'Solution: Delta total: {sol_delta_total}, Weighted: {abs(sol_delta_total) ** risk_adj_power}')

    holdings = {g_scoped_options[i]: v for i, v in enumerate(vars) if v != 0}
    pprint(holdings)


def get_instance_holdings(inst, scoped_options: List[Option]) -> Dict[Option, float]:
    vars = [inst.v_var_p[i].value + inst.v_var_n[i].value for i in range(len(inst.v_var_p))]
    holdings = {scoped_options[i]: v for i, v in enumerate(vars) if v != 0}
    return holdings


# Quick Local Testing
def run():
    from options.volatility.earnings_release import EarningsConfig, option_in_pf_scope, nlv, delta, get_f_weight_ds_ret, create_option_universe_iv_caches, \
        initialize_portfolio, get_scoped_expiries
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

    option_universe, cache_iv0, cache_iv1 = create_option_universe_iv_caches(sym, ts_pre_release, df0, s0, cfg.v_ds_ret, cfg.use_skew)
    expiries = list(sorted(set(df0.index.get_level_values('expiry').values)))
    exp_jump1 = sorted([i for i in expiries if i >= release_date])[0]

    pf = initialize_portfolio(df0, ts_pre_release, sym, exp_jump1, s0, cfg, option_universe)

    scoped_expiries = get_scoped_expiries(expiries)
    get_p0 = lambda o: val_from_df(df0.loc[ts_pre_release], o.expiry, o.optionContract.strike, o.right, 'mid_price')
    scoped_options = [o for o in option_universe.values() if option_in_pf_scope(o, cfg.min_tenor, cfg.moneyness_limits, calcDate0, s0, get_p0(o), cfg.abs_delta_limits, scoped_expiries=scoped_expiries, pf=pf)]
    print(f'Scoped options: #{len(scoped_options)} out of # {len(option_universe)}')
    # scoped_options = descope_options(scoped_options, s0, cfg.max_scoped_options)

    nlv0 = np.array([nlv(o, s0, cache_iv0[o][1], 1, calcDate0) for o in scoped_options])
    v_delta0 = np.array([delta(o, s0, cache_iv0[o][1], 1, calcDate0) for o in scoped_options])
    m_dnlv01 = np.array([[nlv(o, s0 * ds_ret, cache_iv1[o][ds_ret], 1, calcDate1) for o in scoped_options] - nlv0 for ds_ret in cfg.v_ds_ret])
    f_weight_ds = get_f_weight_ds_ret(0.1, plot=False)

    pf_target, m, inst = derive_portfolio_milnp(scoped_options, m_dnlv01, v_delta0, cfg, f_weight_ds=f_weight_ds, pf=pf)


if __name__ == '__main__':
    run()
    """
    Find all possible portfolios within 90% of the best option portfolio containing 20 options.
    Simplification: Don't vary the quantities of an option. Just iteratively kicking an option out, until objective goes < 90% of best.
    Count portfolios?
    Count of options to buy vs best pf?
    """
    print('Done.')
