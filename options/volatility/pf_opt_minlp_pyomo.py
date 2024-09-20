import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import cloudpickle
import psutil
import pyomo.environ as pyo
import signal

from scipy.stats import t
from multiprocessing import Process, Queue
from typing import Tuple, List
from pprint import pprint
from pyomo.environ import ConcreteModel, SolverFactory, Constraint
from options.helper import apply_ds_ret_weights, cache_to_disk
from options.typess.earnings_config import EarningsConfig
from options.typess.option import Option
from options.typess.portfolio import Portfolio
from shared.modules.logger import warning, info, error
from shared.paths import Paths


@dataclass
class PfMilnpResult:
    pf: Portfolio
    pyo_model: ConcreteModel
    pyo_inst: ConcreteModel


@cache_to_disk('derive_portfolio_milnp', Paths.path_analysis_frames)
def derive_portfolio_milnp(
        scoped_options: List[Option],
        m_dnlv01_buy: np.ndarray,
        m_dnlv01_sell: np.ndarray,
        cfg: EarningsConfig,
        pdf_t_params=(3.5983921085778388, 0.370699060063924, 17.868437182218727),
        weight_max_t_curve=5,
        f_weight_ds=None,
        pf: Portfolio = None,
        tee=True,
        weight_wing_lift: float = 0.0
) -> PfMilnpResult:
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
        cfg.n_contracts = max(cfg.n_contracts, sum([abs(h.quantity) for h in pf]))
    n_options = len(scoped_options)
    n_option_max = cfg.n_contracts // 5

    m = ConcreteModel()

    # Constants

    # Constants T curve related
    v_ds_pct = [100 * (x - 1) for x in cfg.v_ds_ret]
    y = t.pdf(v_ds_pct, *pdf_t_params)
    y_scaled = y / max(y)
    i_ds_eq_0 = list(cfg.v_ds_ret).index(1)

    # Constants position related
    ix_neg_ds = [i for i, v in enumerate(cfg.v_ds_ret) if v < 1]
    ix_pos_ds = [i for i, v in enumerate(cfg.v_ds_ret) if v > 1]

    # Parameters
    m.scoped_options = list(scoped_options)
    m.g_m_dnlv01_buy = m_dnlv01_buy[:, :n_options]
    m.g_m_dnlv01_sell = m_dnlv01_sell[:, :n_options]
    m.g_m_dnlv01_ds_weighted_buy = apply_ds_ret_weights(m.g_m_dnlv01_buy, f_weight_ds, cfg) if f_weight_ds else m.g_m_dnlv01_buy
    m.g_m_dnlv01_ds_weighted_sell = apply_ds_ret_weights(m.g_m_dnlv01_sell, f_weight_ds, cfg) if f_weight_ds else m.g_m_dnlv01_sell
    # g_v_delta0 = v_delta0[:n_options]

    # Variables
    m.set_var = pyo.Set(initialize=range(n_options))
    m.set_var_ds = pyo.Set(initialize=range(len(cfg.v_ds_ret)))

    m.v_var_p = pyo.Var(m.set_var, initialize=0, bounds=(0, cfg.n_contracts), domain=pyo.Integers)
    m.v_var_n = pyo.Var(m.set_var, initialize=0, bounds=(-cfg.n_contracts, 0), domain=pyo.Integers)
    m.v_var_abs = [m.v_var_p[i] - m.v_var_n[i] for i in range(n_options)]
    m.v_var_net = [m.v_var_p[i] + m.v_var_n[i] for i in range(n_options)]

    m.v_var_nlv_lt_t_curve = pyo.Var(m.set_var_ds, initialize=0, bounds=(None, 0))

    m.var_max_t_curve = pyo.Var(initialize=0)

    # Intermediates
    # Incorporate spread. _pos/buys, _neg/sells. Not getting filled at mid...
    m.t = [m.v_var_p @ m.g_m_dnlv01_ds_weighted_buy[i] + m.v_var_n @ m.g_m_dnlv01_ds_weighted_sell[i] for i in range(m.g_m_dnlv01_ds_weighted_buy.shape[0])]

    m.dnlv_where_ds_eq_0 = m.t[i_ds_eq_0]
    m.y_desired_dnlv = y_scaled * m.dnlv_where_ds_eq_0
    m.nlv_mn_t_curve = m.t - m.y_desired_dnlv
    if weight_wing_lift:
        info(f'Lifting wings by {100*weight_wing_lift}% of t curve at 0 ds.')
        m.nlv_mn_t_curve = m.nlv_mn_t_curve - m.t[i_ds_eq_0] * weight_wing_lift  # last term lifts the wings up. student t underestimates tails.

    # Constraints

    # Constraint on volume or total abs position
    m.cons_quantity = pyo.ConstraintList()
    m.c_max_abs_positions = Constraint(expr=sum([m.v_var_abs[i] for i in range(n_options)]) <= cfg.n_contracts)
    for i in range(n_options):
        m.cons_quantity.add(expr=m.v_var_abs[i] <= n_option_max)

    # Want to maximize max t curve, therefore need to constrain max_t_curve not to be smaller
    m.c_max_t_curve = Constraint(expr=m.dnlv_where_ds_eq_0 >= m.var_max_t_curve)

    # Shouldn't be modeled as hard constraint but rather var to optimize...
    m.cons_t_curve = pyo.ConstraintList()
    for i in range(len(m.y_desired_dnlv)):
        m.cons_t_curve.add(expr=m.nlv_mn_t_curve[i] >= m.v_var_nlv_lt_t_curve[i])

    # Upward sloping wings. Not up to optimization. A hard constraint
    m.cons_wings = pyo.ConstraintList()
    m.cons_wings.add(m.t[0] >= m.t[1])
    m.cons_wings.add(m.t[-1] >= m.t[-2])

    # Existing Portfolio
    if pf:  # If any existing portfolio is provided, add constraints
        m.cons_pf = pyo.ConstraintList()
        assert sum([abs(h.quantity) for h in pf]) <= cfg.n_contracts
        for h in pf:
            q = h.quantity
            try:
                m.scoped_options.index(h.symbol)
            except ValueError:
                raise ValueError(f'Option {h.symbol} not in scoped options.')
            i = m.scoped_options.index(h.symbol)

            # +1/-1 q to allow the algo to reduce abs position by 1 in order to adapt to changed spot prices primarily
            x = 0
            if q < 0:
                m.cons_pf.add(expr=(m.v_var_p[i] + m.v_var_n[i]) <= q + x)
            elif q > 0:
                m.cons_pf.add(expr=(m.v_var_p[i] + m.v_var_n[i]) >= q - x)

    # Objectives.
    # from pyomo.contrib.preprocessing import deactivate_trivial_constraints

    m.obj = pyo.Objective(expr=m.var_max_t_curve * weight_max_t_curve + pyo.summation(m.v_var_nlv_lt_t_curve), sense=pyo.maximize)
    # m.obj = pyo.Objective(expr=sum(m.t) + m.var_max_t_curve + pyo.summation(m.v_var_nlv_lt_t_curve), sense=pyo.maximize)
    info(f'Solving for #Options: {n_options}, #Contracts: {cfg.n_contracts}')
    # log_infeasible_constraints(m, logger=logger, log_expression=True, log_variables=True)
    solver = SolverFactory('mindtpy')

    inst = m.create_instance()
    # https://pyomo.readthedocs.io/en/stable/contributed_packages/mindtpy.html#mindtpy-implementation-and-optional-arguments
    t0 = time.time()
    models = {}

    def cb_main_solve(m):
        obj = pyo.value(m.obj)
        models[m] = obj
        info(f'Main solve Obj Value: {obj}')

    res = solver.solve(inst,
                       strategy='ECP',
                       nlp_solver='ipopt',
                       nlp_solver_args={'timelimit': 200, 'options': {'max_iter': 100}},
                       # mip_solver='glpk',
                       mip_solver='cbc',
                       mip_solver_args={'options': {'ratio': 0.005, 'sec': 200, 'threads': 8}},
                       iteration_limit=3,
                       time_limit=200,
                       tee=tee,
                       call_after_main_solve=cb_main_solve,
                       )
    info(f'Solved in {time.time() - t0}s')
    inst = max(models, key=models.get)
    # print(res)
    # inst.display()

    report_model_instance(inst, m.scoped_options, n_options, m.g_m_dnlv01_buy, m.g_m_dnlv01_sell, cfg, ix_neg_ds, ix_pos_ds)
    pf = get_instance_holdings(inst, m.scoped_options)

    return PfMilnpResult(pf, m, inst)


def task(num: int, inst: bytes, queue: Queue):
    signal.set_wakeup_fd(-1)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    pid = os.getpid()
    print(f"{num} {pid}")
    queue.put((pid, pp_solve(inst)))


def get_sub_portfolios(m: ConcreteModel, pf: Portfolio, scoped_options: List[Option], min_obj_val: float = None) -> Tuple[List[Portfolio], List[ConcreteModel]]:
    """not yet recursive..."""
    t0 = time.time()

    # with multiprocessing.Pool(8) as pool:
    #     sub_instances = pool.map(pp_solve, )

    queue = Queue()
    # Add some dict here when process was started. If gt 5min, kill it.
    processes_start_time = {}
    # need to restrict max processes: it's blocking whole trading server...
    for i, b in enumerate(get_sub_problems(m, pf, m.scoped_options)):
        p = Process(target=task, args=(i, b, queue))
        p.start()
        processes_start_time[p] = time.time()

    sub_instances = []
    while len(processes_start_time) > 0:
        if queue.qsize() > 0:
            pid, res = queue.get()
            sub_instances.append(res)
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                info(f'Terminated PID due to result in queue but p alive: {pid}')
            except psutil.NoSuchProcess:
                pass
            p = next(iter([p for p in processes_start_time.keys() if p.pid == pid]), None)
            if p and not p.is_alive():
                p.join()
                info(f'Joined PID: {pid}')
            processes_start_time = {p: ts for p, ts in processes_start_time.items() if p.pid != pid}

        remove_ps = []
        for p, start_time in processes_start_time.items():
            if not p.is_alive():
                p.join()
                info(f'Joined PID: {p.pid}')
                del processes_start_time[p]
            elif time.time() - start_time > 200:
                p.terminate()
                info(f'Terminated PID due to Timeout (200s): {p.pid}')
                remove_ps.append(p)
        for p in remove_ps:
            del processes_start_time[p]

    # sub_instances = []
    # for b in get_sub_problems(m, holdings, m.scoped_options):
    #     try:
    #         sub_instances.append(pp_solve(b))
    #     except Exception as e:
    #         error(f'Error in pp_solve: {e}')

    sub_instances = [cloudpickle.loads(b) for b in sub_instances if b]
    info(f'get_sub_portfolios: Done MP Time: {time.time() - t0}s')
    # Done in 3 min. With websockets, can explore recursively, adding more portfolios over time.

    portfolios = []
    out_instances = []
    s_holdings = set()
    for s_inst in sub_instances:
        obj_val = pyo.value(s_inst.obj)
        if min_obj_val:
            info(f'Sub obj: {pyo.value(s_inst.obj)}; % of min_obj_val: {100 * (pyo.value(s_inst.obj) / min_obj_val):.2f}%')
        if min_obj_val and obj_val < min_obj_val:
            info(f'Skipping. sub obj value too low: {pyo.value(s_inst.obj)}')
            continue
        portfolios.append(get_instance_holdings(s_inst, scoped_options))
        out_instances.append(s_inst)

        info(get_instance_holdings(s_inst, scoped_options))
        s_holdings = s_holdings.union(set(get_instance_holdings(s_inst, scoped_options).keys()))
    info(f'# Viable additional options: {len(s_holdings) - len(pf)}')
    return portfolios, out_instances


def pp_solve(b) -> bytes | None:
    try:
        inst = cloudpickle.loads(b)
        solver = SolverFactory('mindtpy')
        models = {}

        def cb_main_solve(m):
            obj = pyo.value(m.obj)
            models[m] = obj
            info(f'Main solve Obj Value: {obj}')
        solver.solve(inst,
                     strategy='ECP',
                     nlp_solver='ipopt',
                     nlp_solver_args={'timelimit': 100, 'options': {'max_iter': 100}},
                     mip_solver='cbc',
                     mip_solver_args={'options': {'ratio': 0.01, 'sec': 100, 'threads': 8}},
                     iteration_limit=3,
                     tee=False,
                     call_after_main_solve=cb_main_solve,
                     )
        inst = max(models, key=models.get)
        return cloudpickle.dumps(inst)
    except Exception as e:
        error(f'Error in pp_solve: {e}')
        return None


def get_sub_problems(m: ConcreteModel, pf: Portfolio, scoped_options: List[Option]) -> bytes:
    for sec in pf.holdings.keys():
        if not isinstance(sec, Option):
            continue
        try:
            m.del_component('c4')
        except Exception as e:
            warning(e)
            pass
        inst = m.create_instance()
        inst.c4 = Constraint(expr=inst.v_var_abs[scoped_options.index(sec)] == 0)
        yield cloudpickle.dumps(inst)


def serialize_instance(inst):
    with open('D:\\inst.pkl', mode='wb') as file:
        cloudpickle.dump(inst, file)
    # with open('D:\\inst.pkl', mode='rb') as file:
    #     inst = cloudpickle.load(file)
    return inst


def get_obj_value_from_holdings():
    """m.obj = pyo.Objective(expr=m.var_max_t_curve * weight_max_t_curve + pyo.summation(m.v_var_nlv_lt_t_curve), sense=pyo.maximize)"""


def report_model_instance(inst, g_scoped_options: List[Option], n_options, g_m_dnlv01_buy, g_m_dnlv01_sell, cfg, ix_neg_ds, ix_pos_ds):
    vars_p = [inst.v_var_p[i].value for i in range(n_options)]
    vars_n = [inst.v_var_n[i].value for i in range(n_options)]
    vars = [inst.v_var_p[i].value + inst.v_var_n[i].value for i in range(n_options)]
    print(f'Weighted Objective: {pyo.value(inst.obj)}\n'
          f'Sum of m.t / dNLV: {sum([pyo.value(inst.t[i]) for i in range(len(inst.t))])}\n'
          f'NLV at ds 0%: {pyo.value(inst.dnlv_where_ds_eq_0)}, Variable: {inst.var_max_t_curve.value}\n'
          f'desired curve diff sum: {sum([inst.v_var_nlv_lt_t_curve[i].value for i in range(len(inst.v_var_nlv_lt_t_curve))])}'
    )
    # print(f'y_scaled: {y_scaled}')
    # print(f'm.y_desired_dnlv: {[pyo.value(v) for v in inst.y_desired_dnlv]}')
    # print(f'desired curve: {[pyo.value(v) for v in inst.t]}\n\n')
    # print(f'diff to curve: {[pyo.value(v) for v in inst.nlv_mn_t_curve]}\n\n')
    # print(f'desired curve var: {[inst.v_var_nlv_lt_t_curve[i].value for i in range(len(inst.v_var_nlv_lt_t_curve))]}')

    nlv_by_ds = np.sum(np.array(vars_p) * g_m_dnlv01_buy, axis=1) + np.sum(np.array(vars_n) * g_m_dnlv01_sell, axis=1)
    sol_dnlv_total = np.sum(nlv_by_ds)
    print(f'dNLV Total dS weighted: {sol_dnlv_total}')

    ls_diff = sum(nlv_by_ds[ix_neg_ds]) - sum(nlv_by_ds[ix_pos_ds])
    print(f'Weighted LS Diff: {ls_diff}')

    dnlv_by_ds = np.sum(np.array(vars_p) * g_m_dnlv01_buy, axis=1) + np.sum(np.array(vars_n) * g_m_dnlv01_sell, axis=1)
    print(f'Weighted dLNLV by ds: \n{pd.Series(dnlv_by_ds, index=cfg.v_ds_ret)}')

    # sol_delta_total = np.array(vars) @ g_v_delta0
    # print(f'Solution: Delta total: {sol_delta_total}')

    holdings = {g_scoped_options[i]: v for i, v in enumerate(vars) if v != 0}
    pprint(holdings)


def get_instance_holdings(inst, scoped_options: List[Option]) -> Portfolio:
    vars = [inst.v_var_p[i].value + inst.v_var_n[i].value for i in range(len(inst.v_var_p))]
    return Portfolio({scoped_options[i]: v for i, v in enumerate(vars) if v != 0})


if __name__ == '__main__':
    """
    Find all possible portfolios within 90% of the best option portfolio containing 20 options.
    Simplification: Don't vary the quantities of an option. Just iteratively kicking an option out, until objective goes < 90% of best.
    Count portfolios?
    Count of options to buy vs best pf?
    """
    print('Done.')
