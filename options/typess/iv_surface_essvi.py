import time
from dataclasses import dataclass
from itertools import chain

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from datetime import date, timedelta
from typing import Dict, List, Tuple, Callable
from plotly.subplots import make_subplots
from pyomo.core import ConcreteModel
from pyomo.opt import SolverFactory
from pyomo.util.infeasible import log_infeasible_constraints

from options.helper import get_tenor
from options.typess.calibration_item import CalibrationItem, df2calibration_items
from options.typess.option import Option
from shared.constants import EarningsPreSessionDates
from options.typess.enums import Resolution, OptionRight
from options.typess.equity import Equity
from scipy.optimize import least_squares
from scipy.stats import norm

from shared.modules.logger import warning, info, logger
from shared.plotting import show, plot_scatter_3d
import pyomo.environ as pyo


class Params:
    tenor_dt: date
    right: str | OptionRight
    theta: float
    rho: float
    psi: float


@dataclass
class SliceParams:
    right: str | OptionRight
    tenor_dt: date
    theta: float
    rho: float
    psi: float


@dataclass
class CalibrationArgs:
    right: str | OptionRight
    fills: Dict[date, SliceParams]
    bids: Dict[date, SliceParams]
    asks: Dict[date, SliceParams]
    # f_weight: f(log_fwd_mny, tenor)


@dataclass
class CalibrationResult:
    right: str | OptionRight
    surface_params: Dict[date, SliceParams]
    model: ConcreteModel


def f_essvi_total_variance(k, theta, rho, psi):
    """
    Implied total variance surface
    eSSVI(K, T) = 1/2 ( θ(T) + ρ(T)ψ(T)k + sqrt((ψ(T)k + θ(T)ρ(T))**2 + θ(T)**2 * (1 − ρ(T)**2))
    """
    return 0.5 * (theta + rho * psi * k + ((psi * k + theta * rho) ** 2 + theta ** 2 * (1 - rho ** 2)) ** (1 / 2))


def f_essvi_iv(k, theta, rho, psi, tenor=None):
    """
    Implied total variance surface
    eSSVI(K, T) = 1/2 ( θ(T) + ρ(T)ψ(T)k + sqrt((ψ(T)k + θ(T)ρ(T))**2 + θ(T)**2 * (1 − ρ(T)**2))
    """
    return (f_essvi_total_variance(k, theta, rho, psi) / tenor) ** (1/2)


def f_min(x, u, y, f=f_essvi_total_variance, weights=None):
    theta, rho, psi = x
    k = u

    residuals = f(k, theta, rho, psi) - y
    if weights is not None:
        residuals *= weights

    if np.isnan(residuals).any() or np.isinf(residuals).any():
        print('nan or inf residuals')
        ix_nan = np.isnan(residuals) | np.isinf(residuals)
        res_max = np.abs(residuals[~ix_nan])
        if len(res_max) > 0:
            residuals[ix_nan] = np.max(res_max) * 10
        else:
            residuals[ix_nan] = 1e3

    return residuals


def f_min_reparameterized(calibration_params: list, samples: Dict[date, CalibrationItem], right: str | OptionRight):

    lst_residuals = []

    for i, (tenor_dt, item) in enumerate(samples.items()):
        theta = calibration_params[i * 3]
        rho = calibration_params[i * 3 + 1]
        psi = calibration_params[i * 3 + 2]

        model_variance = f_essvi_total_variance(item.mny_fwd_ln, theta, rho, psi)
        model_iv = (model_variance / item.tenor) ** 0.5

        residuals = model_iv - item.iv
        # print(item.price[-2:])
        # print(np.array(f(item.spot[-2:], np.array(item.strike)[-2:], item.tenor, item.iv[-2:], item.rf, item.dividend_yield)))

        f: Callable = Option.price_put if right == OptionRight.put else Option.price_call
        # item_price_assert = np.array(f(item.spot, np.array(item.strike), item.tenor, item.iv, item.rf, item.dividend_yield))
        # residuals_price = item.price - item_price_assert
        # fig = make_subplots(rows=2, cols=1, subplot_titles=[f'IV {item.tenor_dt} {item.right}'])
        # fig.add_trace(go.Scatter(x=item.strike, y=item_price_assert, mode='markers', marker=dict(size=4), name='iv'), row=1, col=1)
        # fig.add_trace(go.Scatter(x=item.strike, y=item.price, mode='markers', marker=dict(size=4), name='iv'), row=1, col=1)
        # fig.add_trace(go.Scatter(x=item.strike, y=residuals_price, mode='markers', marker=dict(size=4), name='iv'), row=2, col=1)
        # show(fig, fn=f'iv_{item.tenor_dt}_{item.right}.html', open_browser=True)

        # model_prices = np.array(f(item.spot, np.array(item.strike), item.tenor, model_iv, item.rf, item.dividend_yield))
        # residuals = model_prices - item.price

        if item.weights is not None:
            residuals *= item.weights

        if np.isnan(residuals).any() or np.isinf(residuals).any():
            print('nan or inf residuals')
            ix_nan = np.isnan(residuals) | np.isinf(residuals)
            res_max = np.abs(residuals[~ix_nan])
            if len(res_max) > 0:
                residuals[ix_nan] = np.max(res_max) * 10
            else:
                residuals[ix_nan] = 1e3

        lst_residuals += list(residuals)

    return lst_residuals


class IVSurface:
    """
    Enrich a frame with index ts, expiry, strike, right
    So plotting is easy
    """
    def __init__(self, underlying: Equity):
        self.underlying = underlying
        self.is_calibrated = False
        self.calibration_items: Dict[Tuple[date, str | OptionRight], CalibrationItem] = {}
        self.params: Dict[Tuple[date, str | OptionRight], Tuple] = {}
        self.rmse: Dict[Tuple[date, str | OptionRight], Tuple] = {}

    def calibrate(self, calibration_items: List[CalibrationItem], verbose=1, initial_params=None, plot_cost_gt=10):
        initial_params = initial_params or [0.2, -0.1, 2]

        for item in calibration_items:

            total_variance = item.iv ** 2 * item.tenor
            fit_res = least_squares(f_min, initial_params, args=(item.mny_fwd_ln, total_variance, f_essvi_total_variance, item.weights), verbose=verbose, max_nfev=1000)

            self.calibration_items[item.tenor_dt, item.right] = item
            self.params[item.tenor_dt, item.right] = fit_res.x
            self.rmse[item.tenor_dt, item.right] = fit_res.cost

            if fit_res.cost > plot_cost_gt:
                warning(f'Calibration failed for {self.underlying} tenor={item.tenor_dt} {item.right} cost: {fit_res.cost}, calcDate={item.calculation_date}')
                self.plot_smile(item.tenor_dt, item.right, item.calculation_date)

        info(f'Calibration done for {self.underlying} total_rmse={sum(self.rmse.values())}')
        self.is_calibrated = True

        return self

    def ps_rmse(self) -> pd.Series:
        return pd.Series(self.rmse)

    def plot_all_smiles(self, calc_date: date, calibration_items_bid=None, calibration_items_ask=None, fn=None, open_browser=True):
        for tenor_dt, right in self.calibration_items.keys():
            self.plot_smile(tenor_dt, right, calc_date, calibration_items_bid, calibration_items_ask, fn=fn, open_browser=open_browser)

    def plot_smile(self, tenor_dt: date, right: OptionRight | str, calc_date: date, calibration_items_bid=None, calibration_items_ask=None, fn=None, open_browser=True):
        params = self.params[tenor_dt, right]
        v_mny_fwd_ln = self.calibration_items[tenor_dt, right].mny_fwd_ln
        v_iv = self.calibration_items[tenor_dt, right].iv
        v_vega = self.calibration_items[tenor_dt, right].vega
        v_weights = self.calibration_items[tenor_dt, right].weights
        y_pred = f_essvi_iv(v_mny_fwd_ln, *params, tenor=get_tenor(tenor_dt, calc_date))

        fig = make_subplots(rows=3, cols=1, subplot_titles=[f'IV {self.underlying} {right} {tenor_dt}', 'Weights', 'Vega & Count Hist'])
        fig.add_trace(go.Scatter(x=v_mny_fwd_ln, y=v_iv, mode='markers', marker=dict(size=4), name='iv'), row=1, col=1)

        # Quotes if any
        if calibration_items_bid is not None and (i_bid := next(iter([i for i in calibration_items_bid if i.tenor_dt == tenor_dt and i.right == right]), None)):
            ix_sample = pd.Series(range(len(i_bid.iv))).sample(min(len(i_bid.iv), 500)).values
            x = i_bid.mny_fwd_ln[ix_sample]
            fig.add_trace(go.Scatter(x=x, y=i_bid.iv[ix_sample], mode='markers', marker=dict(size=2), name='iv_bid'), row=1, col=1)
            if calibration_items_ask is not None and (i_ask := next(iter([i for i in calibration_items_ask if i.tenor_dt == tenor_dt and i.right == right]), None)):
                fig.add_trace(go.Scatter(x=x, y=i_ask.iv[ix_sample], mode='markers', marker=dict(size=2), name='iv_ask'), row=1, col=1)

        fig.add_trace(go.Scatter(x=v_mny_fwd_ln, y=y_pred, mode='markers', marker=dict(size=4), name='iv_pred'), row=1, col=1)
        fig.add_trace(go.Scatter(x=v_mny_fwd_ln, y=v_weights, mode='markers', marker=dict(size=4), name='weights'), row=2, col=1)
        if v_vega is not None:
            fig.add_trace(go.Scatter(x=v_mny_fwd_ln, y=v_vega, mode='markers', marker=dict(size=4), name='vega'), row=3, col=1)
        fig.add_trace(go.Histogram(x=v_mny_fwd_ln, nbinsx=int((max(v_mny_fwd_ln) - min(v_mny_fwd_ln)) * 100), name='count'), row=3, col=1)
        show(fig, fn=fn or f'essvi_{self.underlying}_{tenor_dt}_{right}.html', open_browser=open_browser)

    def plot_all_prices(self, calc_date: date, calibration_items_bid=None, calibration_items_ask=None, fn=None, open_browser=True):
        for tenor_dt, right in self.calibration_items.keys():
            self.plot_prices(tenor_dt, right, calc_date, calibration_items_bid, calibration_items_ask, fn=fn, open_browser=open_browser)

    def plot_prices(self, tenor_dt: date, right: OptionRight | str, calc_date: date, calibration_items_bid=None, calibration_items_ask=None, fn=None, open_browser=True):
        rf = self.calibration_items[tenor_dt, right].rf
        dividend_yield = self.calibration_items[tenor_dt, right].dividend_yield

        params = self.params[tenor_dt, right]
        item = self.calibration_items[tenor_dt, right]
        v_mny_fwd_ln = item.mny_fwd_ln
        v_strike = item.strike
        tenor = get_tenor(tenor_dt, calc_date)

        v_spot = item.spot
        v_vega = item.vega
        v_weights = item.weights
        y_pred = f_essvi_iv(v_mny_fwd_ln, *params, tenor=get_tenor(tenor_dt, calc_date))

        fig = make_subplots(rows=3, cols=1, subplot_titles=[f'Prices {self.underlying} {right} {tenor_dt}', 'Weights', 'Vega & Count Hist'])
        f: Callable = Option.price_put if right == OptionRight.put else Option.price_call
        fig.add_trace(go.Scatter(x=v_mny_fwd_ln, y=item.price, mode='markers', marker=dict(size=4), name='price'), row=1, col=1)

        # Quotes if any
        if calibration_items_bid is not None and (i_bid := next(iter([i for i in calibration_items_bid if i.tenor_dt == tenor_dt and i.right == right]), None)):
            ix_sample = pd.Series(range(len(i_bid.iv))).sample(min(len(i_bid.iv), 500)).values
            x = i_bid.mny_fwd_ln[ix_sample]

            fig.add_trace(go.Scatter(x=x, y=i_bid.price[ix_sample], mode='markers', marker=dict(size=2), name='price_bid'), row=1, col=1)
            if calibration_items_ask is not None and (i_ask := next(iter([i for i in calibration_items_ask if i.tenor_dt == tenor_dt and i.right == right]), None)):
                fig.add_trace(go.Scatter(x=x, y=i_ask.price[ix_sample], mode='markers', marker=dict(size=2), name='price_ask'), row=1, col=1)

        y_price_pred = np.array(f(v_spot, np.array(v_strike), tenor, y_pred, rf, dividend_yield))
        fig.add_trace(go.Scatter(x=v_mny_fwd_ln, y=y_price_pred, mode='markers', marker=dict(size=4), name='price_pred'), row=1, col=1)
        fig.add_trace(go.Scatter(x=v_mny_fwd_ln, y=v_weights, mode='markers', marker=dict(size=4), name='weights'), row=2, col=1)
        if v_vega is not None:
            fig.add_trace(go.Scatter(x=v_mny_fwd_ln, y=v_vega, mode='markers', marker=dict(size=4), name='vega'), row=3, col=1)
        fig.add_trace(go.Histogram(x=v_mny_fwd_ln, nbinsx=int((max(v_mny_fwd_ln) - min(v_mny_fwd_ln)) * 100), name='count'), row=3, col=1)
        show(fig, fn=fn or f'essvi_{self.underlying}_{tenor_dt}_{right}.html', open_browser=open_browser)

    def plot_surface(self, fn=None, open_browser=True):
        fig = go.Figure()
        for right_trace in [OptionRight.call, OptionRight.put]:
            v_mny = []
            v_tenor = []
            v_iv = []
            for k, v in self.calibration_items.items():
                tenor_dt, right = k
                if right != right_trace:
                    continue
                tenor = get_tenor(tenor_dt, v.calculation_date)
                v_mny += list(v.mny_fwd_ln)
                v_tenor += [tenor] * len(v.mny_fwd_ln)
                params = self.params[tenor_dt, right]
                v_iv += list(f_essvi_iv(v.mny_fwd_ln, *params, tenor=tenor))

            fig.add_trace(go.Scatter3d(x=v_mny, y=v_tenor, z=v_iv, mode='markers', marker=dict(size=2), name=f'{right_trace} IV'))

        fn_plot = fn or f'essvi_{self.underlying}_IV_surface.html'
        fig.update_layout(title=fn_plot, autosize=True, scene=dict(
            xaxis_title='Log forward moneyness',
            yaxis_title='Tenor',
            zaxis_title='Implied Volatility',),
        )
        show(fig, fn=fn_plot, open_browser=open_browser)

    def plot_params_rmse(self, calc_date: date, fn=None, open_browser=True):
        """Plot params and errors for each right across the surface."""
        rights = [OptionRight.call, OptionRight.put]
        fig = make_subplots(rows=4, cols=2, subplot_titles=list(chain(*[[f'{right} {metric}' for right in rights] for metric in ['Theta', 'Rho', 'Psi', 'RMSE']])))
        for col, right in enumerate(rights):
            p_dct = {k: v for k, v in self.params.items() if k[1] == right}
            x = [get_tenor(k[0], calc_date) for k in p_dct.keys()]
            theta = [v[0] for k, v in p_dct.items()]
            rho = [v[1] for k, v in p_dct.items()]
            psi = [v[2] for k, v in p_dct.items()]
            fig.add_trace(go.Scatter(x=x, y=theta, mode='markers', marker=dict(size=4), name=f'{right} Theta'), row=1, col=col + 1)
            fig.add_trace(go.Scatter(x=x, y=rho, mode='markers', marker=dict(size=4), name=f'{right} Rho'), row=2, col=col + 1)
            fig.add_trace(go.Scatter(x=x, y=psi, mode='markers', marker=dict(size=4), name=f'{right} Psi'), row=3, col=col + 1)

            p_dct = {k: v for k, v in self.rmse.items() if k[1] == right}
            rmse = [v for k, v in p_dct.items()]
            fig.add_trace(go.Scatter(x=x, y=rmse, mode='markers', marker=dict(size=4), name=f'{right} RMSE'), row=4, col=col + 1)
        show(fig, fn=fn or f'essvi_{self.underlying}_params_rmse.html', open_browser=open_browser)

    def iv(self, moneyness_fwd_ln: float, tenor: date, right: OptionRight, calc_date: date) -> float:
        """
        Interpolating IV from the frame.
        """
        if not self.is_calibrated:
            raise ValueError('IVSurface is not calibrated')
        return f_essvi_iv(moneyness_fwd_ln, *self.params[tenor, right], tenor=get_tenor(tenor, calc_date))

    def calibrate_surface(self, right: str | OptionRight, samples: Dict[date, CalibrationItem], verbose=True):
        tenors = sorted(samples.keys())
        n_tenors = len(tenors)

        # p, a, c
        calibration_params = [0, 0.2, 0] * n_tenors

        # Theta, Rho, Psi
        model_params = np.empty((n_tenors, 3))
        model_params[:, 0] = 0.2
        model_params[:, 1] = -0.1
        model_params[:, 2] = 2
        # Theta 0
        model_params[0, 0] = 0

        fit_res = least_squares(f_min_reparameterized, calibration_params, args=(samples, right), verbose=verbose, max_nfev=1000)

        for i, (tenor_dt, item) in enumerate(samples.items()):
            theta = fit_res.x[i * 3]
            rho = fit_res.x[i * 3 + 1]
            psi = fit_res.x[i * 3 + 2]

            self.calibration_items[item.tenor_dt, item.right] = item
            self.params[item.tenor_dt, item.right] = (theta, rho, psi)
            # self.rmse[item.tenor_dt, item.right] = fit_res.cost

        # if fit_res.cost > plot_cost_gt:
        #     warning(f'Calibration failed for {self.underlying} tenor={item.tenor_dt} {item.right} cost: {fit_res.cost}, calcDate={item.calculation_date}')
        #     self.plot_smile(item.tenor_dt, item.right, item.calculation_date)

        info(f'Calibration done for {self.underlying} total_rmse={sum(self.rmse.values())}')
        self.is_calibrated = True

        return self

    def calibrate_surface_pyomo(
            self,
            right: str | OptionRight,
            fills: Dict[date, CalibrationItem],
            quotes: Dict[date, CalibrationItem],
            ) -> CalibrationResult:
        """
        Issue: Minimize RMSE for prices requires turning a model implied vol into a price. That requires the CFF, pyomo doesn't support that out of the box.
        """
        m = ConcreteModel()
        get_price: Callable = Option.price_put if right == OptionRight.put else Option.price_call

        tenors = sorted(fills.keys())
        n_tenors = len(tenors)

        # Parameters - Fills, quotes, weights
        m.tenors_flt = [get_tenor(tenor, fills[tenor].calculation_date) for tenor in tenors]

        # Variables
        m.set_var = pyo.Set(initialize=range(n_tenors))
        m.set_rho = pyo.Set(initialize=range(n_tenors))
        m.set_psi = pyo.Set(initialize=range(n_tenors))

        m.v_theta = pyo.Var(m.set_var, initialize=0.1)
        m.v_rho = pyo.Var(m.set_var, initialize=-0.1)
        m.v_psi = pyo.Var(m.set_var, initialize=0.1)

        # Intermediates
        m.price_fills_ssvi = {}
        for i, (tenor_dt, slice) in enumerate(fills.items()):
            tenor = get_tenor(tenor_dt, slice.calculation_date)
            iv = f_essvi_iv(slice.mny_fwd_ln, m.v_theta[i], m.v_rho[i], m.v_psi[i], tenor=tenor)
            m.price_fills_ssvi[tenor_dt] = get_price(slice.spot, slice.strike, tenor, iv, slice.net_yield)
        if bids:
            m.price_bids_model = {}
            for i, (tenor_dt, slice) in enumerate(bids.items()):
                tenor = get_tenor(tenor_dt, slice.calculation_date)
                iv = f_essvi_iv(slice.mny_fwd_ln, m.v_theta[i], m.v_rho[i], m.v_psi[i], tenor=tenor)
                m.price_bids_model[tenor_dt] = get_price(slice.spot, slice.strike, tenor, iv, slice.net_yield)
        if asks:
            m.price_asks_model = {}
            for i, (tenor_dt, slice) in enumerate(asks.items()):
                tenor = get_tenor(tenor_dt, slice.calculation_date)
                iv = f_essvi_iv(slice.mny_fwd_ln, m.v_theta[i], m.v_rho[i], m.v_psi[i], tenor)
                m.price_asks_model[tenor_dt] = get_price(slice.spot, slice.strike, tenor, iv, slice.net_yield)

        m.sqrt_right_term = {tenor_dt: (m.v_psi[i] * m.tenors_flt[i] + m.v_theta[i] * m.v_rho[i]) ** 2 + m.v_theta[i] ** 2 * (1 - m.v_rho[i] ** 2) for i, tenor_dt in enumerate(tenors)}

        m.rmse_fills = [(sum((m.price_fills_model[tenor] - fills[tenor].price)**2))**0.5 for tenor in tenors]
        if quotes:
            m.rmse_quotes = [(sum((m.price_quotes_model[tenor] - bids[tenor].mid_price)**2))**0.5 for tenor in tenors]

        # Constrains no nan
        m.cons_no_nan = pyo.ConstraintList()
        for i in range(n_tenors):
            m.cons_no_nan.add(expr=m.v_rho[i] <= 0)
            m.cons_no_nan.add(expr=m.v_rho[i] >= -1)
            # m.cons_no_nan.add(expr=m.sqrt_right_term[tenors[i]] >= 0)
            # for k in [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]:
            #     m.left_term = {tenor_dt: m.v_theta[i] + m.v_rho[i] * m.v_psi[i] * k for i, tenor_dt in enumerate(tenors)}

            # left term + sqrt of right term must be gt 0
            # m.cons_no_nan.add(expr=m.left_term[tenors[i]] >= m.sqrt_right_term[tenors[i]]**0.5)

            # m.cons_no_nan.add(expr=m.v_psi[i] >= 0)
            # m.cons_no_nan.add(expr=m.sqrt_right_term[tenors[i]] >= 0)
            # m.cons_no_nan.add(expr=((m.v_psi[i] * m.tenors_flt[i] + m.v_theta[i] * m.v_rho[i]) ** 2 + m.v_theta[i] ** 2 * (1 - m.v_rho[i] ** 2)) >= 0)

        # Constraints on Calendar arbitrage
        m.cons_calendar_arb = pyo.ConstraintList()
        m.cons_calendar_arb.add(expr=m.v_theta[0] >= 0)
        for i in range(n_tenors-1):
            m.cons_calendar_arb.add(expr=m.v_theta[i] <= m.v_theta[i + 1])

        # # Constraints on Quotes
        # m.cons_quantity = pyo.ConstraintList()
        # m.c_max_abs_positions = Constraint(expr=sum([m.v_var_abs[i] for i in range(n_options)]) <= cfg.n_contracts)
        # for i in range(n_options):
        #     m.cons_quantity.add(expr=m.v_var_abs[i] <= n_option_max)

        # Objectives.
        # from pyomo.contrib.preprocessing import deactivate_trivial_constraints

        # m.obj = pyo.Objective(expr=sum(m.rmse_fills) + m.rmse_quotes, sense=pyo.minimize)
        m.obj = pyo.Objective(expr=sum(m.rmse_fills), sense=pyo.minimize)
        logger.info(f'Solving for #Tenor: {n_tenors}')
        log_infeasible_constraints(m, logger=logger, log_expression=True, log_variables=True)
        solver = SolverFactory('ipopt')

        inst = m.create_instance()
        # https://pyomo.readthedocs.io/en/stable/contributed_packages/mindtpy.html#mindtpy-implementation-and-optional-arguments
        t0 = time.time()

        res = solver.solve(inst)
        logger.info(f'Solved in {time.time() - t0}s')
        print(res)
        inst.display()

        # Storing state, params and error
        surface_params = {}

        for i, tenor_dt in enumerate(tenors):
            params = pyo.value(inst.v_theta[i]), pyo.value(inst.v_rho[i]), pyo.value(inst.v_psi[i])
            surface_params[tenor_dt] = SliceParams(right, tenor_dt, *params)

            self.calibration_items[tenor_dt, right] = fills.get(tenor_dt)
            self.params[tenor_dt, right] = params
            iv_fills_ssvi = f_essvi_iv(fills[tenor_dt].mny_fwd_ln, *params, tenor=fills[tenor_dt].tenor)
            self.rmse[tenor_dt, right] = sum((iv_fills_ssvi - fills[tenor_dt].iv)**2)**0.5

        info(f'Calibration done for {self.underlying} total_rmse={sum(self.rmse.values())}')
        self.is_calibrated = True

        return CalibrationResult(right, surface_params, m)


if __name__ == '__main__':
    # from connector.api_minlp.common import repair_quotes_keeping_holding_quotes
    import sys
    sys.path.append('C:/repos/trade/src')
    from derivatives.frame_builder import get_option_frame
    from connector.api_minlp.common import exclude_outlier_quotes
    sym = 'FDX'.lower()

    equity = Equity(sym)
    resolution = Resolution.second
    seq_ret_threshold = 0.002
    release_date = EarningsPreSessionDates(sym)[-1]

    dates = [d.date() for d in (pd.date_range(release_date, release_date + timedelta(days=1), freq='D'))]
    option_frame = get_option_frame(equity, dates, resolution=resolution, seq_ret_threshold=seq_ret_threshold)

    df = option_frame.df_options.sort_index()
    df_trades = option_frame.df_option_trades.sort_index()

    # A little cleaning
    df_trades = exclude_outlier_quotes(df_trades, [], equity)
    df_trades = df_trades[
        (df_trades['fill_iv'] < 4) &
        ((df_trades['bid_close'] <= df_trades['close']) | (df_trades['ask_close'] >= df_trades['close']))
        ]
    df = df[(df['bid_iv'] < 4) & (df['ask_iv'] < 4)]

# ##############################################################
    ivs = IVSurface(equity)
    # Calibration
    y_col_nm = 'mid_price_iv'
    calibration_items = []

    calibration_items_fill = df2calibration_items(df_trades, release_date, y_col_nm='fill_iv', weight_col_nm='volume', vega_col_nm='vega_fill_iv')
    calibration_items_quotes = df2calibration_items(df, release_date, y_col_nm='mid_price', weight_col_nm='vega_mid_iv')

    for right in [OptionRight.call, OptionRight.put]:
        fills = {item.tenor_dt: item for item in calibration_items_fill if item.right == right}
        bids = {item.tenor_dt: item for item in calibration_items_bid if item.right == right}
        asks = {item.tenor_dt: item for item in calibration_items_ask if item.right == right}
        ivs.calibrate_surface(right, fills, quotes)


# #############################################################
#     ivs_bid = IVSurface(equity)
#     # Calibration
#     y_col_nm = 'bid_iv'
#     calibration_items = []
#     for expiry, expiry_df in df.groupby('expiry'):
#         for right, right_df in expiry_df.groupby('right'):
#             calibration_items.append(CalibrationItem(right_df['moneyness_fwd_ln'].values.astype(float), release_date, expiry, right, right_df[y_col_nm].values.astype(float),
#                                                      right_df['vega_mid_iv'].values.astype(float)))
#     ivs_bid.calibrate(calibration_items)
#     print(ivs_bid.iv(0.1, expiry, OptionRight.put, release_date))
# ###################################################################
#     ivs_ask = IVSurface(equity)
#     # Calibration
#     y_col_nm = 'ask_iv'
#     calibration_items = []
#     for expiry, expiry_df in df.groupby('expiry'):
#         for right, right_df in expiry_df.groupby('right'):
#             calibration_items.append(CalibrationItem(right_df['moneyness_fwd_ln'].values.astype(float), release_date, expiry, right, right_df[y_col_nm].values.astype(float),
#                                                      right_df['vega_mid_iv'].values.astype(float)))
#     ivs_ask.calibrate(calibration_items)
#     print(ivs_ask.iv(0.1, expiry, OptionRight.put, release_date))
