from dataclasses import dataclass, fields
from functools import reduce
from itertools import chain

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from datetime import date, timedelta, datetime
from typing import Dict, List, Tuple, Callable, Iterable
from plotly.subplots import make_subplots
from statsmodels.tools.eval_measures import rmse

from options.helper import get_tenor
from options.typess.calibration_item import CalibrationItem, df2calibration_items
from options.typess.option import Option
from shared.constants import EarningsPreSessionDates
from options.typess.enums import Resolution, OptionRight
from options.typess.equity import Equity
from scipy.optimize import least_squares

from shared.modules.logger import warning, info
from shared.plotting import show, plot_scatter_3d


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


def f_min_price_surface_theta_rho_psi(calibration_params: list, samples: Dict[date, CalibrationItem], right: str | OptionRight):

    lst_residuals = []

    for i, (tenor_dt, item) in enumerate(samples.items()):
        theta = calibration_params[i * 3]
        rho = calibration_params[i * 3 + 1]
        psi = calibration_params[i * 3 + 2]

        model_variance = f_essvi_total_variance(item.mny_fwd_ln, theta, rho, psi)
        model_iv = (model_variance / item.tenor) ** 0.5

        # residuals = model_iv - item.iv

        f: Callable = Option.price_put if right == OptionRight.put else Option.price_call
        model_prices = np.array(f(item.spot, np.array(item.strike), item.tenor, model_iv, item.rf, item.dividend_yield))
        residuals = model_prices - item.price

        # if item.weights is not None:
        #     residuals *= item.weights

        if np.isnan(residuals).any() or np.isinf(residuals).any():
            print(f'nan or inf residuals {right} {tenor_dt}')
            ix_nan = np.isnan(residuals) | np.isinf(residuals)
            res_max = np.abs(residuals[~ix_nan])
            if len(res_max) > 0:
                residuals[ix_nan] = np.max(res_max) * 10
            else:
                residuals[ix_nan] = 1e3

        lst_residuals += list(residuals)

    return lst_residuals


def f_gj(theta, rho):
    return 4*theta / (1 + abs(rho))


def convert_rho_b_c2theta_rho_psi(model_params: list, n_tenors: int):
    v_f = np.empty(n_tenors)
    v_p = np.empty(n_tenors)
    v_psi = np.empty(n_tenors)
    v_theta = np.empty(n_tenors)
    v_A = np.zeros(n_tenors)
    v_C = np.empty(n_tenors)

    v_theta[0] = model_params[0]
    v_rho = model_params[1:n_tenors+1]
    v_a = model_params[1+n_tenors:2*n_tenors]
    v_c = model_params[-n_tenors:]

    assert len(v_rho) + len(v_a) + len(v_c) + 1 == 3 * n_tenors == len(model_params)

    for i in range(n_tenors):
        rho = v_rho[i]
        if i > 0:
            rho_m1 = v_rho[i-1]
            v_p[i] = max((1 + rho_m1) / (1 + rho), (1 - rho_m1) / (1 - rho))
            v_theta[i] = v_theta[i-1] * v_p[i] + v_a[i-1]

        v_f[i] = min(4 / (1 + abs(rho)), f_gj(v_theta[i], rho) ** 0.5)

    v_C[0] = min(np.array([v_f[0]] + [v_f[i] / v_p[i] for i in range(1, n_tenors)]))
    v_psi[0] = v_c[0] * (v_C[0] - v_A[0]) + v_A[0]

    for i in range(1, n_tenors):
        rest = [v_f[k] / v_p[k] for k in range(i+1, n_tenors)]
        v_A[i] = v_psi[i-1] * v_p[i]
        v_C[i] = min(np.array([(v_psi[i-1] / v_theta[i-1]) * v_theta[i], v_f[i]] + rest))
        v_psi[i] = v_c[i] * (v_C[i] - v_A[i]) + v_A[i]

    return v_theta, v_rho, v_psi


def f_min_reparameterized(model_params: list, samples: Dict[date, CalibrationItem], right: str | OptionRight):

    lst_residuals = []
    v_theta, v_rho, v_psi = convert_rho_b_c2theta_rho_psi(model_params, len(samples))

    for i, (tenor_dt, item) in enumerate(samples.items()):
        model_variance = f_essvi_total_variance(item.mny_fwd_ln, v_theta[i], v_rho[i], v_psi[i])
        model_iv = (model_variance / item.tenor) ** 0.5

        f: Callable = Option.price_put if right == OptionRight.put else Option.price_call

        model_prices = np.array(f(item.spot, np.array(item.strike), item.tenor, model_iv, item.rf, item.dividend_yield))
        residuals = model_prices - item.price

        if item.weights is not None:
            residuals *= item.weights

        if np.isnan(residuals).any() or np.isinf(residuals).any():
            print(f'nan or inf residuals {right} {tenor_dt}')
            ix_nan = np.isnan(residuals) | np.isinf(residuals)
            res_max = np.abs(residuals[~ix_nan])
            if len(res_max) > 0:
                residuals[ix_nan] = np.max(res_max) * 10
            else:
                residuals[ix_nan] = 1e3

        lst_residuals += list(residuals)

    return lst_residuals


@dataclass
class MetricSSVI:
    theta: str = 'theta'
    rho: str = 'rho'
    psi: str = 'psi'
    rmse: str = 'rmse'

    @classmethod
    def names(cls):
        return [f.name for f in fields(cls)]


@dataclass
class DataPointSSVIOverTime:
    underlying: str
    right: OptionRight | str
    tenor_dt: date
    tenor: float
    model_param: MetricSSVI | str
    value: float
    ts_end: datetime


class IVSurface:
    """
    Enrich a frame with index ts, expiry, strike, right
    So plotting is easy
    """
    def __init__(self, underlying: Equity, tag: str = ''):
        self.underlying = underlying
        self.is_calibrated = False
        self.calibration_items: Dict[Tuple[date, str | OptionRight], CalibrationItem] = {}
        self.params: Dict[Tuple[date, str | OptionRight], Tuple] = {}
        self.rmse: Dict[Tuple[date, str | OptionRight], Tuple] = {}
        self.tag = tag

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
        subs = list(chain(*[[f'{right} {metric}' for right in rights] for metric in ['Theta', 'Rho', 'Psi', 'RMSE']]))
        fig = make_subplots(rows=4, cols=2, subplot_titles=subs)
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
            fig.add_trace(go.Scatter(x=x, y=rmse, mode='markers', marker=dict(size=4), name=f'{right} RMSE by tenor'), row=4, col=col + 1)

            # fig.add_trace(go.Scatter(x=x, y=rmse, mode='markers', marker=dict(size=4), name=f'{right} RMSE by ln mny fwd'), row=5, col=col + 1)

        show(fig, fn=fn or f'essvi_{self.underlying}_params_rmse.html', open_browser=open_browser)

    def iv(self, moneyness_fwd_ln: float, tenor: date, right: OptionRight, calc_date: date) -> float:
        """
        Interpolating IV from the frame.
        """
        if not self.is_calibrated:
            raise ValueError('IVSurface is not calibrated')
        return f_essvi_iv(moneyness_fwd_ln, *self.params[tenor, right], tenor=get_tenor(tenor, calc_date))

    def calibrate_surface_arb_free(self, right: str | OptionRight, samples: Dict[date, CalibrationItem], verbose=True):
        tenors = sorted(samples.keys())
        n_tenors = len(tenors)

        # theta0, rho, a, c
        model_params = [0.05]  # theta0
        model_params += [0] * n_tenors  # rho
        model_params += [1e-3] * (n_tenors-1)  # a
        model_params += [0.5] * n_tenors  # c

        bounds_left = [0]
        bounds_left += [-1] * n_tenors
        bounds_left += [0] * (n_tenors-1)
        bounds_left += [-1] * n_tenors

        bounds_right = [np.inf]
        bounds_right += [1] * n_tenors
        bounds_right += [np.inf] * (n_tenors-1)
        bounds_right += [1] * n_tenors

        fit_res = least_squares(f_min_reparameterized, model_params, args=(samples, right), verbose=verbose, max_nfev=1000,
                                bounds=(bounds_left, bounds_right)
                                )

        v_theta, v_rho, v_psi = convert_rho_b_c2theta_rho_psi(fit_res.x, len(samples))

        for i, (tenor_dt, item) in enumerate(samples.items()):
            self.calibration_items[item.tenor_dt, item.right] = item
            self.params[item.tenor_dt, item.right] = (v_theta[i], v_rho[i], v_psi[i])
            self.rmse[item.tenor_dt, item.right] = rmse(
                item.iv,
                f_essvi_iv(item.mny_fwd_ln, v_theta[i], v_rho[i], v_psi[i], tenor=item.tenor)
            )

        # if fit_res.cost > plot_cost_gt:
        #     warning(f'Calibration failed for {self.underlying} tenor={item.tenor_dt} {item.right} cost: {fit_res.cost}, calcDate={item.calculation_date}')
        #     self.plot_smile(item.tenor_dt, item.right, item.calculation_date)

        info(f'Calibration done for {self.underlying} {right} total_rmse={sum(self.rmse.values())}')
        self.is_calibrated = True

        return self

    def calibrate_surface(self, right: str | OptionRight, samples: Dict[date, CalibrationItem], verbose=True):
        if not samples:
            warning(f'No samples for {self.underlying} {right}')
            return self

        tenors = sorted(samples.keys())
        n_tenors = len(tenors)

        # theta, rho, psi
        calibration_params = [0.05, -0.3, 0.2] * n_tenors

        fit_res = least_squares(f_min_price_surface_theta_rho_psi, calibration_params, args=(samples, right), verbose=verbose, max_nfev=1000,
                                bounds=([0, -np.inf, -np.inf] * n_tenors, [np.inf, np.inf, np.inf] * n_tenors)  # ensure theta > 0
                                )

        for i, (tenor_dt, item) in enumerate(samples.items()):
            theta = fit_res.x[i * 3]
            rho = fit_res.x[i * 3 + 1]
            psi = fit_res.x[i * 3 + 2]

            self.calibration_items[item.tenor_dt, item.right] = item
            self.params[item.tenor_dt, item.right] = (theta, rho, psi)
            self.rmse[item.tenor_dt, item.right] = rmse(
                item.iv,
                f_essvi_iv(item.mny_fwd_ln, theta, rho, psi, tenor=item.tenor)
            )

        info(f'Calibration done for {self.underlying} {right} total_rmse={sum(self.rmse.values())}')
        self.is_calibrated = True

        return self

    @property
    def datapoints(self) -> List[DataPointSSVIOverTime]:
        values = []
        calc_date = next(iter(self.calibration_items.values())).calculation_date
        ts_end = list(self.calibration_items.values())[-1].ts[-1]

        for (tenor_dt, right), (theta, rho, psi) in self.params.items():
            tenor = get_tenor(tenor_dt, calc_date)
            values += [
                DataPointSSVIOverTime(self.underlying.symbol, right, tenor_dt, tenor, MetricSSVI.theta, theta, ts_end),
                DataPointSSVIOverTime(self.underlying.symbol, right, tenor_dt, tenor, MetricSSVI.rho, rho, ts_end),
                DataPointSSVIOverTime(self.underlying.symbol, right, tenor_dt, tenor, MetricSSVI.psi, psi, ts_end),
                DataPointSSVIOverTime(self.underlying.symbol, right, tenor_dt, tenor, MetricSSVI.rmse, self.rmse[(tenor_dt, right)], ts_end),
            ]
        return values


def plot_ssvi_params_over_time(v_ivs: Iterable[IVSurface], fn=None, open_browser=True):
    """Plot how the surface params change over time"""
    if not v_ivs:
        return
    df = pd.DataFrame(chain(*(ivs.datapoints for ivs in v_ivs)))

    rights = [OptionRight.call, OptionRight.put]
    metrics = MetricSSVI.names()
    subs = list(chain(*[[f'{right} {metric}' for right in rights] for metric in metrics]))

    for underlying, s_df in df.groupby('underlying'):
        fig = make_subplots(rows=4, cols=2, subplot_titles=subs)
        for right, ss_df in s_df.groupby('right'):
            ix_col = rights.index(right) + 1
            for metric, sss_df in ss_df.groupby('model_param'):
                ix_row = metrics.index(metric) + 1
                for tenor_dt, tenor_df in sss_df.groupby('tenor_dt'):
                    fig.add_trace(go.Scatter(x=tenor_df['ts_end'], y=tenor_df['value'], mode='markers+lines', marker=dict(size=4), name=f'{right} {tenor_dt} {metric}'), row=ix_row, col=ix_col)

        fig.update_layout(title=f'{underlying} SSVI params over time', autosize=True)
        show(fig, fn=fn or f'essvi_{underlying}_params_params.html', open_browser=open_browser)


# if __name__ == '__main__':
#     # from connector.api_minlp.common import repair_quotes_keeping_holding_quotes
#     import sys
#     sys.path.append('C:/repos/trade/src')
#     from derivatives.frame_builder import get_option_frame
#     from connector.api_minlp.common import exclude_outlier_quotes
#     sym = 'FDX'.lower()
#
#     equity = Equity(sym)
#     resolution = Resolution.second
#     seq_ret_threshold = 0.002
#     release_date = EarningsPreSessionDates(sym)[-1]
#
#     dates = [d.date() for d in (pd.date_range(release_date, release_date + timedelta(days=1), freq='D'))]
#     option_frame = get_option_frame(equity, dates, resolution=resolution, seq_ret_threshold=seq_ret_threshold)
#
#     df = option_frame.df_options.sort_index()
#     df_trades = option_frame.df_option_trades.sort_index()
#
#     # A little cleaning
#     df_trades = exclude_outlier_quotes(df_trades, [], equity)
#     df_trades = df_trades[
#         (df_trades['fill_iv'] < 4) &
#         ((df_trades['bid_close'] <= df_trades['close']) | (df_trades['ask_close'] >= df_trades['close']))
#         ]
#     df = df[(df['bid_iv'] < 4) & (df['ask_iv'] < 4)]
#
# # ##############################################################
#     ivs = IVSurface(equity)
#     # Calibration
#     y_col_nm = 'mid_price_iv'
#     calibration_items = []
#
#     calibration_items_fill = df2calibration_items(df_trades, release_date, y_col_nm='fill_iv', weight_col_nm='volume', vega_col_nm='vega_fill_iv')
#     calibration_items_quotes = df2calibration_items(df, release_date, y_col_nm='mid_price', weight_col_nm='vega_mid_iv')
#
#     for right in [OptionRight.call, OptionRight.put]:
#         fills = {item.tenor_dt: item for item in calibration_items_fill if item.right == right}
#         bids = {item.tenor_dt: item for item in calibration_items_bid if item.right == right}
#         asks = {item.tenor_dt: item for item in calibration_items_ask if item.right == right}
#         ivs.calibrate_surface(right, fills, quotes)
