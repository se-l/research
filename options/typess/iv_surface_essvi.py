import time
from functools import lru_cache

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import numba as nb

from uuid import uuid4, UUID
from dataclasses import dataclass, fields
from itertools import chain, groupby
from datetime import date, datetime
from typing import Dict, List, Tuple, Callable, Iterable, Literal
from plotly.subplots import make_subplots
from statsmodels.tools.eval_measures import rmse, meanabs
from options.helper import get_tenor, get_v_iv
from options.typess.calibration_item import CalibrationItem
from options.typess.iv_surface_model_evaluation import IVSurfaceModelEvaluation
from options.typess.option import Option, price_put, price_call
from options.typess.enums import OptionRight
from options.typess.equity import Equity
from scipy.optimize import least_squares

from shared.constants import EarningsPreSessionDates, DiscountRateMarket, DividendYield
from shared.modules.logger import warning, info, error
from shared.plotting import show


@nb.jit(nopython=True)
def f_essvi_total_variance(k: np.ndarray, theta: float, rho: float, psi: float):
    """
    Implied total variance surface
    eSSVI(K, T) = 1/2 ( θ(T) + ρ(T)ψ(T)k + sqrt((ψ(T)k + θ(T)ρ(T))**2 + θ(T)**2 * (1 − ρ(T)**2))
    """
    return 0.5 * (theta + rho * psi * k + ((psi * k + theta * rho) ** 2 + theta ** 2 * (1 - rho ** 2)) ** (1 / 2))


@nb.jit(nopython=True)
def f_essvi_iv(k: np.ndarray, theta: float, rho: float, psi: float, tenor: float):
    """
    Implied total variance surface
    eSSVI(K, T) = 1/2 ( θ(T) + ρ(T)ψ(T)k + sqrt((ψ(T)k + θ(T)ρ(T))**2 + θ(T)**2 * (1 − ρ(T)**2))
    """
    return (f_essvi_total_variance(k, theta, rho, psi) / tenor) ** (1 / 2)


@nb.njit()
def f_min(x, u, y, f=f_essvi_total_variance, weights=None):
    theta, rho, psi = x
    k = u

    residuals = f(k, theta, rho, psi) - y
    if weights is not None:
        residuals *= weights

    if np.isnan(residuals).any() or np.isinf(residuals).any():
        # print('nan or inf residuals')
        residuals = fill_nan_residuals(residuals)

    return residuals


def f_min_price_surface_theta_rho_psi(calibration_params: List[float], samples: Dict[date, CalibrationItem], right: str | OptionRight):
    lst_residuals = []
    nan_encountered = []

    for i, (tenor_dt, item) in enumerate(samples.items()):
        theta = calibration_params[i * 3]
        rho = calibration_params[i * 3 + 1]
        psi = calibration_params[i * 3 + 2]

        model_variance = f_essvi_total_variance(item.mny_fwd_ln, theta, rho, psi)
        model_iv = (model_variance / item.tenor) ** 0.5

        # residuals = model_iv - item.iv

        f: Callable = price_put if right == OptionRight.put else price_call
        model_prices = np.array(f(item.spot, item.strike, item.tenor, model_iv, item.rf, item.dividend_yield))
        residuals = model_prices - item.price

        # if item.weights is not None:
        #     residuals *= item.weights

        if np.isnan(residuals).any() or np.isinf(residuals).any():
            nan_encountered.append((tenor_dt, right))
            residuals = fill_nan_residuals(residuals)

        lst_residuals += list(residuals)

    if nan_encountered:
        print(f'nan or inf residuals {right} {len(nan_encountered)}')
    return lst_residuals


@nb.njit()
def fill_nan_residuals(residuals: np.ndarray) -> np.ndarray:
    ix_nan = np.isnan(residuals) | np.isinf(residuals)
    res_max = np.abs(residuals[~ix_nan])
    if len(res_max) > 0:
        residuals[ix_nan] = np.max(res_max) * 10
    else:
        residuals[ix_nan] = 1e3
    return residuals


@nb.njit()
def f_gj(theta: float, rho: float):
    return 4 * theta / (1 + abs(rho))


@nb.njit()
def convert_rho_b_c2theta_rho_psi(model_params: np.ndarray[float], n_tenors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v_f = np.empty(n_tenors)
    v_p = np.empty(n_tenors)
    v_psi = np.empty(n_tenors)
    v_theta = np.empty(n_tenors)
    v_A = np.zeros(n_tenors)
    v_C = np.empty(n_tenors)

    v_theta[0] = model_params[0]
    v_rho = model_params[1:n_tenors + 1]
    v_a = model_params[1 + n_tenors:2 * n_tenors]
    v_c = model_params[-n_tenors:]

    # assert len(v_rho) + len(v_a) + len(v_c) + 1 == 3 * n_tenors == len(model_params)

    for i in range(n_tenors):
        rho = v_rho[i]
        if i > 0:
            rho_m1 = v_rho[i - 1]
            v_p[i] = max((1 + rho_m1) / (1 + rho), (1 - rho_m1) / (1 - rho))
            v_theta[i] = v_theta[i - 1] * v_p[i] + v_a[i - 1]

        v_f[i] = min(4 / (1 + abs(rho)), f_gj(v_theta[i], rho) ** 0.5)

    v_C[0] = min(np.array([v_f[0]] + [v_f[i] / v_p[i] for i in range(1, n_tenors)]))
    v_psi[0] = v_c[0] * (v_C[0] - v_A[0]) + v_A[0]

    for i in range(1, n_tenors):
        rest = [v_f[k] / v_p[k] for k in range(i + 1, n_tenors)]
        v_A[i] = v_psi[i - 1] * v_p[i]
        v_C[i] = min(np.array([(v_psi[i - 1] / v_theta[i - 1]) * v_theta[i], v_f[i]] + rest))
        v_psi[i] = v_c[i] * (v_C[i] - v_A[i]) + v_A[i]

    return v_theta, v_rho, v_psi


def f_min_reparameterized(model_params: np.ndarray[float], samples: Dict[date, CalibrationItem], right: str | OptionRight, f: Callable, n_residuals: int):
    v_residuals = np.empty(n_residuals)
    v_theta, v_rho, v_psi = convert_rho_b_c2theta_rho_psi(model_params, len(samples))
    nan_encountered = []

    ix = 0
    for i, (tenor_dt, item) in enumerate(samples.items()):
        model_variance = f_essvi_total_variance(item.mny_fwd_ln, v_theta[i], v_rho[i], v_psi[i])
        model_iv = (model_variance / item.tenor) ** 0.5

        model_prices = f(item.spot, item.strike, item.tenor, model_iv, item.rf, item.dividend_yield)
        residuals = model_prices - item.price

        if item.weights is not None:
            residuals *= item.weights

        if np.isnan(residuals).any() or np.isinf(residuals).any():
            nan_encountered.append((tenor_dt, right))
            residuals = fill_nan_residuals(residuals)

        len_residuals = len(residuals)
        v_residuals[ix:ix + len_residuals] = residuals
        ix += len_residuals

    if nan_encountered:
        print(f'# nan encountered: {len(nan_encountered)}: {nan_encountered}')
    return v_residuals


@dataclass
class MetricSSVI:
    theta: str = 'theta'
    rho: str = 'rho'
    psi: str = 'psi'

    @classmethod
    def names(cls):
        return [f.name for f in fields(cls)]


@dataclass
class DataPointSSVIOverTime:
    surface_id: UUID
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
        self.id = uuid4()
        self.underlying = underlying
        self.is_calibrated = False
        self.calibration_items: Dict[Tuple[date, str | OptionRight], CalibrationItem] = {}
        self.params: Dict[Tuple[date, str | OptionRight], Tuple] = {}
        self.rmse_iv: Dict[Tuple[date, str | OptionRight], Tuple] = {}
        self.evaluation: IVSurfaceModelEvaluation = None
        self.tag = tag
        self._last_calibration_ts: datetime = None

    def __repr__(self):
        return f'IVS_SSVI_{self.underlying.symbol}_{self.tag}_{self.last_calibration_ts().isoformat()}'

    def calibrate(self, calibration_items: List[CalibrationItem], verbose=1, initial_params=None, plot_cost_gt=10):
        initial_params = initial_params or [0.2, -0.1, 2]

        for item in calibration_items:

            total_variance = item.iv ** 2 * item.tenor
            fit_res = least_squares(f_min, initial_params, args=(item.mny_fwd_ln, total_variance, f_essvi_total_variance, item.weights), verbose=verbose, max_nfev=1000)

            self.calibration_items[item.tenor_dt, item.right] = item
            self.params[item.tenor_dt, item.right] = fit_res.x
            self.rmse_iv[item.tenor_dt, item.right] = fit_res.cost

            if fit_res.cost > plot_cost_gt:
                warning(f'Calibration failed for {self.underlying} tenor={item.tenor_dt} {item.right} cost: {fit_res.cost}, calcDate={item.calculation_date}')
                self.plot_smile(item.tenor_dt, item.right, item.calculation_date)

        info(f'Calibration done for {self.underlying} rmse(iv)={np.mean(list(self.rmse_iv.values()))}')
        self.is_calibrated = True

        return self

    def set_params(self, params: Dict[Tuple[date, str | OptionRight], Tuple]):
        self.params = params
        self.is_calibrated = True
        return self

    @lru_cache(maxsize=1)
    def n_params(self):
        return len(self.params)

    def is_calibrated_slice(self, tenor_dt: date, right: OptionRight | str):
        return (tenor_dt, right) in self.params is not None

    def set_calibration_items(self, calibration_items: List[CalibrationItem]):
        for item in calibration_items:
            if (item.tenor_dt, item.right) in self.params:
                self.calibration_items[item.tenor_dt, item.right] = item

    def ps_rmse(self) -> pd.Series:
        return pd.Series(self.rmse_iv)

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

        fig = make_subplots(rows=3, cols=1, subplot_titles=[f'IV {self.underlying} {right} expiry: {tenor_dt}', 'Weights', 'Sample Count Hist'])
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
        # if v_vega is not None:
        #     fig.add_trace(go.Scatter(x=v_mny_fwd_ln, y=v_vega, mode='markers', marker=dict(size=4), name='vega'), row=3, col=1)
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

        fig = make_subplots(rows=3, cols=1, subplot_titles=[f'Prices {self.underlying} {right} expiry: {tenor_dt}', 'Weights', 'Sample Count Hist'])
        f: Callable = price_put if right == OptionRight.put else price_call
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
        # if v_vega is not None:
        #     fig.add_trace(go.Scatter(x=v_mny_fwd_ln, y=v_vega, mode='markers', marker=dict(size=4), name='vega'), row=3, col=1)
        fig.add_trace(go.Histogram(x=v_mny_fwd_ln, nbinsx=int((max(v_mny_fwd_ln) - min(v_mny_fwd_ln)) * 100), name='count'), row=3, col=1)
        show(fig, fn=fn or f'essvi_{self.underlying}_{tenor_dt}_{right}.html', open_browser=open_browser)

    def plot_surface_calibration_items(self, fn=None, open_browser=True):
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

                v_iv += get_v_iv(
                    p=v.price,
                    s=v.spot,
                    k=v.strike.astype(float),
                    t=np.array([tenor] * len(v.mny_fwd_ln)),
                    r=DiscountRateMarket,
                    right=np.full(len(v.mny_fwd_ln), {OptionRight.call: 'c', OptionRight.put: 'p'}[right_trace]),
                    q=DividendYield[self.underlying.symbol],
                ).tolist()

            fig.add_trace(go.Scatter3d(x=v_mny, y=v_tenor, z=v_iv, mode='markers', marker=dict(size=2), name=f'{right_trace} IV'))

        fn_plot = fn or f'raw_{self.underlying}_IV_surface.html'
        fig.update_layout(title=fn_plot, autosize=True, scene=dict(
            xaxis_title='Log forward moneyness',
            yaxis_title='Tenor',
            zaxis_title='Implied Volatility', ),
                          )
        show(fig, fn=fn_plot, open_browser=open_browser)

    def plot_surface_vanilla(self, calc_date1=date(2024, 3, 11), expiries: List[date] = None) -> go.Figure:
        _expiries = expiries or [date(2024, 3, 15), date(2024, 3, 22), date(2024, 3, 28), date(2024, 4, 5), date(2024, 4, 12), date(2024, 4, 19), date(2024, 4, 26),
                                 date(2024, 5, 17), date(2024, 6, 21), date(2024, 7, 19), date(2024, 8, 16), date(2024, 9, 20), date(2024, 12, 20), date(2025, 1, 17),
                                 date(2025, 6, 20), date(2025, 12, 19), date(2026, 1, 16)]
        fig = go.Figure()
        x = np.linspace(-0.2, 0.2, 100)
        y = []
        z = []
        for tenor_dt in _expiries:
            z += list(self.iv(x, tenor_dt, OptionRight.call, calc_date1))
            y += [get_tenor(tenor_dt, calc_date1)] * len(x)
        fig.add_trace(go.Scatter3d(x=x.tolist() * len(_expiries), y=y, z=z, mode='markers', marker=dict(size=2)))
        return fig

    def plot_surface_as_of(self, calc_date: date, fn=None, open_browser=True):
        fig = go.Figure()
        for right_trace in [OptionRight.call, OptionRight.put]:
            v_mny = []
            v_tenor = []
            v_iv = []
            for (tenor_dt, right), params in self.params.items():
                if right != right_trace:
                    continue
                x = np.linspace(-0.3, 0.3, 100)
                v_mny += list(x)
                v_iv += list(f_essvi_iv(x, *params, tenor=get_tenor(tenor_dt, calc_date)))
                v_tenor += [get_tenor(tenor_dt, calc_date)] * len(x)

            fig.add_trace(go.Scatter3d(x=v_mny, y=v_tenor, z=v_iv, mode='markers', marker=dict(size=2), name=f'{right_trace} IV'))

        fn_plot = fn or f'essvi_{self.underlying}_IV_surface.html'
        fig.update_layout(title=fn_plot, autosize=True, scene=dict(
            xaxis_title='Log forward moneyness',
            yaxis_title='Tenor',
            zaxis_title='Implied Volatility', ),
                          )
        show(fig, fn=fn_plot, open_browser=open_browser)


    def plot_model_params(self, calc_date: date, fn=None, open_browser=True):
        """Plot params and errors for each right across the surface."""
        rights = [OptionRight.call, OptionRight.put]
        subs = list(chain(*[[f'{right} {metric}' for right in rights] for metric in ['Theta', 'Rho', 'Psi']]))
        fig = make_subplots(rows=3, cols=2, subplot_titles=subs)
        for col, right in enumerate(rights):
            p_dct = {k: v for k, v in self.params.items() if k[1] == right}
            x = [get_tenor(k[0], calc_date) for k in p_dct.keys()]
            theta = [v[0] for k, v in p_dct.items()]
            rho = [v[1] for k, v in p_dct.items()]
            psi = [v[2] for k, v in p_dct.items()]
            fig.add_trace(go.Scatter(x=x, y=theta, mode='markers', marker=dict(size=4), name=f'{right} Theta'), row=1, col=col + 1)
            fig.add_trace(go.Scatter(x=x, y=rho, mode='markers', marker=dict(size=4), name=f'{right} Rho'), row=2, col=col + 1)
            fig.add_trace(go.Scatter(x=x, y=psi, mode='markers', marker=dict(size=4), name=f'{right} Psi'), row=3, col=col + 1)

        show(fig, fn=fn or f'essvi_{self.underlying}_model_params_{time.time()}.html', open_browser=open_browser)

    # def nlv(self, o: Option, s0, q: float) -> float:
    #     calc_date = self.first_calibration_calc_date()
    #
    #     iv = self.iv(get_moneyness_fwd_ln(self.underlying, strike: float | np.ndarray, spot: float | np.ndarray, tenor: float | np.ndarray))
    #     if iv is None or np.isnan(iv):
    #         return None
    #     o.npv(iv, s0, calc_date) * q * 100

    def iv(self, moneyness_fwd_ln: float | np.ndarray, tenor: date, right: OptionRight | str, calc_date: date) -> np.ndarray | float:
        """
        Interpolating IV from the frame.
        """
        if not self.is_calibrated:
            raise ValueError('IVSurface is not calibrated')
        try:
            return f_essvi_iv(moneyness_fwd_ln, *self.params[tenor, right], tenor=get_tenor(tenor, calc_date))
        except KeyError as e:
            # Could interpolate the surface. But probably should rather ifx something in the calibration.
            raise e

    def iv_jump_right(self, right: OptionRight, calc_date: date = None, release_date: date = None, min_days=5, tenor_dt1_dt2: Tuple[date, date] = None) -> float:
        """
        iv_jump**2 = (iv_t1**2 - iv_t2**2) / (1/t_1 - 1/t_2)
        """
        _calc_date = calc_date or self.last_calibration_ts().date()
        _release_date = release_date or self.next_release_date(_calc_date)

        if tenor_dt1_dt2 is None:
            tenor_dt1 = None
            tenor_dt2 = None
            for dt in self.tenors():
                if not (dt, OptionRight.put) in self.params or not (dt, OptionRight.call) in self.params:
                    continue
                if tenor_dt1 is None and (dt - _release_date).days > min_days:
                    tenor_dt1 = dt
                    continue
                if tenor_dt1 is not None:
                    tenor_dt2 = dt
                    break
        else:
            tenor_dt1, tenor_dt2 = tenor_dt1_dt2

        try:
            tenor_t1 = get_tenor(tenor_dt1, _calc_date)
            tenor_t2 = get_tenor(tenor_dt2, _calc_date)
            iv_t1 = self.iv(0, tenor_dt1, right, _calc_date)
            iv_t2 = self.iv(0, tenor_dt2, right, _calc_date)  # tenor available for calls, but not puts...

            return (iv_t1 ** 2 - iv_t2 ** 2) / (1 / tenor_t1 - 1 / tenor_t2) ** 0.5
        except KeyError as e:
            return np.nan

    def iv_jump(self, calc_date: date = None, release_date: date = None):
        _calc_date = calc_date or self.last_calibration_ts().date()
        _release_date = release_date or self.next_release_date(_calc_date)
        return (
                self.iv_jump_right(OptionRight.call, _calc_date, _release_date) +
                self.iv_jump_right(OptionRight.put, _calc_date, _release_date)
        ) / 2

    def iv_jumps(self, calc_date: date = None, release_date: date = None, min_days=1, max_days=365, max_k=0, max_i=5) -> Dict[str, float]:
        dct = {}

        _calc_date = calc_date or self.last_calibration_ts().date()
        _release_date = release_date or self.next_release_date(_calc_date)

        tenors = sorted([dt for dt in self.tenors() if max_days > (dt - _release_date).days > min_days])
        for i, tenor_dt in enumerate(tenors):
            if i > max_i:
                break
            match_tenors = sorted([dt for dt in self.tenors() if dt > tenor_dt])
            for k, match_tenor in enumerate(match_tenors):
                if k > max_k:
                    break
                try:
                    dct[f'iv_jump_{i}_{i+k+1}'] = (
                        self.iv_jump_right(OptionRight.call, _calc_date, _release_date, tenor_dt1_dt2=(tenor_dt, match_tenor)) +
                        self.iv_jump_right(OptionRight.put, _calc_date, _release_date, tenor_dt1_dt2=(tenor_dt, match_tenor))
                    ) / 2
                except KeyError:
                    pass
        return dct

    def tenors(self) -> List[date]:
        return sorted(list(set([k[0] for k in self.params.keys()])))

    def next_release_date(self, calc_date: date) -> date:
        return next(iter([d for d in EarningsPreSessionDates(self.underlying.symbol) if d >= calc_date]), date.max)

    def calibrate_surface_arb_free(self, right: str | OptionRight, samples: Dict[date, CalibrationItem], verbose=True):
        self._validate_calibration_samples(samples)
        t0 = time.time()
        tenors = sorted(samples.keys())
        n_tenors = len(tenors)

        # theta0, rho, a, c
        model_params = [0.05]  # theta0
        model_params += [0] * n_tenors  # rho
        model_params += [1e-3] * (n_tenors - 1)  # a
        model_params += [0.5] * n_tenors  # c
        model_params = np.array(model_params)

        bounds_left = [0]
        bounds_left += [-1] * n_tenors
        bounds_left += [0] * (n_tenors - 1)
        bounds_left += [-1] * n_tenors

        bounds_right = [np.inf]
        bounds_right += [1] * n_tenors
        bounds_right += [np.inf] * (n_tenors - 1)
        bounds_right += [1] * n_tenors

        f_pricing: Callable = price_put if right == OptionRight.put else price_call
        n_residuals = sum([len(item.iv) for item in samples.values()])
        fit_res = least_squares(f_min_reparameterized, model_params, args=(samples, right, f_pricing, n_residuals), verbose=verbose, max_nfev=1000,
                                bounds=(bounds_left, bounds_right)
                                )
        info(f'Calibrated arb free in {time.time() - t0:.2f}sec for {self.underlying} {right}')

        v_theta, v_rho, v_psi = convert_rho_b_c2theta_rho_psi(fit_res.x, len(samples))

        for i, (tenor_dt, item) in enumerate(samples.items()):
            self.calibration_items[item.tenor_dt, item.right] = item
            self.params[item.tenor_dt, item.right] = (v_theta[i], v_rho[i], v_psi[i])

        self.is_calibrated = True
        self.evaluate()

        return self

    @staticmethod
    def _validate_calibration_samples(items: Dict[date, CalibrationItem]):
        prev_tenor = None
        for tenor_dt, item in items.items():
            if prev_tenor is not None and (tenor_dt - prev_tenor).days < 0:
                raise ValueError(f'Tenors are not sorted for {prev_tenor} and {tenor_dt}')
            prev_tenor = tenor_dt
            if tenor_dt < item.calculation_date:
                raise ValueError(f'Calculation date {item.calculation_date} is the same as tenor date {tenor_dt}. Likely it"s the contracts expiry date. On expiry date, '
                                 f'option still trades. annualized tenor should not be zero.')
            if item.tenor == 0:
                raise ValueError(f'Tenor is zero for {item}')
            if item.weights is not None and (np.isnan(item.weights).any() or (item.weights <= 0).sum() > 0):
                raise ValueError(f'Weights contain nan or 0 for {item}')
        return True

    def calibrate_surface(self, right: str | OptionRight, samples: Dict[date, CalibrationItem], verbose=True):
        self._validate_calibration_samples(samples)
        t0 = time.time()
        if not samples:
            warning(f'No samples for {self.underlying} {right}')
            return self

        tenors = sorted(samples.keys())
        n_tenors = len(tenors)

        # theta (ATM impl var), rho (ATM slope), psi (ATM curvature)
        calibration_params = [0.05, -0.3, 0.2] * n_tenors

        info(f'Calibrating {self.underlying} {right}. # tenors: {n_tenors}. # samples: {len(samples)}. # params: {len(calibration_params)}')
        fit_res = least_squares(f_min_price_surface_theta_rho_psi, calibration_params, args=(samples, right), verbose=verbose, max_nfev=1000,
                                bounds=([0, -np.inf, -np.inf] * n_tenors, [np.inf, np.inf, np.inf] * n_tenors)  # ensure theta > 0
                                )
        info(f'Calibrated in {time.time() - t0:.2f}sec for {self.underlying} {right}')

        for i, (tenor_dt, item) in enumerate(samples.items()):
            theta = fit_res.x[i * 3]
            rho = fit_res.x[i * 3 + 1]
            psi = fit_res.x[i * 3 + 2]

            self.calibration_items[item.tenor_dt, item.right] = item
            self.params[item.tenor_dt, item.right] = (theta, rho, psi)

        self.is_calibrated = True
        self.evaluate()

        return self

    def evaluate(self, mny_bin_size=0.05) -> IVSurfaceModelEvaluation | None:
        if not self.is_calibrated:
            return None

        dct_iv = {}
        dct_iv_pred = {}
        dct_price = {}
        dct_price_pred = {}
        dct_spread = {}

        for (tenor_dt, right), item in self.calibration_items.items():
            theta, rho, psi = self.params[tenor_dt, right]

            dct_iv[(tenor_dt, right)] = item.iv
            dct_iv_pred[(tenor_dt, right)] = f_essvi_iv(item.mny_fwd_ln, theta, rho, psi, tenor=item.tenor)
            dct_price[(tenor_dt, right)] = item.price
            dct_price_pred[(tenor_dt, right)] = Option.price(item.spot, np.array(item.strike), item.tenor, dct_iv_pred[(tenor_dt, right)], item.rf, item.dividend_yield, right)
            if item.bid_price is not None and item.ask_price is not None:
                dct_spread[(tenor_dt, right)] = item.ask_price - item.bid_price

        dct_iv_by_mny = {}
        dct_iv_pred_by_mny = {}
        dct_price_by_mny = {}
        dct_price_pred_by_mny = {}
        dct_spread_by_mny = {}
        for (tenor_dt, right), item in self.calibration_items.items():
            theta, rho, psi = self.params[tenor_dt, right]

            for ix, gp in groupby(item.mny_fwd_ln, lambda x: x // mny_bin_size):
                key = (tenor_dt, right, ix * mny_bin_size)
                index = item.mny_fwd_ln // mny_bin_size == ix

                dct_iv_by_mny[key] = item.iv[index]
                dct_iv_pred_by_mny[key] = f_essvi_iv(item.mny_fwd_ln[index], theta, rho, psi, tenor=item.tenor)
                dct_price_by_mny[key] = item.price[index]
                dct_price_pred_by_mny[key] = Option.price(item.spot[index], np.array(item.strike[index]), item.tenor, dct_iv_pred_by_mny[key], item.rf, item.dividend_yield, right)
                if item.bid_price is not None and item.ask_price is not None:
                    dct_spread_by_mny[key] = item.ask_price[index] - item.bid_price[index]

        self.rmse_iv = {k: rmse(v, dct_iv_pred[k]) for k, v in dct_iv.items()}

        self.evaluation = IVSurfaceModelEvaluation(
            rmse_iv=rmse(list(chain(*dct_iv.values())), list(chain(*dct_iv_pred.values()))),
            rmse_iv_right_tenor={k: rmse(v, dct_iv_pred[k]) for k, v in dct_iv.items()},
            rmse_price=rmse(list(chain(*dct_price.values())), list(chain(*dct_price_pred.values()))),
            rmse_price_right_tenor={k: rmse(v, dct_price_pred[k]) for k, v in dct_price.items()},
            rmse_pc_of_spread=rmse(0, (np.array(list(chain(*dct_price_pred.values()))) - np.array(list(chain(*dct_price.values())))) / np.array(list(chain(*dct_spread.values())))),
            rmse_pc_of_spread_right_tenor={k: rmse(0, (dct_price_pred[k] - dct_price[k]) / dct_spread[k]) for k, v in dct_spread.items()},

            mae_iv=meanabs(list(chain(*dct_iv.values())), list(chain(*dct_iv_pred.values()))),
            mae_iv_right_tenor={k: meanabs(v, dct_iv_pred[k]) for k, v in dct_iv.items()},
            mae_price=meanabs(list(chain(*dct_price.values())), list(chain(*dct_price_pred.values()))),
            mae_price_right_tenor={k: meanabs(v, dct_price_pred[k]) for k, v in dct_price.items()},
            mae_pc_of_spread=meanabs(0,
                                     (np.array(list(chain(*dct_price_pred.values()))) - np.array(list(chain(*dct_price.values())))) / np.array(list(chain(*dct_spread.values())))),
            mae_pc_of_spread_right_tenor={k: meanabs(0, (dct_price_pred[k] - dct_price[k]) / dct_spread[k]) for k, v in dct_spread.items()},
        )
        return self.evaluation

    def plot_error_surface(self, metric: Literal['iv', 'usd'], error: Literal['mae', 'rmse'], fn=None, open_browser=True):
        """
        Need to measure error buckets of moneyness first
        USD error is vega * iv error
        """
        if self.evaluation is None:
            self.evaluate()

    @property
    def datapoints(self) -> List[DataPointSSVIOverTime]:
        values = []
        calc_date = self.first_calibration_calc_date()
        ts_end = self.last_calibration_ts()

        for (tenor_dt, right), (theta, rho, psi) in self.params.items():
            tenor = get_tenor(tenor_dt, calc_date)
            values += [
                DataPointSSVIOverTime(self.id, self.underlying.symbol, right, tenor_dt, tenor, MetricSSVI.theta, theta, ts_end),
                DataPointSSVIOverTime(self.id, self.underlying.symbol, right, tenor_dt, tenor, MetricSSVI.rho, rho, ts_end),
                DataPointSSVIOverTime(self.id, self.underlying.symbol, right, tenor_dt, tenor, MetricSSVI.psi, psi, ts_end),
            ]
        return values

    def last_calibration_ts(self) -> datetime:
        if not hasattr(self, '_last_calibration_ts') or not self._last_calibration_ts:
            np_ts = max([ci.ts[-1] for ci in self.calibration_items.values() if ci is not None and len(ci.ts) > 0])
            self._last_calibration_ts = pd.Timestamp(np_ts).to_pydatetime()
        return self._last_calibration_ts

    def set_last_calibration_ts(self, ts: datetime):
        self._last_calibration_ts = ts

    def first_calibration_calc_date(self) -> date:
        obj = next(iter(self.calibration_items.values()), None)
        return obj.calculation_date if obj is not None else date.min


def plot_ssvi_params_over_time(v_ivs: Iterable[IVSurface], fn=None, open_browser=True):
    """Plot how the surface params change over time"""
    if not v_ivs:
        return
    df = pd.DataFrame(chain(*(ivs.datapoints for ivs in v_ivs)))

    rights = [OptionRight.call, OptionRight.put]
    metrics = MetricSSVI.names()
    subs = list(chain(*[[f'{right} {metric}' for right in rights] for metric in metrics]))

    for underlying, s_df in df.groupby('underlying'):
        from_ = s_df['ts_end'].min().date()
        to = s_df['ts_end'].max().date()
        fig = make_subplots(rows=len(subs) // 2, cols=2, subplot_titles=subs)
        for right, ss_df in s_df.groupby('right'):
            ix_col = rights.index(right) + 1
            for metric, sss_df in ss_df.groupby('model_param'):
                ix_row = metrics.index(metric) + 1
                for tenor_dt, tenor_df in sss_df.groupby('tenor_dt'):
                    fig.add_trace(go.Scatter(x=tenor_df['ts_end'], y=tenor_df['value'], mode='markers', marker=dict(size=4), name=f'{right} {tenor_dt} {metric}'), row=ix_row,
                                  col=ix_col)

                    metric_ma = tenor_df['value'].rolling(5).mean()
                    fig.add_trace(go.Scatter(x=tenor_df['ts_end'], y=metric_ma, mode='lines', name=f'{right} {tenor_dt} {metric} MA', line=dict(color='red')), row=ix_row, col=ix_col)

        fig.update_layout(title=f'{underlying} SSVI params (t)', autosize=True)
        show(fig, fn=fn or f'essvi_{underlying}_params_params_{from_}-{to}.html', open_browser=open_browser)
