import os
import time

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from functools import lru_cache
from uuid import uuid4, UUID
from dataclasses import dataclass, fields
from itertools import chain, groupby
from datetime import date, datetime
from typing import Dict, List, Tuple, Callable, Iterable, Literal

from numpy._typing import NDArray
from plotly.subplots import make_subplots
from statsmodels.tools.eval_measures import rmse, meanabs
from options.helper import get_tenor, timer, get_moneyness_fwd, date_to_sod, dividends2amount_times, get_pkl_cache_key # jl
from options.types.SSVIParams import SSVISurfParams, SSVITenorParams # jl
from options.types.calibration_item import CalibrationItem # jl
from options.types.dividend import get_dividends # jl
from options.types.iv_surface_model_evaluation import IVSurfaceModelEvaluation
from options.types.enums import OptionRight, Resolution # jl
from options.types.equity import Equity # jl
from scipy.optimize import least_squares
from fino.fino_load import jl
from options.types.quote_side import QuoteSide # jl

from shared.constants import EarningsPreSessionDates # jl
from shared.modules.logger import warning, info # jl
from shared.paths import Paths # jl
from shared.plotting import show
from shared.yield_curve import YieldCurve, ZeroCurveData # jl


# @nb.jit(nopython=True, fastmath=True)
def f_essvi_total_variance(k: NDArray[np.float64] | float, theta: NDArray[np.float64] | float, rho: NDArray[np.float64] | float, psi: NDArray[np.float64] | float):
    """
    Implied total variance surface
    eSSVI(K, T) = 1/2 ( θ(T) + ρ(T)ψ(T)k + sqrt((ψ(T)k + θ(T)ρ(T))**2 + θ(T)**2 * (1 − ρ(T)**2))
    """
    return 0.5 * (theta + rho * psi * k + (
            (psi * k + theta * rho) ** 2 +
            theta ** 2 * (1 - rho ** 2)
    ) ** (1 / 2))


# @nb.jit(nopython=True, fastmath=True)
def f_essvi_iv(k: NDArray[np.float64], theta: float, rho: float, psi: float, tenor: float):
    """
    Implied total variance surface
    eSSVI(K, T) = 1/2 ( θ(T) + ρ(T)ψ(T)k + sqrt((ψ(T)k + θ(T)ρ(T))**2 + θ(T)**2 * (1 − ρ(T)**2))
    """
    return (f_essvi_total_variance(k, theta, rho, psi) / tenor) ** (1 / 2)


# @nb.njit(fastmath=True)
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


_F_MIN_CACHE: dict[bytes, np.ndarray] = {}
_F_MIN_CACHE_MAX = 1024**3

def _cache_get(key: bytes):
    return _F_MIN_CACHE.get(key)


def _cache_put(key: bytes, value: np.ndarray):
    if len(_F_MIN_CACHE) >= _F_MIN_CACHE_MAX:
        # FIFO-ish eviction (good enough; replace with LRU if needed)
        _F_MIN_CACHE.pop(next(iter(_F_MIN_CACHE)))
    _F_MIN_CACHE[key] = value

def jac_price_surface_theta_rho_ps(
        calibration_params: List[float],
        equity: Equity,
        bids, asks, s, k, t, v_is_call,
        zero_rates: ZeroCurveData,
        tenor_offsets,
        dividends,
        calculation_date,
        diff_step: float,
):
    """
    Central-difference (3-point) Jacobian using one GPU batch call.

    Builds scenarios:
      - base: x
      - for each param j: x(+h_j), x(-h_j)

    Then:
      J[:, j] = (r(x+h_j) - r(x-h_j)) / (2*h_j)
    """
    from options.types.option import get_price_cuda
    x = np.asarray(calibration_params, dtype=np.float64)
    n_params = x.size
    n_opt = len(t)

    # Scenario matrix: base + (plus, minus) per parameter => 1 + 2*n_params
    n_scenarios = 1 + 2 * n_params
    X = np.tile(x, (n_scenarios, 1))

    # Step size per parameter (h_j)
    h = np.empty(n_params, dtype=np.float64)
    for j in range(n_params):
        xj = x[j]
        h[j] = (abs(xj) * diff_step) if xj != 0.0 else diff_step
        X[1 + 2 * j, j] = xj + h[j]      # +h
        X[1 + 2 * j + 1, j] = xj - h[j]  # -h

    # Build model_iv for every scenario (CPU-side assembly; GPU does pricing in one batch)
    model_iv_all = np.empty((n_scenarios, n_opt), dtype=np.float64)

    for p_ix in range(n_scenarios):
        params = X[p_ix]

        v_theta = np.empty(n_opt, dtype=np.float64)
        v_rho = np.empty(n_opt, dtype=np.float64)
        v_psi = np.empty(n_opt, dtype=np.float64)

        for i, (start, end) in enumerate(tenor_offsets):
            v_theta[start:end] = params[i * 3 + 0]
            v_rho[start:end] = params[i * 3 + 1]
            v_psi[start:end] = params[i * 3 + 2]

        mny_fwd_ln = np.log(get_moneyness_fwd(equity, k, s, t, date_to_sod(calculation_date), yield_curve_in=zero_rates))
        variance = f_essvi_total_variance(mny_fwd_ln, v_theta, v_rho, v_psi) / t
        variance[(variance < 0) | np.isnan(variance)] = 0  # essentially also 0 IV. becomes intrinics value but without the invalid sqrt error warning
        model_iv_all[p_ix] = np.sqrt(variance)
        # impute np.nan with zero - bad assumption that we dont get infinitely high IVs from above, but rather just negative / nan.
        model_iv_all[np.isnan(model_iv_all)] = 0  # returns intrinsic price...

    # Tile option inputs to match the flattened iv vector
    s_all = np.tile(np.asarray(s, dtype=np.float64), n_scenarios)
    k_all = np.tile(np.asarray(k, dtype=np.float64), n_scenarios)
    t_all = np.tile(np.asarray(t, dtype=np.float64), n_scenarios)
    v_is_call_all = np.tile(np.asarray(v_is_call, dtype=np.int32), n_scenarios)

    prices_all = get_price_cuda(
        s=s_all,
        k=k_all,
        t=t_all,
        v_is_call=v_is_call_all,
        iv=model_iv_all.reshape(-1),
        dividends=dividends,
        calculation_date=calculation_date,
        yield_curve=zero_rates,
        cpu=False
    )
    model_prices = np.asarray(prices_all, dtype=np.float64).reshape((n_scenarios, n_opt))

    # residuals_all = prices_all - np.asarray(item_prices, dtype=np.float64)[None, :]
    residuals_all, _, _, _ = logistic_spread_weighted_residuals(model_prices, bids, asks)

    # central-difference residuals per parameter
    r_plus = residuals_all[1::2]   # shape: (n_params, n_opt)
    r_minus = residuals_all[2::2]  # shape: (n_params, n_opt)

    # J is (m, n): m = n_opt residuals, n = n_params
    return (r_plus - r_minus).T / (2.0 * h[None, :])

def _key_from_x(x: NDArray) -> bytes:
    x = np.asarray(x, dtype=np.float64)
    return x.tobytes()

def f_min_price_surface_theta_rho_psi(
        calibration_params: List[float],
        equity: Equity,
        bids, asks, s, k, t, v_is_call, zero_rates: ZeroCurveData, tenor_offsets: List[Tuple[int, int]],
        dividends, calculation_date, diff_step) -> NDArray[float]:
    from options.types.option import get_price_cuda

    n = len(t)
    v_theta = np.empty(n, dtype=np.float64)
    v_rho = np.empty(n, dtype=np.float64)
    v_psi = np.empty(n, dtype=np.float64)

    for i, (start, end) in enumerate(tenor_offsets):
        v_theta[start:end] = calibration_params[i * 3 + 0]
        v_rho[start:end] = calibration_params[i * 3 + 1]
        v_psi[start:end] = calibration_params[i * 3 + 2]

    mny_fwd_ln = np.log(get_moneyness_fwd(equity, k, s, t, date_to_sod(calculation_date), zero_rates))

    # if any((el < 0 for el in f_essvi_total_variance(mny_fwd_ln, v_theta, v_rho, v_psi) / t)):
    #     pass
    v_variance = f_essvi_total_variance(mny_fwd_ln, v_theta, v_rho, v_psi) / t
    v_variance[(v_variance < 0) | np.isnan(v_variance)] = 0  # Becomes 0 IV -> intrinsic value price. Avoids warning "invalid value encountered in sqrt"
    model_iv = np.sqrt(v_variance)
    # impute np.nan with zero - bad assumption that we don't get infinitely high IVs from above, but rather just negative / nan.
    model_iv[np.isnan(model_iv)] = 0  # returns intrinsic price...

    # Here, a small batch size is processed only, CPU faster
    model_prices = get_price_cuda(
        s=s,
        k=k,
        t=t,
        v_is_call=v_is_call,
        iv=model_iv,
        dividends=dividends,
        calculation_date=calculation_date,
        yield_curve=zero_rates,
        cpu=len(s) < 100
    )

    residuals, _, _, _ = logistic_spread_weighted_residuals(model_prices, bids, asks)

    # if item.weights is not None:
    #     residuals *= item.weights

    ix_nan = np.isnan(residuals) | np.isinf(residuals)
    if ix_nan.any():
        info(f'# nan: {ix_nan.sum()}')

    return residuals

def logistic_spread_weighted_residuals(
    model_prices: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    w_min: float = 0.05,   # penalty factor at mid (u=0)
    alpha: float = 15.0,   # steepness of transition
    eps: float = 0.01,     # want w(u=1) = 1 - eps (almost 1 at bid/ask boundary)
    min_half_spread: float = 1e-12
):
    """
    Continuous residuals that downweight errors inside bid/ask, and are ~fully weighted at the boundary.

    u = |p_model - mid| / half_spread
    w(u) = w_min + (1-w_min) * sigmoid(alpha*(u-beta))
    beta chosen so that w(1) = 1 - eps

    Returns
    -------
    residuals : np.ndarray
        weighted residuals r = (model - mid) * w(u)
    weights : np.ndarray
        w(u)
    u : np.ndarray
        normalized distance to mid
    beta : float
        logistic shift used
    """
    model_prices = np.asarray(model_prices, dtype=float)
    bid = np.asarray(bids, dtype=float)
    ask = np.asarray(asks, dtype=float)

    mid = 0.5 * (bid + ask)
    half_spread = np.maximum(0.5 * (ask - bid), min_half_spread)

    e = model_prices - mid
    u = np.abs(e) / half_spread

    # Enforce w(1) = 1 - eps by solving for beta
    # s_star = ((1-eps) - w_min) / (1-w_min) must be in (0,1)
    s_star = ((1.0 - eps) - w_min) / (1.0 - w_min)
    s_star = np.clip(s_star, 1e-12, 1.0 - 1e-12)  # numerical safety
    logit_s = np.log(s_star / (1.0 - s_star))
    beta = 1.0 - (logit_s / alpha)

    # sigmoid
    z = alpha * (u - beta)
    w = w_min + (1.0 - w_min) * (1.0 / (1.0 + np.exp(-z)))

    residuals = e * w
    return residuals, w, u, beta

# @nb.njit(fastmath=True)
def fill_nan_residuals(residuals: np.ndarray) -> np.ndarray:
    ix_nan = np.isnan(residuals) | np.isinf(residuals)
    res_max = np.abs(residuals[~ix_nan])
    if len(res_max) > 0:
        residuals[ix_nan] = np.max(res_max) * 10
    else:
        residuals[ix_nan] = 1e3
    return residuals


# @nb.njit(fastmath=True)
def f_gj(theta: float, rho: float):
    return 4 * theta / (1 + abs(rho))


# @nb.njit(fastmath=True)
def convert_rho_b_c2theta_rho_psi(model_params: NDArray[np.float64], n_tenors: int) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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


def f_min_reparameterized(model_params: NDArray[np.float64], samples: Dict[date, CalibrationItem], get_price: Callable, n_residuals: int):
    v_residuals = np.empty(n_residuals)
    v_theta, v_rho, v_psi = convert_rho_b_c2theta_rho_psi(model_params, len(samples))
    nan_encountered = []

    ix = 0
    # No need to loop over tenor - get_price can handle vector of tenors.
    for i, (tenor_dt, item) in enumerate(samples.items()):
        model_variance = f_essvi_total_variance(item.mny_fwd_ln, v_theta[i], v_rho[i], v_psi[i])
        model_iv = (model_variance / item.tenor) ** 0.5

        v_is_call = np.array([1 if r == OptionRight.call else 0 for r in item.rights])
        model_prices = get_price(item.spot, item.strike, [item.tenor]*len(item.strike), v_is_call, model_iv, item.rf, item.dividends, item.calculation_date)
        residuals = model_prices - item.price

        if item.weights is not None:
            residuals *= item.weights

        if np.isnan(residuals).any() or np.isinf(residuals).any():
            nan_encountered.append(tenor_dt)
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
    tenor_dt: date
    tenor: float
    model_param: MetricSSVI | str
    value: float
    ts_end: datetime


class IVSurface:
    """
    Enrich a frame with index ts, expiry, strike, right. So plotting is easy.
    """
    def __init__(self, underlying: Equity, tag: str = ''):
        self.id = uuid4()
        self.underlying = underlying
        self.is_calibrated = False
        self.calibration_items: Dict[date, CalibrationItem] = {}
        self.params: SSVISurfParams = SSVISurfParams()
        self.rmse_iv: Dict[date, Tuple] = {}
        self.evaluation: IVSurfaceModelEvaluation = None
        self.tag = tag
        self._last_calibration_ts: datetime = None
        self.v_calibrated_params: NDArray = None

    def __repr__(self):
        return f'IVS_SSVI_{self.underlying.symbol}_{self.tag}{"_" + self.last_calibration_ts().isoformat() if self._last_calibration_ts else ''}'

    def calibrate(self, calibration_items: List[CalibrationItem], verbose=1, initial_params=None, plot_cost_gt=10):
        initial_params = initial_params or [0.2, -0.1, 2]

        for item in calibration_items:

            total_variance = item.iv ** 2 * item.tenor
            fit_res = least_squares(f_min, initial_params, args=(item.mny_fwd_ln, total_variance, f_essvi_total_variance, item.weights), verbose=verbose, max_nfev=1000)

            self.calibration_items[item.tenor_dt] = item
            self.params[item.tenor_dt] = SSVITenorParams(*fit_res.x)
            self.rmse_iv[item.tenor_dt] = fit_res.cost

            if fit_res.cost > plot_cost_gt:
                warning(f'Calibration failed for {self.underlying} tenor={item.tenor_dt} cost: {fit_res.cost}, calcDate={item.calculation_date}')

        info(f'Calibration done for {self.underlying} rmse(iv)={np.mean(list(self.rmse_iv.values()))}')
        self.is_calibrated = True

        return self

    def set_params(self, params: SSVISurfParams):
        self.params = params
        self.is_calibrated = True
        return self

    @lru_cache(maxsize=1)
    def n_params(self):
        return len(self.params)

    def is_calibrated_slice(self, tenor_dt: date):
        return tenor_dt in self.params is not None

    def set_calibration_items(self, calibration_items: List[CalibrationItem]):
        for item in calibration_items:
            if item.tenor_dt in self.params:
                self.calibration_items[item.tenor_dt] = item

    def ps_rmse(self) -> pd.Series:
        return pd.Series(self.rmse_iv)

    def plot_all_prices(self, calc_date: date, calibration_items_bid=None, calibration_items_ask=None, fn=None, open_browser=True):
        for tenor_dt in self.calibration_items.keys():
            self.plot_prices(tenor_dt, calc_date, calibration_items_bid, calibration_items_ask, fn=fn, open_browser=open_browser)

    def plot_prices(self, tenor_dt: date, calc_date: date, calibration_items_bid=None, calibration_items_ask=None, fn=None, open_browser=True):
        from options.types.option import get_price_cuda
        dividends = self.calibration_items[tenor_dt].dividends

        params = self.params[tenor_dt]
        item = self.calibration_items[tenor_dt]
        v_mny_fwd_ln = item.mny_fwd_ln
        v_strike = item.strike
        tenor = get_tenor(tenor_dt, calc_date)

        v_spot = item.spot
        v_vega = item.vega
        v_weights = item.weights
        y_pred = f_essvi_iv(v_mny_fwd_ln, *params, tenor=get_tenor(tenor_dt, calc_date))

        fig = make_subplots(rows=3, cols=1, subplot_titles=[f'Prices {self.underlying} expiry: {tenor_dt}', 'Weights', 'Sample Count Hist'])
        fig.add_trace(go.Scatter(x=v_mny_fwd_ln, y=item.price, mode='markers', marker=dict(size=4), name='price'), row=1, col=1)

        # Quotes if any
        if calibration_items_bid is not None and (i_bid := next(iter([i for i in calibration_items_bid if i.tenor_dt == tenor_dt]), None)):
            ix_sample = pd.Series(range(len(i_bid.iv))).sample(min(len(i_bid.iv), 500)).values
            x = i_bid.mny_fwd_ln[ix_sample]

            fig.add_trace(go.Scatter(x=x, y=i_bid.price[ix_sample], mode='markers', marker=dict(size=2), name='price_bid'), row=1, col=1)
            if calibration_items_ask is not None and (i_ask := next(iter([i for i in calibration_items_ask if i.tenor_dt == tenor_dt]), None)):
                fig.add_trace(go.Scatter(x=x, y=i_ask.price[ix_sample], mode='markers', marker=dict(size=2), name='price_ask'), row=1, col=1)

        y_price_pred = get_price_cuda(s=v_spot, k=np.array(v_strike), t=np.asarray([tenor]*len(v_spot)), iv=y_pred, dividends=dividends, calculation_date=calc_date, equity=self.underlying)
        fig.add_trace(go.Scatter(x=v_mny_fwd_ln, y=y_price_pred, mode='markers', marker=dict(size=4), name='price_pred'), row=1, col=1)
        fig.add_trace(go.Scatter(x=v_mny_fwd_ln, y=v_weights, mode='markers', marker=dict(size=4), name='weights'), row=2, col=1)
        # if v_vega is not None:
        #     fig.add_trace(go.Scatter(x=v_mny_fwd_ln, y=v_vega, mode='markers', marker=dict(size=4), name='vega'), row=3, col=1)
        fig.add_trace(go.Histogram(x=v_mny_fwd_ln, nbinsx=int((max(v_mny_fwd_ln) - min(v_mny_fwd_ln)) * 100), name='count'), row=3, col=1)
        show(fig, fn=fn or f'essvi_{self.underlying}_{tenor_dt}.html', open_browser=open_browser)

    def plot_surface_calibration_items(self, fn=None, open_browser=True, samples_by_tenor=None):
        from options.types.option import get_v_iv
        fig = go.Figure()
        v_mny = []
        v_tenor = []
        v_iv = []
        for tenor_dt, v in self.calibration_items.items():
            tenor = get_tenor(tenor_dt, v.calculation_date)
            v_mny += list(v.mny_fwd_ln)
            v_tenor += [tenor] * len(v.mny_fwd_ln)

            v_iv += get_v_iv(
                p=v.price,
                s=v.spot,
                k=v.strike.astype(float),
                t=np.array([tenor] * len(v.mny_fwd_ln)),
                dividends=get_dividends(self.underlying.symbol.upper(), v.calculation_date),
                calculation_date=v.calculation_date,
            ).tolist()

        fig.add_trace(go.Scatter3d(x=v_mny, y=v_tenor, z=v_iv, mode='markers', marker=dict(size=2), name=f'IV'))

        fn_plot = fn or f'raw_{self.underlying}_IV_surface.html'
        fig.update_layout(title=fn_plot, autosize=True, scene=dict(
            xaxis_title='Log forward moneyness',
            yaxis_title='Tenor',
            zaxis_title='Implied Volatility', ),
                          )
        show(fig, fn=fn_plot, open_browser=open_browser)


    def plot_surface_vanilla(self, calc_date1=date(2024, 3, 11), expiries: List[date] = None) -> go.Figure:
        fig = go.Figure()
        x = np.linspace(-0.2, 0.2, 100)
        y = []
        z = []
        _expiries = expiries or sorted(self.params.keys())
        for tenor_dt in _expiries:
            z += list(self.iv(x, tenor_dt, calc_date1))
            y += [get_tenor(tenor_dt, calc_date1)] * len(x)
        fig.add_trace(go.Scatter3d(x=x.tolist() * len(_expiries), y=y, z=z, mode='markers', marker=dict(size=2)))
        return fig

    def plot_surface_as_of(self, calc_date: date, fn=None, open_browser=True):
        fig = go.Figure()
        v_mny = []
        v_tenor = []
        v_iv = []
        for tenor_dt, params in self.params.items():
            x = np.linspace(-0.3, 0.3, 100)
            v_mny += list(x)
            v_iv += list(f_essvi_iv(x, *params, tenor=get_tenor(tenor_dt, calc_date)))
            v_tenor += [get_tenor(tenor_dt, calc_date)] * len(x)

        fig.add_trace(go.Scatter3d(x=v_mny, y=v_tenor, z=v_iv, mode='markers', marker=dict(size=2), name=f'IV'))

        fn_plot = fn or f'essvi_{self.underlying}_IV_surface.html'
        fig.update_layout(title=fn_plot, autosize=True, scene=dict(
            xaxis_title='Log forward moneyness',
            yaxis_title='Tenor',
            zaxis_title='Implied Volatility', ),
                          )
        show(fig, fn=fn_plot, open_browser=open_browser)


    def plot_model_params(self, calc_date: date, fn=None, open_browser=True):
        """Plot params and errors for each right across the surface."""
        subs = [f'{metric}' for metric in ['Theta', 'Rho', 'Psi']]
        fig = make_subplots(rows=3, cols=1, subplot_titles=subs)
        p_dct = {k: v for k, v in self.params.items()}
        x = [get_tenor(k, calc_date) for k in p_dct.keys()]
        theta = [v.theta for k, v in p_dct.items()]
        rho = [v.rho for k, v in p_dct.items()]
        psi = [v.psi for k, v in p_dct.items()]
        fig.add_trace(go.Scatter(x=x, y=theta, mode='markers', marker=dict(size=4), name=f'Theta'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=rho, mode='markers', marker=dict(size=4), name=f'Rho'), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=psi, mode='markers', marker=dict(size=4), name=f'Psi'), row=3, col=1)

        show(fig, fn=fn or f'essvi_{self.underlying}_model_params_{time.time()}.html', open_browser=open_browser)

    # def nlv(self, o: Option, s0, q: float) -> float:
    #     calc_date = self.first_calibration_calc_date()
    #
    #     iv = self.iv(get_moneyness_fwd_ln(self.underlying, strike: float | NDArray[np.float64], spot: float | NDArray[np.float64], tenor: float | NDArray[np.float64]))
    #     if iv is None or np.isnan(iv):
    #         return None
    #     o.npv(iv, s0, calc_date) * q * 100

    def iv(self, moneyness_fwd_ln: float | NDArray[np.float64], tenor: date, calc_date: datetime) -> NDArray[np.float64] | float:
        """
        Interpolating IV from the frame.
        """
        if not self.is_calibrated:
            raise ValueError('IVSurface is not calibrated')
        try:
            return f_essvi_iv(moneyness_fwd_ln, *self.params[tenor], tenor=get_tenor(tenor, calc_date))
        except KeyError as e:
            # Could interpolate the surface. But probably should rather ifx something in the calibration.
            raise e

    def iv_jump_right(self, calc_date: date = None, release_date: date = None, min_days=5, tenor_dt1_dt2: Tuple[date, date] = None) -> float:
        """
        iv_jump**2 = (iv_t1**2 - iv_t2**2) / (1/t_1 - 1/t_2)
        """
        _calc_date = calc_date or self.last_calibration_ts().date()
        _release_date = release_date or self.next_release_date(_calc_date)

        if tenor_dt1_dt2 is None:
            tenor_dt1 = None
            tenor_dt2 = None
            for dt in self.tenors():
                if not dt in self.params:
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
            iv_t1 = self.iv(0, tenor_dt1, _calc_date)
            iv_t2 = self.iv(0, tenor_dt2, _calc_date)  # tenor available for calls, but not puts...

            return (iv_t1 ** 2 - iv_t2 ** 2) / (1 / tenor_t1 - 1 / tenor_t2) ** 0.5
        except KeyError as e:
            return np.nan

    def iv_jump(self, calc_date: date = None, release_date: date = None):
        _calc_date = calc_date or self.last_calibration_ts().date()
        _release_date = release_date or self.next_release_date(_calc_date)
        return self.iv_jump_right(_calc_date, _release_date)

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
                    dct[f'iv_jump_{i}_{i+k+1}'] = self.iv_jump_right(_calc_date, _release_date, tenor_dt1_dt2=(tenor_dt, match_tenor))
                except KeyError:
                    pass
        return dct

    def tenors(self) -> List[date]:
        return sorted(list(set([k for k in self.params.keys()])))

    def next_release_date(self, calc_date: date) -> date:
        return next(iter([d for d in EarningsPreSessionDates(self.underlying.symbol) if d >= calc_date]), date.max)

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

    @timer
    def calibrate_surface_py(
            self,
            equity: Equity,
            samples: Dict[date, CalibrationItem],
            calc_date: date,
            verbose: int = 1, max_nfev=1000,
            x0_in: Dict[date, List[float]] = None,
            min_max_mny_fwd_ln = (-0.3, 0.3)
    ):
        """
        Optimize by passing good x0 starting conditions
        """
        self._validate_calibration_samples(samples)
        t0 = time.time()
        if not samples:
            warning(f'No samples for {self.underlying}')
            return self

        zero_rates = YieldCurve().get_last_zero_curve(calc_date, equity, -14)
        tenors = sorted(samples.keys())
        n_tenors = len(tenors)
        x0 = {t: x0_in.get(t, [0.05, -0.1, 0.1]) for t in tenors}

        calibration_params = [x for d in sorted(x0) for x in x0[d]]
        # theta (ATM impl var), rho (ATM slope), psi (ATM curvature)

        vals = samples.values()
        mny_fwd_ln = np.array(list(chain(*[i.mny_fwd_ln for i in vals])))
        ix_mny = (min_max_mny_fwd_ln[0] < mny_fwd_ln) & (mny_fwd_ln < min_max_mny_fwd_ln[1])

        bids = np.array(list(chain(*[i.bid_price for i in vals])))[ix_mny]
        asks = np.array(list(chain(*[i.ask_price for i in vals])))[ix_mny]
        # item_prices = np.array(list(chain(*[i.price for i in vals])))[ix_mny]
        s = np.array(list(chain(*[i.spot for i in vals])))[ix_mny]
        k = np.array(list(chain(*[i.strike for i in vals])))[ix_mny]
        t = np.array(list(chain(*[[i.tenor] * len(i.strike) for i in vals])))[ix_mny]
        v_is_call = np.array(list(chain(*[[1 if r == OptionRight.call else 0 for r in i.rights] for i in vals])))[ix_mny]

        tenor_index = np.array(list(chain(*[[i]*len(el.strike) for i, el in enumerate(vals)])))[ix_mny]
        n = len(tenor_index)
        tenor_offsets = []
        for i in range(n_tenors):
            bool_ix = tenor_index == i
            tenor_offsets.append((np.argmax(bool_ix), n - np.argmax(bool_ix[::-1])))

        dividends = next((i.dividends for i in vals if i.dividends is not None), [])
        calculation_date: date = next((i.calculation_date for i in vals if i.calculation_date is not None), None)

        diff_step = 0.01
        try:
            fit_res = least_squares(
                fun=f_min_price_surface_theta_rho_psi,
                x0=calibration_params,
                jac=jac_price_surface_theta_rho_ps,
                args=(equity, bids, asks, s, k, t, v_is_call, zero_rates, tenor_offsets, dividends, calculation_date, diff_step),
                verbose=verbose,
                max_nfev=max_nfev,
                diff_step=diff_step,  # None -> super tiny 1e-6 or sth value. Small IV, change, small price. too tiny epsilon wouldnt reflect in residuals.
                # loss='huber',
                bounds=([0, -np.inf, -np.inf] * n_tenors, [np.inf, np.inf, np.inf] * n_tenors)  # ensure theta > 0
            )
        except ValueError as ex:
            pass
        info(f'Calibrated {self.underlying}. # tenors: {n_tenors}. # samples: {len(s)}. # params: {len(calibration_params)} max_nfev={max_nfev} in {time.time() - t0:.2f}sec. Had x0={len(x0_in or {}) > 0}')

        self.v_calibrated_params = fit_res.x

        for i, (tenor_dt, item) in enumerate(samples.items()):
            theta = fit_res.x[i * 3]
            rho = fit_res.x[i * 3 + 1]
            psi = fit_res.x[i * 3 + 2]

            self.calibration_items[item.tenor_dt] = item
            self.params[item.tenor_dt] = SSVITenorParams(theta, rho, psi)

        self.is_calibrated = True
        # pprint(self.evaluate())

        return self

    @timer
    def calibrate_surface(
            self,
            equity: Equity,
            samples: Dict[date, CalibrationItem],
            calc_date: date,
            verbose: int = 0,
            max_nfev = 1000,
            x0_in: Dict[date, List[float]] = None,
            min_max_mny_fwd_ln = (-0.3, 0.3)
            ):
        """
        Optimize by passing good x0 starting conditions
        """
        self._validate_calibration_samples(samples)
        t0 = time.time()
        if not samples:
            warning(f'No samples for {self.underlying}')
            return self

        zero_rates = YieldCurve().get_last_zero_curve(calc_date, equity, -14)
        tenors = sorted(samples.keys())
        n_tenors = len(tenors)
        x0 = {t: x0_in.get(t, [0.05, -0.1, 0.1]) for t in tenors}

        calibration_params = [x for d in sorted(x0) for x in x0[d]]
        # theta (ATM impl var), rho (ATM slope), psi (ATM curvature)

        vals = samples.values()
        mny_fwd_ln = np.array(list(chain(*[i.mny_fwd_ln for i in vals])))
        ix_mny = (min_max_mny_fwd_ln[0] < mny_fwd_ln) & (mny_fwd_ln < min_max_mny_fwd_ln[1])

        bids = np.array(list(chain(*[i.bid_price for i in vals])))[ix_mny]
        asks = np.array(list(chain(*[i.ask_price for i in vals])))[ix_mny]
        # item_prices = np.array(list(chain(*[i.price for i in vals])))[ix_mny]
        s = np.array(list(chain(*[i.spot for i in vals])))[ix_mny]
        k = np.array(list(chain(*[i.strike for i in vals])))[ix_mny]
        t = np.array(list(chain(*[[i.tenor] * len(i.strike) for i in vals])))[ix_mny]
        v_is_call = np.array(list(chain(*[[1 if r == OptionRight.call else 0 for r in i.rights] for i in vals])))[ix_mny]

        tenor_index = np.array(list(chain(*[[i]*len(el.strike) for i, el in enumerate(vals)])))[ix_mny]
        n = len(tenor_index)
        tenor_offsets = []
        for i in range(n_tenors):
            bool_ix = tenor_index == i
            tenor_offsets.append((np.argmax(bool_ix), n - np.argmax(bool_ix[::-1])))

        dividends = next((i.dividends for i in vals if i.dividends is not None), [])
        calculation_date: date = next((i.calculation_date for i in vals if i.calculation_date is not None), None)

        diff_step = 0.01
        div_amounts, div_times = dividends2amount_times(dividends, calculation_date)
        mny_fwd_ln = np.log(get_moneyness_fwd(equity, k, s, t, date_to_sod(calculation_date), zero_rates))

        # for v in [bids, asks, s, k, t, v_is_call, mny_fwd_ln]:
        #     print(f"# Missing {np.isnan(v).sum()})")
        #     print(f"Len {len(v)})")
        #     print(f"Sum: {sum(v)})")

        res, _ = jl.Fino.CalibrateIVS.calibrate_surface(
            jl.Vector[jl.Float32](np.array(calibration_params, dtype=np.float32)),
            jl.Vector[jl.Float32](bids.astype(np.float32)),
            jl.Vector[jl.Float32](asks.astype(np.float32)),
            jl.Vector[jl.Float32](s.astype(np.float32)),
            jl.Vector[jl.Float32](k.astype(np.float32)),
            jl.Vector[jl.Float32](t.astype(np.float32)),
            jl.Vector[jl.UInt8]((v_is_call != 0).astype(np.uint8)),
            jl.Vector[jl.Float32](mny_fwd_ln.astype(np.float32)),
            jl.Vector[jl.Tuple[jl.Int64, jl.Int64]](tenor_offsets),
            jl.Vector[jl.Float32](np.array(zero_rates.rates, dtype=np.float32)),
            jl.Vector[jl.Float32](np.array(zero_rates.times, dtype=np.float32)),
            jl.Vector[jl.Float32](np.array(div_amounts, dtype=np.float32)),
            jl.Vector[jl.Float32](np.array(div_times, dtype=np.float32)),
            diff_step=np.float32(diff_step),
            max_nfev=1000,
            verbose=verbose == 1
        )
        res = np.array(res, dtype=np.float32)
        info(f'Jl calibrated {self.underlying}. # tenors: {n_tenors}. # samples: {len(s)}. # params: {len(calibration_params)} max_nfev={max_nfev} in {time.time() - t0:.2f}sec. Had x0={len(x0_in or {}) > 0}')

        self.v_calibrated_params = res

        for i, (tenor_dt, item) in enumerate(samples.items()):
            theta = res[i * 3]
            rho = res[i * 3 + 1]
            psi = res[i * 3 + 2]

            self.calibration_items[item.tenor_dt] = item
            self.params[item.tenor_dt] = SSVITenorParams(theta, rho, psi)

        self.is_calibrated = True
        # pprint(self.evaluate())

        # IVSurface("Tst").set_last_calibration_ts(calculation_date).set_params(self.params).plot_surface_vanilla(calculation_date)
        # plot_calibration_surface_debug(
        #     bids, asks, mny_fwd_ln, t, v_is_call, tenor_offsets, tenors,
        #     self.params,
        #     title=f"{equity.symbol} — before calibration",
        #     fn=f"debug_{equity.symbol}_before.html",
        # )

        return self

    def evaluate(self, mny_bin_size=0.05, min_max_mny_fwd_ln=(-0.3, 0.3)) -> IVSurfaceModelEvaluation | None:
        from options.types.option import get_price_cuda
        if not self.is_calibrated:
            return None

        dct_iv = {}
        dct_iv_pred = {}
        dct_price = {}
        dct_bid_price = {}
        dct_ask_price = {}
        dct_price_pred = {}
        dct_spread = {}

        for tenor_dt, item in self.calibration_items.items():
            theta, rho, psi = self.params[tenor_dt]

            ix_scoped_mny = (min_max_mny_fwd_ln[0] < item.mny_fwd_ln) & (item.mny_fwd_ln < min_max_mny_fwd_ln[1])

            for right in [OptionRight.call, OptionRight.put]:
                ix = (item.rights == right) & ix_scoped_mny
                if not np.any(ix):
                    continue

                mny = item.mny_fwd_ln[ix]
                iv_obs = item.iv[ix]
                iv_pred = f_essvi_iv(mny, theta, rho, psi, tenor=item.tenor)

                dct_iv[(tenor_dt, right)] = iv_obs
                dct_iv_pred[(tenor_dt, right)] = iv_pred

                dct_price[(tenor_dt, right)] = item.price[ix]
                dct_bid_price[(tenor_dt, right)] = item.bid_price[ix]
                dct_ask_price[(tenor_dt, right)] = item.ask_price[ix]

                dct_price_pred[(tenor_dt, right)] = get_price_cuda(
                    s=item.spot[ix],
                    k=item.strike[ix],
                    t=np.asarray([item.tenor] * int(np.sum(ix))),
                    iv=iv_pred,
                    v_is_call=np.asarray([1 if right == OptionRight.call else 0] * int(np.sum(ix))),
                    dividends=item.dividends,
                    calculation_date=item.calculation_date,
                    equity=self.underlying,
                    yield_curve=YieldCurve().get_last_zero_curve(item.calculation_date, self.underlying, -14),
                )

                if item.bid_price is not None and item.ask_price is not None:
                    dct_spread[(tenor_dt, right)] = (item.ask_price[ix] - item.bid_price[ix])

        dct_iv_by_mny = {}
        dct_iv_pred_by_mny = {}
        dct_price_by_mny = {}
        dct_bid_price_by_mny = {}
        dct_ask_price_by_mny = {}
        dct_price_pred_by_mny = {}
        dct_spread_by_mny = {}

        for tenor_dt, item in self.calibration_items.items():
            theta, rho, psi = self.params[tenor_dt]
            ix_scoped_mny = (min_max_mny_fwd_ln[0] < item.mny_fwd_ln) & (item.mny_fwd_ln < min_max_mny_fwd_ln[1])

            for right in [OptionRight.call, OptionRight.put]:
                ix = (item.rights == right) & ix_scoped_mny
                if not np.any(ix):
                    continue

                mny = item.mny_fwd_ln[ix]
                iv_obs = item.iv[ix]
                price_obs = item.price[ix]
                bid_price_obs = item.bid_price[ix]
                ask_price_obs = item.ask_price[ix]
                spot = item.spot[ix]
                strike = item.strike[ix]

                for bucket_ix, _ in groupby(mny, lambda x: x // mny_bin_size):
                    in_bucket = (mny // mny_bin_size) == bucket_ix
                    if not np.any(in_bucket):
                        continue

                    key = (tenor_dt, right, bucket_ix * mny_bin_size)

                    dct_iv_by_mny[key] = iv_obs[in_bucket]
                    dct_iv_pred_by_mny[key] = f_essvi_iv(mny[in_bucket], theta, rho, psi, tenor=item.tenor)

                    dct_price_by_mny[key] = price_obs[in_bucket]
                    dct_bid_price_by_mny[key] = bid_price_obs[in_bucket]
                    dct_ask_price_by_mny[key] = ask_price_obs[in_bucket]

                    dct_price_pred_by_mny[key] = get_price_cuda(
                        s=spot[in_bucket],
                        k=strike[in_bucket],
                        t=np.asarray([item.tenor] * int(np.sum(in_bucket))),
                        iv=dct_iv_pred_by_mny[key],
                        v_is_call=np.asarray([1 if right == OptionRight.call else 0] * int(np.sum(in_bucket))),
                        dividends=item.dividends,
                        calculation_date=item.calculation_date,
                        equity=self.underlying,
                        yield_curve=YieldCurve().get_last_zero_curve(calculation_date=item.calculation_date, equity=self.underlying, delta_days=-14)
                    )

                    if item.bid_price is not None and item.ask_price is not None:
                        dct_spread_by_mny[key] = (item.ask_price[ix][in_bucket] - item.bid_price[ix][in_bucket])

        self.rmse_iv = {k: rmse(v, dct_iv_pred[k]) for k, v in dct_iv.items()}

        self.evaluation = IVSurfaceModelEvaluation(
            rmse_iv=rmse(list(chain(*dct_iv.values())), list(chain(*dct_iv_pred.values()))),
            rmse_iv_right_tenor={k: rmse(v, dct_iv_pred[k]) for k, v in dct_iv.items()},
            rmse_price=rmse(list(chain(*dct_price.values())), list(chain(*dct_price_pred.values()))),
            rmse_price_right_tenor={k: rmse(v, dct_price_pred[k]) for k, v in dct_price.items()},
            rmse_err_over_spread=rmse(
                0,
                (np.array(list(chain(*dct_price_pred.values()))) - np.array(list(chain(*dct_price.values()))))
                / np.array(list(chain(*dct_spread.values())))
            ),
            rmse_err_over_spread_right_tenor={k: rmse(0, (dct_price_pred[k] - dct_price[k]) / dct_spread[k]) for k, v in dct_spread.items()},

            mae_iv=meanabs(list(chain(*dct_iv.values())), list(chain(*dct_iv_pred.values()))),
            mae_iv_right_tenor={k: meanabs(v, dct_iv_pred[k]) for k, v in dct_iv.items()},
            mae_price=meanabs(list(chain(*dct_price.values())), list(chain(*dct_price_pred.values()))),
            mae_price_right_tenor={k: meanabs(v, dct_price_pred[k]) for k, v in dct_price.items()},
            logit_err_price=np.mean(abs(logistic_spread_weighted_residuals(
                np.array(list(chain(*dct_price.values()))),
                np.array(list(chain(*dct_bid_price_by_mny.values()))),
                np.array(list(chain(*dct_ask_price_by_mny.values()))),
            )[0])),
            logit_err_price_right_tenor={k: np.mean(abs(logistic_spread_weighted_residuals(v, dct_bid_price[k], dct_ask_price[k],)[0])) for k, v in dct_price_pred.items()},
            mae_err_over_spread=meanabs(
                0,
                (np.array(list(chain(*dct_price_pred.values()))) - np.array(list(chain(*dct_price.values()))))
                / np.array(list(chain(*dct_spread.values())))
            ),
            mae_err_over_spread_right_tenor={k: meanabs(0, (dct_price_pred[k] - dct_price[k]) / dct_spread[k]) for k, v in dct_spread.items()},
            me_iv_by_mny={k: np.mean(v - dct_iv_pred_by_mny[k]) for k, v in dct_iv_by_mny.items()},
            me_price_by_mny={k: np.mean(v - dct_price_pred_by_mny[k]) for k, v in dct_price_by_mny.items()},
        )
        return self.evaluation

    def plot_error_surface(self, metric: Literal['iv', 'usd'], error: Literal['mae', 'rmse'], fn=None, open_browser=True):
        """
        Need to measure error buckets of moneyness first
        USD error is vega * iv error
        """
        if self.evaluation is None:
            self.evaluate()

    def n_samples(self) -> int:
        return sum(map(len, (ci.price for ci in self.calibration_items.values())))

    @property
    def datapoints(self) -> List[DataPointSSVIOverTime]:
        # backwards comparible. Delete later
        if not isinstance(self.params, SSVISurfParams):
            self.params = SSVISurfParams({k: SSVITenorParams(*v) for k, v in self.params.items()})

        values = []
        calc_date = self.first_calibration_calc_date()
        ts_end = self.last_calibration_ts()

        for tenor_dt, sSVITenorParams in self.params.items():
            theta, rho, psi = sSVITenorParams
            tenor = get_tenor(tenor_dt, calc_date)
            values += [
                DataPointSSVIOverTime(self.id, self.underlying.symbol, tenor_dt, tenor, MetricSSVI.theta, theta, ts_end),
                DataPointSSVIOverTime(self.id, self.underlying.symbol, tenor_dt, tenor, MetricSSVI.rho, rho, ts_end),
                DataPointSSVIOverTime(self.id, self.underlying.symbol, tenor_dt, tenor, MetricSSVI.psi, psi, ts_end),
            ]
        return values

    def last_calibration_ts(self) -> datetime:
        # try:
        if not hasattr(self, '_last_calibration_ts') or not self._last_calibration_ts:
            np_ts = max([ci.ts[-1] for ci in self.calibration_items.values() if ci is not None and len(ci.ts) > 0])
            self._last_calibration_ts = pd.Timestamp(np_ts).to_pydatetime()
        return self._last_calibration_ts
        # except Exception as ex:
        #     warning(f'last_calibration_ts(): {ex}')
        #     return datetime(1970, 1, 1)

    def set_last_calibration_ts(self, ts: datetime):
        self._last_calibration_ts = ts
        return self

    def first_calibration_calc_date(self) -> date:
        obj = next(iter(self.calibration_items.values()), None)
        return obj.calculation_date if obj is not None else date.min

    @classmethod
    def is_cached(cls, equity: Equity, dates: List[date], resolution: Resolution|str = Resolution.second, seq_ret_threshold: float = 0.002, arb_free=False, seq_ret_threshold_surface=None) -> bool:
        root = Paths.path_analysis_frames
        payload = 'v_ivs', equity, dates, resolution, seq_ret_threshold, QuoteSide.mid, arb_free, seq_ret_threshold_surface
        fn = get_pkl_cache_key(*payload)
        return os.path.exists(os.path.join(root, fn))


def plot_ssvi_params_over_time(v_ivs: Iterable[IVSurface], fn=None, open_browser=True):
    """Plot how the surface params change over time"""
    if not v_ivs:
        return
    df = pd.DataFrame(chain(*(ivs.datapoints for ivs in v_ivs)))

    metrics = MetricSSVI.names()
    subs = [f'{metric}' for metric in metrics]

    for underlying, s_df in df.groupby('underlying'):
        from_ = s_df['ts_end'].min().date()
        to = s_df['ts_end'].max().date()
        fig = make_subplots(rows=len(subs) + 1, cols=2, subplot_titles=subs)
        for metric, sss_df in s_df.groupby('model_param'):
            ix_row = metrics.index(metric) + 1
            for tenor_dt, tenor_df in sss_df.groupby('tenor_dt'):
                fig.add_trace(go.Scatter(x=tenor_df['ts_end'], y=tenor_df['value'], mode='markers', marker=dict(size=4), name=f'{tenor_dt} {metric}'), row=ix_row,
                              col=1)

                metric_ma = tenor_df['value'].rolling(5).mean()
                # fig.add_trace(go.Scatter(x=tenor_df['ts_end'], y=metric_ma, mode='lines', name=f'{tenor_dt} {metric} MA', line=dict(color='red')), row=ix_row, col=1)

        fig.update_layout(title=f'{underlying} SSVI params (t)', autosize=True)
        show(fig, fn=fn or f'essvi_{underlying}_params_params_{from_}-{to}.html', open_browser=open_browser)

def plot_calibration_surface_debug(
        bids: np.ndarray,
        asks: np.ndarray,
        mny_fwd_ln: np.ndarray,
        t: np.ndarray,
        v_is_call: np.ndarray,
        tenor_offsets: list,
        tenors: list,
        calibration_params: list,
        title: str = "Calibration Debug",
        fn: str = "calibration_debug.html",
        open_browser: bool = True,
        plot_bid_ask_band: bool = False,
):
    """
    Plot bid/ask market quotes vs SSVI model IV per tenor slice.
    Designed to be called inside calibrate_surface() before/after the solver.

    Parameters
    ----------
    bids, asks       : flat filtered price arrays (already moneyness-filtered)
    mny_fwd_ln       : flat log-forward-moneyness array
    t                : flat tenor array (years)
    tenor_offsets    : list of (start, end) index pairs per tenor
    tenors           : sorted list of tenor_dt dates (same order as tenor_offsets)
    calibration_params : flat [theta, rho, psi, ...] per tenor
    """
    from options.types.option import get_price_cuda

    mid = 0.5 * (bids + asks)
    fig = make_subplots(
        rows=len(tenors), cols=1,
        subplot_titles=[str(td) for td in tenors],
        shared_xaxes=True,
        vertical_spacing=0.03,
    )

    for i, (tenor_dt, (start, end)) in enumerate(zip(tenors, tenor_offsets)):
        row = i + 1
        mny_sl = mny_fwd_ln[start:end]
        mid_sl = mid[start:end]
        bid_sl = bids[start:end]
        ask_sl = asks[start:end]
        t_sl = t[start:end]

        tenor_val = float(t_sl[0]) if len(t_sl) > 0 else 0.0
        theta = calibration_params[i * 3 + 0]
        rho = calibration_params[i * 3 + 1]
        psi = calibration_params[i * 3 + 2]

        # SSVI model IV on a smooth moneyness grid
        x_grid = np.linspace(mny_sl.min(), mny_sl.max(), 200)
        iv_model_grid = f_essvi_iv(x_grid, theta, rho, psi, tenor=tenor_val)

        # Back out IV from market mid prices for scatter
        from options.types.option import get_v_iv
        iv_mid = get_v_iv(
            p=mid_sl,
            s=np.full_like(mid_sl, mid_sl.mean()),  # spot approximation; replace if available
            k=np.exp(mny_sl) * mid_sl.mean(),  # approximate; replace with actual k slice if passed
            v_is_call=v_is_call,
            t=t_sl,
            dividends=[],
            calculation_date=tenor_dt,
        )

        sort_ix = np.argsort(mny_sl)
        x_sorted = mny_sl[sort_ix]

        # Bid/ask band
        if plot_bid_ask_band:
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_sorted, x_sorted[::-1]]),
                y=np.concatenate([ask_sl[sort_ix], bid_sl[sort_ix][::-1]]),
                fill='toself', fillcolor='rgba(100,150,255,0.15)',
                line=dict(color='rgba(0,0,0,0)'),
                name=f'{tenor_dt} bid/ask band', showlegend=(i == 0),
            ), row=row, col=1)

        # Market mid IV
        fig.add_trace(go.Scatter(
            x=x_sorted, y=iv_mid[sort_ix],
            mode='markers', marker=dict(size=3, color='blue'),
            name='market mid IV', showlegend=(i == 0),
        ), row=row, col=1)

        # SSVI model IV
        fig.add_trace(go.Scatter(
            x=x_grid, y=iv_model_grid,
            mode='lines', line=dict(color='red', width=1),
            name='SSVI model', showlegend=(i == 0),
        ), row=row, col=1)

    fig.update_layout(title=title, height=300 * len(tenors))
    show(fig, fn=fn, open_browser=open_browser)

if __name__ == '__main__':
    import pickle, os
    cache_key = 'v_ivs-c1d8c15a9e0045f8a9c2eeea2aa522b7.pkl'
    with open(os.path.join(r'D:\trade\Analytics\analysis_frames', cache_key), 'rb') as f:
        v_ivs = pickle.load(f)
    ivs = v_ivs[0]
    print('# Params')
    print(ivs.params)
    print('# Params end')
    ivs.plot_model_params(ivs.last_calibration_ts().date())
    ivs.plot_surface_vanilla(ivs.last_calibration_ts().date())
    # print(ivs.evaluate())
