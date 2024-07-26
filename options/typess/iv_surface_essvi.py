from itertools import chain

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from datetime import date
from typing import Dict, List, Tuple
from plotly.subplots import make_subplots
from options.helper import get_tenor
from options.typess.calibration_item import CalibrationItem
from shared.constants import EarningsPreSessionDates
from options.typess.enums import Resolution, OptionRight
from options.typess.equity import Equity
from scipy.optimize import least_squares

from shared.modules.logger import warning, info
from shared.plotting import show, plot_scatter_3d


class Params:
    tenor_dt: date
    right: str | OptionRight
    theta: float
    rho: float
    psi: float


def f_essvi_total_variance(k, theta, rho, psi):
    """
    Implied total variance surface
    eSSVI(K, T) = 1/2 ( θ(T) + ρ(T)ψ(T)k + sqrt((ψ(T)k + θ(T)ρ(T))**2 + θ(T)**2 * (1 − ρ(T)**2))
    :return:
    """
    return 0.5 * (theta + rho * psi * k + np.sqrt((psi * k + theta * rho) ** 2 + theta ** 2 * (1 - rho ** 2)))


def f_essvi_iv(k, theta, rho, psi, tenor=None):
    """
    Implied total variance surface
    eSSVI(K, T) = 1/2 ( θ(T) + ρ(T)ψ(T)k + sqrt((ψ(T)k + θ(T)ρ(T))**2 + θ(T)**2 * (1 − ρ(T)**2))
    """
    # return 0.5 * (theta + rho * psi * k + np.sqrt((psi * k + theta * rho) ** 2 + theta ** 2 * (1 - rho ** 2)))
    implied_total_variance = 0.5 * (theta + rho * psi * k + np.sqrt((psi * k + theta * rho) ** 2 + theta ** 2 * (1 - rho ** 2)))
    return np.sqrt(implied_total_variance / tenor)


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
            # ixiv0 = item.iv != 0
            total_variance = item.iv ** 2 * item.tenor
            fit_res = least_squares(f_min, initial_params, args=(item.mny_fwd_ln, total_variance, f_essvi_total_variance, item.weights), verbose=verbose, max_nfev=1000)
            # fit_res = least_squares(f_min, initial_params, args=(item.mny_fwd_ln, item.iv, f_essvi_iv, item.weights), verbose=verbose)

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

    def skew(self, moneyness_fwd_ln: float, tenor: float, right: OptionRight = None) -> float: pass


# if __name__ == '__main__':
#     # from connector.api_minlp.common import repair_quotes_keeping_holding_quotes
#     # from derivatives.frame_builder import get_option_frame
#     tickers = 'UNH'
#     equity = Equity('UNH')
#     resolution = Resolution.second
#     seq_ret_threshold = 0.002
#     release_date = EarningsPreSessionDates(equity.symbol)[-3]
#     dates = [release_date]
#
#     option_frame = get_option_frame(equity, dates, resolution=resolution, seq_ret_threshold=seq_ret_threshold)
#     df = option_frame.df_options
#
#     # Clean frame before calibration
#     y_col_nm = 'mid_price_iv'
#     df = df[(df[y_col_nm] > 0) & (df[y_col_nm] < 4)]
#     if 'bid_iv' in df.columns:
#         df = df[(df[y_col_nm] >= df['bid_iv']) & (df[y_col_nm] <= df['ask_iv'])]
#     df = repair_quotes_keeping_holding_quotes(df, [], equity)
#     df = df[df['vega_mid_iv'].notna() & df[y_col_nm].notna() & df['moneyness_fwd_ln'].notna()]
# ##############################################################
#     ivs = IVSurface(equity)
#     # Calibration
#     y_col_nm = 'mid_price_iv'
#     calibration_items = []
#     for expiry, expiry_df in df.groupby('expiry'):
#         for right, right_df in expiry_df.groupby('right'):
#             calibration_items.append(CalibrationItem(right_df['moneyness_fwd_ln'].values.astype(float), release_date, expiry, right, right_df[y_col_nm].values.astype(float), right_df['vega_mid_iv'].values.astype(float)))
#     ivs.calibrate(calibration_items)
#     print(ivs.iv(0.1, expiry, OptionRight.put, release_date))
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
