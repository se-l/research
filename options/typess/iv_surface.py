import numpy as np
import pandas as pd
import multiprocessing

from datetime import date
from decimal import Decimal
from typing import Dict
from functools import partial

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, CloughTocher2DInterpolator
from scipy.interpolate.interpnd import NDInterpolatorBase

from shared.modules.logger import logger
from shared.utils.decorators import time_it
from shared.constants import EarningsPreSessionDates, DiscountRateMarket
from options.helper import year_quarter, interpolate_pt, get_tenor, get_dividend_yield
from options.typess.enums import Resolution, OptionRight
from options.typess.equity import Equity
from options.typess.option_frame import OptionFrame
from shared.plotting import show
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from plotly import graph_objects as go


class Grid:
    tenor_mny_fwd = 'tenor_mny_fwd'
    tenor_mny_fwd_ln = 'tenor_mny_fwd_ln'
    tenor_mny_fwd_ln_right = 'tenor_mny_fwd_ln_right'


class InterpolationMethod:
    linearND = 'linearND'


class IVSurface:
    """
    Enrich a frame with index ts, expiry, strike, right
    So plotting is easy
    """
    def __init__(self, underlying: Equity, df: pd.DataFrame, calc_date: date = None):
        self.underlying = underlying
        self.df = df
        self.calc_date = calc_date
        self.df_ah = pd.DataFrame()
        self.interpolators: Dict[(Grid, InterpolationMethod, OptionRight), NDInterpolatorBase] = {}

    def init_interpolator(self, grid: Grid = Grid.tenor_mny_fwd_ln, method: InterpolationMethod = InterpolationMethod.linearND, right: OptionRight = None, col='mid_iv'):
        """
        Initialize interpolator for IV surface.
        """
        if not self.calc_date:
            raise ValueError('IVSurface: calc_date not provided. Needed to caculate tenor.')

        if (grid, method) == ('tenor_mny_fwd_ln', 'linearND'):
            c_x, c_y = 'mny_fwd_ln', 'tenor'

        elif (grid, method) == ('tenor_mny_fwd', 'linearND'):
            c_x, c_y = 'mny_fwd', 'tenor'
        else:
            raise NotImplementedError(f'Grid {grid} not implemented.')

        df = self.df
        if 'ts' in df.index.names:
            raise ValueError('IVSurface: IV surface grid does not accept timestamps. Provide a snapshot. Index by expiry, strike, right.')

        if c_y == 'tenor' and c_y not in df.columns:
            df[c_y] = df.index.get_level_values('expiry').map(lambda dt: get_tenor(dt, self.calc_date))

        strike = df.index.get_level_values('strike').astype(float)
        if c_x not in df.columns and c_x in ('mny_fwd', 'mny_fwd_ln'):
            net_yield = DiscountRateMarket - get_dividend_yield(self.underlying.symbol)
            df[c_y] = df.index.get_level_values('expiry').map(lambda dt: get_tenor(dt, self.calc_date))
            fwd_s = df.apply(lambda ps: ps['spot'] * np.exp(net_yield * ps[c_y]), axis=1)
            df[c_x] = np.log(strike / fwd_s) if c_x == 'mny_fwd_ln' else strike / fwd_s

        df_right = df[df.index.get_level_values('right') == right] if right else df
        z = df_right[col].astype(float).values
        ix_an = ~np.isnan(z)
        z = z[ix_an]
        x = df_right[c_x].values[ix_an]
        y = df_right[c_y].values[ix_an]

        # i1 = CloughTocher2DInterpolator(list(zip(x, y)), z)
        # i2 = NearestNDInterpolator(list(zip(x, y)), z)
        # i3 = LinearNDInterpolator(list(zip(x, y)), z)
        self.interpolators[(grid, method, right, col)] = LinearNDInterpolator(list(zip(x, y)), z)

    def iv(self, tenor: float, moneyness_fwd_ln: float, right: OptionRight = None, grid: Grid = Grid.tenor_mny_fwd_ln, method: InterpolationMethod = InterpolationMethod.linearND, n_neighbors=4, col='mid_iv') -> float:
        """
        Interpolating IV from the frame.
        """
        if (grid, method, right, col) not in self.interpolators:
            self.init_interpolator(grid, method, right, col=col)

        return interpolate_pt(self.interpolators[(grid, method, right, col)], moneyness_fwd_ln, tenor)
        # return interpolate_pt(self.interpolators[(grid, method, right)], tenor, moneyness_fwd_ln)


    @time_it
    def enrich_iv_curvature(self, iv_col='mid_iv'):
        iv_col_curvature = iv_col + '_curvature'
        self.df[iv_col_curvature] = None
        for right, s_df in self.df.groupby('right'):
            for ts, ss_df in s_df.groupby('ts'):
                for expiry, sss_df in ss_df.groupby('expiry'):
                    # sorted_df = sss_df.sort_index(level='strike')
                    if len(sss_df) < 2:
                        continue
                    sorted_df = sss_df.sort_values('moneyness')
                    # sorted_df = sss_df.sort_values('moneyness_fwd_ln')
                    strikes = sorted_df.index.get_level_values('strike')
                    x_val = strikes.astype(float).values
                    # x_val = sorted_df['moneyness'].values
                    v_iv = sorted_df[iv_col].values

                    # interpolate nans and zeros
                    x = x_val[np.isnan(v_iv) | (v_iv == 0)].astype(float)
                    xp = x_val[~np.isnan(v_iv) & (v_iv != 0)].astype(float)
                    if len(x) > 0 and len(xp) > 0:
                        fp = v_iv[~np.isnan(v_iv) & (v_iv != 0)]
                        y = np.interp(x, xp, fp, left=None, right=None, period=None)
                        v_iv[np.isnan(v_iv) | (v_iv == 0)] = y

                    if len(v_iv) < 2:
                        continue
                    iv_curvature = np.gradient(v_iv, x_val)

                    # Insert extrapolating here for 0, np.nan values

                    ix_x = np.argwhere(np.isnan(iv_curvature) | (iv_curvature == 0)).flatten()
                    ix_p = np.argwhere((~np.isnan(iv_curvature)) & (iv_curvature != 0)).flatten()
                    if len(ix_x) > 0 and len(ix_p) > 0:
                        # extrapolate, interpolate
                        x = x_val[ix_x]
                        xp = x_val[ix_p]
                        fp = iv_curvature[ix_p]
                        y = np.interp(x, xp, fp, left=None, right=None, period=None)
                        iv_curvature[ix_x] = y

                    self.df.loc[(ts, expiry, strikes, right), iv_col_curvature] = iv_curvature

        print(f'enrich_iv_curvature: iv_col={iv_col_curvature}, Total Values: {self.df[iv_col_curvature].count()}. # NA: {self.df[iv_col_curvature].isna().sum()}, # Zero: {(self.df[iv_col_curvature] == 0).sum()}')

    def mp_enrich_mean_regressed_skew_ds(self, v_ds_ret, n_processes=12):
        with multiprocessing.Pool(n_processes) as p:
            frames = p.map(partial(enrich_mean_regressed_skew_for_ds, df=self.df), v_ds_ret)
            if frames:
                concat = pd.concat(frames, axis=1)
                for c in concat.columns:
                    self.df[c] = concat[c]

    def mp_enrich_div_skew_of_ds(self, v_ds_pct, iv_col='mid_iv', n_processes=8):
        with multiprocessing.Pool(n_processes) as p:
            frames = p.map(partial(self.enrich_div_skew_of_ds, df=self.df, iv_col=iv_col), v_ds_pct)
        if frames:
            concat = pd.concat(frames, axis=1)
            for c in concat.columns:
                self.df[c] = concat[c]

    # @time_it
    @staticmethod
    def enrich_div_skew_of_ds(ds_pct, df: pd.DataFrame, iv_col='mid_iv'):
        """
        Improvement: edges end up having zero skew. extrapolate them.

        Skew in between 2 points.
        Can be spot prices, strikes, deltas, moneyness, etc. let's pick moneyness for now as that goes onto plots
        For large dS, it spans multiple strikes. Should intermediate strikes be considered? No, let's only consider the strikes
        closest to target s0, s1.
        """
        raise NotImplementedError('This method was implemented for a previous use case. might get refactored to roll smooth iv 3rd moment and gradient.')
        col_nm = f'div_skew_ds_{ds_pct:.2f}'
        df[col_nm] = None
        s1 = df['spot'] * (1 + ds_pct)
        df['dS_tmp'] = s1 - df['spot']

        strike0_fl = df.index.get_level_values('strike').astype(float)
        df['strike0'] = df.index.get_level_values('strike')
        df['strike1'] = strike0_fl - df['dS_tmp']
        df['strike1_nearest'] = None

        for right, s_df in df.groupby('right'):
            for expiry, ss_df in s_df.groupby(level='expiry'):
                for ts, sss_df in ss_df.groupby('ts'):
                    strike0 = sss_df.index.get_level_values('strike')
                    set_strike0 = np.array(list(set(strike0)))
                    set_strike0_fl = set_strike0.astype(float)
                    strike1 = sss_df['strike1'].apply(lambda s: set_strike0[np.argmin(np.abs(set_strike0_fl - s))]).values
                    df.loc[sss_df.index, 'strike1_nearest'] = strike1
        df['dS_nearest'] = strike0_fl - df['strike1_nearest'].astype(float)
        # pd.derivatives.display.max_columns = 100
        # df[['strike1', 'strike1_nearest', 'dS_tmp', 'dS_nearest']]

        iv0 = df[iv_col]
        df_ix_iv1 = df.reset_index().set_index(['ts', 'expiry', 'strike1_nearest', 'right'])
        iv1 = df.loc[df_ix_iv1.index, iv_col]
        df_ix_iv1['iv1'] = iv1
        df_ix_iv1 = df_ix_iv1.reset_index().set_index(['ts', 'expiry', 'strike', 'right'])
        df.loc[df_ix_iv1.index, 'iv1'] = df_ix_iv1['iv1']

        dIV = df['iv1'] - iv0
        df[col_nm] = dIV / df['dS_nearest']

        # ix_ignore = df.index[df['strike1_nearest'] == df['strike0']]

        # Insert extrapolating here for 0, np.nan values
        for expiry, s_df in df.groupby(level='expiry'):
            for ts, ss_df in s_df.groupby('ts'):
                for right, sss_df in ss_df.groupby('right'):
                    ix_x = sss_df[col_nm][(sss_df[col_nm].isna()) | (sss_df[col_nm] == 0)].index
                    ix_p = sss_df[(~sss_df[col_nm].isna()) & (sss_df[col_nm] != 0)].index
                    if len(ix_x) > 0 and len(ix_p) > 0:
                        # extrapolate, interpolate
                        x = ix_x.get_level_values('strike').astype(float).values
                        xp = ix_p.get_level_values('strike').astype(float).values
                        fp = sss_df.loc[ix_p, col_nm].astype(float).values
                        y = np.interp(x, xp, fp, left=None, right=None, period=None)
                        df.loc[ix_x, col_nm] = y

        # Rolling average, reset at SOD.
        col_nm_rolling = f'{col_nm}_rolling'
        df[col_nm_rolling] = None
        for expiry, s_df in df.groupby(level='expiry'):
            for dt, ss_df in s_df.groupby('date'):
                for right, sss_df in ss_df.groupby('right'):
                    mean_skew = sss_df[col_nm].groupby(level='ts').mean()
                    mean_skew_rolling = mean_skew.rolling(window=len(ss_df), min_periods=0).mean().bfill()
                    # likely bad performance here...
                    for ts, val in mean_skew_rolling.items():
                        # print(ts, val)
                        df.loc[(ts, expiry, slice(None), right), col_nm_rolling] = val

        # Drop temp columns
        df.drop(columns=['strike1_nearest', 'strike1', 'dS_tmp'], inplace=True)

        logger.info(f'enrich_div_skew_of_ds: ds%={ds_pct:.2f} , Total Values: {df[col_nm].count()}. # NA: {df[col_nm].isna().sum()}, # Zero: {(df[col_nm] == 0).sum()}')

        return df[[col_nm, col_nm_rolling]]


def a_bx_cx2(x, a, b, c):
    return a + b * x + c * x ** 2


def enrich_regressed_skew(df: pd.DataFrame, x='strike', y='mid_iv_curvature', moneyness_col_nm='moneyness_fwd_ln', plot=False, ts_plot=None):
    """
    Regressing the central moment of the IV surface. This is the skew of the IV surface.
    Applying more weight to ATM IV, IV more uncertain the further OTM/ITM it is. That'll be a sigma array, where sigma~abs(1-moneyness) or sigma~abs(ln(moneyness))
    Enforcing a smirk/smile like shape by setting bounds on the regression coefficients.
    Saving each regression coefficient to the frame.
    """
    bounds = ((-np.inf, 0, -np.inf), (0, np.inf, 0))

    df[f'{y}_a'] = None
    df[f'{y}_b'] = None
    df[f'{y}_c'] = None
    df['strike'] = df.index.get_level_values('strike').astype(float)

    df_scoped = df[(df['moneyness'] > 0.6) & (df['moneyness'] < 1.4)]

    expiries = df_scoped.index.get_level_values('expiry').unique()
    if plot:
        fig = make_subplots(rows=len(expiries)*2+2, cols=2)
        r2s = []

    for k, (expiry, s_df) in enumerate(df_scoped.groupby(level='expiry')):
        for l, (right, ss_df) in enumerate(s_df.groupby(level='right')):
            # print(f'{expiry} {right}')
            if (~ss_df[y].isna()).sum() == 0:
                continue

            ix_compatible = ss_df[y].reset_index().index[ss_df[y].notna()]
            # popt, pcov = curve_fit(a_bx_cx2, ss_df[x].values[ix_compatible], ss_df[y].values[ix_compatible], bounds=bounds)
            # popt, pcov = curve_fit(a_bx_cx2, ss_df[x].values[ix_compatible], ss_df[y].values[ix_compatible])

            # More weight to ATM IV, IV more uncertain the further OTM/ITM it is. That'll be a sigma array, where sigma~abs(1-moneyness) or sigma~abs(ln(moneyness))
            sigma = np.exp(((np.abs(ss_df['moneyness'] - 1) + 1) ** 2).astype(float).values[ix_compatible])
            popt, pcov = curve_fit(a_bx_cx2, ss_df[x].values[ix_compatible], ss_df[y].values[ix_compatible], bounds=bounds, sigma=sigma)

            try:
                df.loc[(expiry, slice(None), right), f'{y}_a'] = popt[0]
                df.loc[(expiry, slice(None), right), f'{y}_b'] = popt[1]
                df.loc[(expiry, slice(None), right), f'{y}_c'] = popt[2]
            except Exception as e:
                print(e)
                continue

            if plot:
                ix_scope = ss_df[(ss_df['moneyness'] > 0.8) & (ss_df['moneyness'] < 1.2)].index
                y_pred = a_bx_cx2(ss_df.loc[ix_scope, x], *popt)
                r2 = r2_score(ss_df.loc[ix_scope, y], y_pred)
                r2s.append(r2)
                # if r2 < 0.1:
                #     pass
                print(f'R2 Quadratic Regr. {right} {expiry}: {r2}')
                fig.add_trace(go.Scatter(x=ss_df[x].values, y=ss_df['mid_iv'].values, mode='markers', name=f'IV {y} {expiry} {right}', marker=dict(size=2)), row=k*2 + 1, col=l + 1)
                fig.add_trace(go.Scatter(x=ss_df[x].values, y=ss_df[y].values, mode='markers', name=f'Skew {y} {expiry} {right}', marker=dict(size=2)), row=k*2+2, col=l + 1)
                ix_plottable = y_pred.reset_index(drop=True).index[y_pred.abs() > 1/10**10]
                yv = y_pred.iloc[ix_plottable].astype(float).values
                fig.add_trace(go.Scatter(x=ss_df[x].values[ix_plottable], y=yv, mode='markers', name=f'Regressed {y} {expiry} {right}', marker=dict(size=3)), row=k*2+2, col=l+1)

    if plot:
        print(f'Mean R2: {np.mean(r2s)}')
        fig.update_layout(title_text=f'Regressed {y} {ts_plot}')
        show(fig, 'skew_regression.html')

    df['mid_iv_curvature_regressed'] = df[f'{y}_a'] + df[f'{y}_b'] * df['strike'] + df[f'{y}_c'] * df['strike'] ** 2
    df.drop(columns=['strike'], inplace=True)
    return df


def enrich_regressed_skew_rolling(df, window: pd.Timedelta | int = pd.Timedelta(hours=24), x='strike', y='mid_iv_curvature', plot=False, ts_plot=None):
    """
    Regressing the central moment of the IV surface. This is the skew of the IV surface.
    Applying more weight to ATM IV, IV more uncertain the further OTM/ITM it is. That'll be a sigma array, where sigma~abs(1-moneyness) or sigma~abs(ln(moneyness))
    Enforcing a smirk/smile like shape by setting bounds on the regression coefficients.
    Saving each regression coefficient to the frame.
    """
    bounds = ((-np.inf, 0, -np.inf), (0, np.inf, 0))

    df['mid_iv_curvature_a'] = None
    df['mid_iv_curvature_b'] = None
    df['mid_iv_curvature_c'] = None
    df['strike'] = df.index.get_level_values('strike').astype(float)

    v_ts = sorted(df.index.get_level_values('ts').unique())
    window_int = len([i for i in v_ts if i > v_ts[-1] - window])

    # roll over a ts window container window ts items
    for ps_ts_window in pd.Series(v_ts).rolling(window=window_int):
        v_ts_window = ps_ts_window.dropna().values
        # more sample use all data points after 10am.
        # ts_scoped = list(sorted([ts for ts in df.index.get_level_values('ts').unique() if ts.hour >= 10]))[-window:]
        df_scoped = df.loc[v_ts_window]
        # df_scoped = df_scoped[df_scoped['tenor'] > 0.5]
        df_scoped = df_scoped[(df_scoped['moneyness'] > 0.6) & (df_scoped['moneyness'] < 1.4)]

        expiries = df_scoped.index.get_level_values('expiry').unique()
        if plot:
            fig = make_subplots(rows=len(expiries)*2+2, cols=2)
            r2s = []

        for k, (expiry, s_df) in enumerate(df_scoped.groupby(level='expiry')):
            for l, (right, ss_df) in enumerate(s_df.groupby(level='right')):
                # print(f'{expiry} {right}')
                if (~ss_df[y].isna()).sum() == 0:
                    continue

                ix_compatible = ss_df[y].reset_index().index[ss_df[y].notna()]
                # popt, pcov = curve_fit(a_bx_cx2, ss_df[x].values[ix_compatible], ss_df[y].values[ix_compatible], bounds=bounds)
                # popt, pcov = curve_fit(a_bx_cx2, ss_df[x].values[ix_compatible], ss_df[y].values[ix_compatible])

                # More weight to ATM IV, IV more uncertain the further OTM/ITM it is. That'll be a sigma array, where sigma~abs(1-moneyness) or sigma~abs(ln(moneyness))
                sigma = np.exp(((np.abs(ss_df['moneyness'] - 1) + 1) ** 2).astype(float).values[ix_compatible])
                popt, pcov = curve_fit(a_bx_cx2, ss_df[x].values[ix_compatible], ss_df[y].values[ix_compatible], bounds=bounds, sigma=sigma)

                try:
                    df.loc[(v_ts_window[-1], expiry, slice(None), right), 'mid_iv_curvature_a'] = popt[0]
                    df.loc[(v_ts_window[-1], expiry, slice(None), right), 'mid_iv_curvature_b'] = popt[1]
                    df.loc[(v_ts_window[-1], expiry, slice(None), right), 'mid_iv_curvature_c'] = popt[2]
                except Exception as e:
                    continue

                if plot and ts_plot == v_ts_window[-1]:
                    ix_scope = ss_df[(ss_df['moneyness'] > 0.8) & (ss_df['moneyness'] < 1.2)].index
                    y_pred = a_bx_cx2(ss_df.loc[ix_scope, x], *popt)
                    r2 = r2_score(ss_df.loc[ix_scope, y], y_pred)
                    r2s.append(r2)
                    # if r2 < 0.1:
                    #     pass
                    print(f'R2 Quadratic Regr. {right} {expiry}: {r2}')
                    fig.add_trace(go.Scatter(x=ss_df[x].values, y=ss_df['mid_iv'].values, mode='markers', name=f'IV {y} {expiry} {right}', marker=dict(size=2)), row=k*2 + 1, col=l + 1)
                    fig.add_trace(go.Scatter(x=ss_df[x].values, y=ss_df[y].values, mode='markers', name=f'Skew {y} {expiry} {right}', marker=dict(size=2)), row=k*2+2, col=l + 1)
                    ix_plottable = y_pred.reset_index(drop=True).index[y_pred.abs() > 1/10**10]
                    yv = y_pred.iloc[ix_plottable].astype(float).values
                    fig.add_trace(go.Scatter(x=ss_df[x].values[ix_plottable], y=yv, mode='markers', name=f'Regressed {y} {expiry} {right}', marker=dict(size=3)), row=k*2+2, col=l+1)

        if plot and ts_plot == v_ts_window[-1]:
            print(f'Mean R2: {np.mean(r2s)}')
            fig.update_layout(title_text=f'Regressed {y} {ts_plot}')
            show(fig, 'skew_regression.html')

    df['mid_iv_curvature_regressed'] = df['mid_iv_curvature_a'] + df['mid_iv_curvature_b'] * df['strike'] + df['mid_iv_curvature_c'] * df['strike'] ** 2
    df.drop(columns=['strike'], inplace=True)
    return df


def col_nm_mean_regressed_skew_ds(ds_ret):
    return f'mean_regressed_skew_for_ds_{ds_ret:.2f}'


def enrich_mean_regressed_skew_for_ds(ds_ret: float, df: pd.DataFrame):
    # Average all skews up to a K close to -ds
    col_nm = col_nm_mean_regressed_skew_ds(ds_ret)
    if col_nm not in df.columns:
        df[col_nm] = None

    df_strikes = df.index.get_level_values('strike').unique()

    # Refactor this: bad code duplication

    if 'ts' in df.index.names:
        for ts, ts_df in df.groupby('ts'):

            s0 = ts_df['spot'].iloc[0]
            ds = s0 * ds_ret - s0
            strike2strike2average = {}
            for strike in df_strikes:
                final_strike = float(strike) + ds
                if final_strike < float(strike):
                    strike2strike2average[strike] = df_strikes[(final_strike <= df_strikes.astype(float)) & (df_strikes.astype(float) <= float(strike))]
                else:
                    strike2strike2average[strike] = df_strikes[(float(strike) <= df_strikes.astype(float)) & (df_strikes.astype(float) <= final_strike)]

            for expiry, s_df in ts_df.groupby(level='expiry'):
                for right, ss_df in s_df.groupby(level='right'):
                    strikes = ss_df.index.get_level_values('strike')
                    expected_d_iv = []
                    for strike in strikes:
                        strikes_to_avg = strike2strike2average[strike].intersection(strikes)
                        expected_d_iv.append(ss_df.loc[(ts, expiry, strikes_to_avg, right), 'mid_iv_curvature_regressed'].mean())

                    df.loc[(ts, expiry, strikes, right), col_nm] = expected_d_iv
    else:
        s0 = df['spot'].iloc[0]
        ds = s0 * ds_ret - s0
        strike2strike2average = {}
        for strike in df_strikes:
            final_strike = float(strike) + ds
            if final_strike < float(strike):
                strike2strike2average[strike] = df_strikes[(final_strike <= df_strikes.astype(float)) & (df_strikes.astype(float) <= float(strike))]
            else:
                strike2strike2average[strike] = df_strikes[(float(strike) <= df_strikes.astype(float)) & (df_strikes.astype(float) <= final_strike)]

        for expiry, s_df in df.groupby(level='expiry'):
            for right, ss_df in s_df.groupby(level='right'):
                strikes = ss_df.index.get_level_values('strike')
                expected_d_iv = []
                for strike in strikes:
                    strikes_to_avg = strike2strike2average[strike].intersection(strikes)
                    expected_d_iv.append(ss_df.loc[(expiry, strikes_to_avg, right), 'mid_iv_curvature_regressed'].mean())

                df.loc[(expiry, strikes, right), col_nm] = expected_d_iv

    logger.info(f'enrich_mean_regressed_skew_for_ds: ds_ret={ds_ret:.2f} , Total Values: {df[col_nm].count()}. # NA: {df[col_nm].isna().sum()}, # Zero: {(df[col_nm] == 0).sum()}')
    return df[col_nm]


def test_skew_calc(option_frame: OptionFrame):
    # option_frame = OptionFrame.load_frame(equity, Resolution.second, seq_ret_threshold, year_quarter(release_date) + '_kept_sec')
    df = option_frame.df_options.sort_index()
    # option_frame.df_options
    if False:
        iv_surface = IVSurface(equity, df)
        iv_surface.enrich_iv_curvature(iv_col='mid_iv')
        iv_surface.df = enrich_regressed_skew_rolling(iv_surface.df)
        enrich_regressed_skew_rolling(iv_surface.df, plot=True, ts_plot=release_date)
        # enrich_mean_regressed_skew_for_ds(0.9, iv_surface.df)

    # Further, would want to experiment with multiple smoothing techniques for the skew measures.
    # Rolling EWMA, outlier removal

    #  plot IV, curvature + regressed skew of every expiry, error plot. That plot better be static.

    sample_df = df  # df[(df['moneyness'] > 0.7) & (df['moneyness'] < 1.3)]
    plot_dt = release_date
    sample_df = sample_df.loc[np.array([ts.date() for ts in sample_df.index.get_level_values('ts')]) == plot_dt]
    expiries = sample_df.index.get_level_values('expiry').unique()

    fig = make_subplots(rows=2, cols=1, subplot_titles=[''])
    for k, (right, ss_df) in enumerate(sample_df.groupby('right')):
        # if ts.date() > date(2023, 11, 14):
        #     continue
        s_df = sample_df.loc[(slice(None), date(2024, 3, 15), Decimal('135'), right)]
        fig.add_trace(go.Scatter(x=s_df.index.get_level_values('ts'), y=s_df['mid_iv_curvature_regressed'], mode='markers', name=f'{right} IV', marker=dict(size=marker_size)), row=k+1, col=1)
    show(fig, fn='iv_curvature_regressed.html')

    nm = col_nm_mean_regressed_skew_ds(1)
    figSkew = make_subplots(rows=len(expiries), cols=4, subplot_titles=['IV', 'IV Curvature Regr.', nm, 'RMSE IV Curvature regr.'])
    for i, expiry in enumerate(expiries):
        r = i+1
        s_df = sample_df.loc[(slice(None), expiry, slice(None))]
        for k, (right, ss_df) in enumerate(s_df.groupby('right')):
            figSkew.add_trace(go.Scatter(x=ss_df['moneyness'], y=ss_df['mid_iv'], mode='markers', name=f'{right} IV {expiry}', marker=dict(size=marker_size)), row=r, col=1)
            # figSkew.add_trace(go.Scatter(x=ss_df['moneyness'], y=ss_df['mid_iv_curvature'], mode='markers', name=f'IV Curvature {right} {expiry}', marker=dict(size=marker_size)), row=r, col=2)
            figSkew.add_trace(go.Scatter(x=ss_df['moneyness'], y=ss_df['mid_iv_curvature_regressed'], mode='markers', name=f'{right} IV Curvature regr. {expiry}', marker=dict(size=marker_size)), row=r, col=2)

            figSkew.add_trace(go.Scatter(x=ss_df['moneyness'], y=ss_df[nm], mode='markers', name=f'{right} {nm} {expiry}', marker=dict(size=marker_size)), row=r, col=3)

            # plot mae RMSE by moneyness
            # figSkew.add_trace(go.Scatter(x=ss_df['moneyness'], y=ss_df['mid_iv_curvature_regressed'] - ss_df['mid_iv_curvature'], mode='markers', name=f'MAE IV Curvature regr. {right} {expiry}', marker=dict(size=marker_size)), row=r, col=3)
            figSkew.add_trace(go.Scatter(x=ss_df['moneyness'], y=(ss_df['mid_iv_curvature_regressed'] - ss_df['mid_iv_curvature'])**2, mode='markers', name=f' {right} RMSE IV Curvature regr.{expiry}', marker=dict(size=marker_size)), row=r, col=4)
            # figSkew.add_vline(x=1, line_width=1, line_dash="dash", line_color="black", row=3, col=1)
    # figSkew.add_vline(x=1, line_width=1, line_dash="dash", line_color="black", row=1, col=1)
    figSkew.update_layout(title_text=f'plot_dt={plot_dt}', xaxis_title='Moneyness')
    show(figSkew)

    # calculate error in estimation of dSdIV combining a dS shift * skew adjusted for a measured shift in ATM(exp). So essentially removing the effect of term structure / horizontal
    # only leaving vertical!
    # benchmark - zero skew. does it improve IV estimate for which expiries and dS?

    # Conversely, adjusted for skew, does the power law apply. Do term structures equilibrate


def test_interpolator_pt(equity, df_option, calc_date):
    iv_surface = IVSurface(equity, df_option, calc_date)
    print(f'IV: {iv_surface.iv(0.1, 0.1)}')
    print(f'IV: {iv_surface.iv(0.1, 0.1, OptionRight.call)}')
    print(f'IV: {iv_surface.iv(0.1, 0.1, OptionRight.put)}')


def median(ps: pd.Series, skipna=True):
    _ps = ps[~ps.isna()] if skipna else ps
    if len(_ps) == 0:
        return np.nan
    return sorted(_ps)[len(_ps) // 2]


def get_df_median(df0):
    """ps0 where mid_iv == median"""
    ps_lst = []
    for expiry, s_df in df0.groupby('expiry'):
        for strike, s_df2 in s_df.groupby('strike'):
            for right, s_df3 in s_df2.groupby('right'):
                mid_iv = s_df3['mid_iv']
                mid_iv_median = median(mid_iv)
                df_median = s_df3.loc[mid_iv == mid_iv_median]
                if len(df_median) > 0:
                    ps_lst.append(df_median.iloc[0])
                else:
                    print('None eel')
    df = pd.concat(ps_lst, axis=1).transpose()
    df.index = df.index.set_names(['ts', 'expiry', 'strike', 'right'])
    return df


# if __name__ == '__main__':
#     sym = 'PANW'
#     equity = Equity(sym)
#     resolution = Resolution.minute
#     seq_ret_threshold = 0.005
#     release_date = EarningsPreSessionDates(sym)[-2]
#     rate = DiscountRateMarket
#     dividend_yield = DividendYield[sym.upper()]
#     net_yield = rate - dividend_yield
#
#     marker_size = 2
#     option_frame = OptionFrame.load_frame(equity, Resolution.minute, seq_ret_threshold, year_quarter(release_date))
#     df_quote = option_frame.df_options
#
#     df = df_quote[df_quote.index.get_level_values('ts').date == release_date]
#     df = get_df_median(df).droplevel('ts')
#     test_interpolator_pt(equity, df, release_date)
#
#     # # T-1 fit initial state
#     # df_trades = option_frame.df_option_trades.sort_index()
#
#     print(f'Done {sym}')
