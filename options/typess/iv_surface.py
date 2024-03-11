import multiprocessing
from functools import partial

# import QuantLib as ql
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from plotly import graph_objects as go

from shared.utils.decorators import time_it
from shared.constants import EarningsPreSessionDates
from options.helper import year_quarter, skew_measure2target_metric
from options.typess.enums import OptionRight, Resolution, SkewMeasure
from options.typess.equity import Equity
from options.typess.option_frame import OptionFrame
from shared.plotting import show


class IVSurface:
    """
    Enrich a frame with index ts, expiry, strike, right
    So plotting is easy
    """
    def __init__(self, underlying: Equity, df: pd.DataFrame):
        self.underlying = underlying
        self.df = df
        self.v_ah_srf = {OptionRight.call: {}, OptionRight.put: {}}
        # self.day_count = ql.Actual365Fixed()
        self.df_ah = pd.DataFrame()

    # @time_it
    # def interpolate_ah(self):
    #     for right, s_df in self.df.groupby('right'):
    #         for ts, ss_df in s_df.groupby('ts'):
    #             calculation_date_ql = to_ql_dt(ts)
    #             spot_quote_handle = ql.QuoteHandle(ql.SimpleQuote(ss_df['spot'].iloc[0]))
    #             calibration_set = get_calibration_set(ss_df, right)
    #             ah_vol_interpolation = ql.AndreasenHugeVolatilityInterpl(calibration_set, spot_quote_handle, self.yield_ts(calculation_date_ql), self.dividend_ts(calculation_date_ql))
    #             ah_vol_surface = ql.AndreasenHugeVolatilityAdapter(ah_vol_interpolation)
    #             self.v_ah_srf[right][ts] = ah_vol_surface

    def other_interpol(self):
        # ql.BicubicSpline(s_df['expiry'], s_df['strike'], s_df['mid_iv'])
        pass

    # def enrich_iv_ah(self, iv_col='mid_iv'):
    #     iv_col_ah = iv_col + '_ah'
    #     if not self.v_ah_srf[OptionRight.call]:
    #         self.interpolate_ah()
    #     self.df[iv_col_ah] = None
    #     for right, s_df in self.df.groupby('right'):
    #         for ts, ss_df in s_df.groupby('ts'):
    #             for expiry, sss_df in ss_df.groupby('expiry'):
    #                 tenor = (datetime.combine(expiry, datetime.min.time()) - ts).days / 365
    #                 v_iv = []
    #                 for strike in sss_df.index.get_level_values('strike'):
    #                     try:
    #                         v_iv.append(self.v_ah_srf[right][ts].blackVol(tenor, float(strike)))
    #                     except RuntimeError as e:
    #                         v_iv.append(np.nan)
    #
    #                 self.df.loc[(ts, expiry, slice(None), right), iv_col_ah] = v_iv
    #
    #     print(f'enrich_iv_ah: {iv_col_ah}. Total Values: {self.df[iv_col_ah].count()}. # NA: {self.df[iv_col_ah].isna().sum()}, # Zero: {(self.df[iv_col_ah] == 0).sum()}')

    @time_it
    def enrich_iv_curvature(self, iv_col='mid_iv'):
        iv_col_curvature = iv_col + '_curvature'
        self.df[iv_col_curvature] = None
        for right, s_df in self.df.groupby('right'):
            for ts, ss_df in s_df.groupby('ts'):
                for expiry, sss_df in ss_df.groupby('expiry'):
                    sorted_df = sss_df.sort_index(level='strike')
                    strikes = sorted_df.index.get_level_values('strike')
                    v_iv = sorted_df[iv_col].values

                    # interpolate nans and zeros
                    x = strikes[np.isnan(v_iv) | (v_iv == 0)]
                    xp = strikes[~np.isnan(v_iv) & (v_iv != 0)]
                    if len(xp) > 0:
                        fp = v_iv[~np.isnan(v_iv) & (v_iv != 0)]
                        y = np.interp(x, xp, fp, left=None, right=None, period=None)
                        v_iv[np.isnan(v_iv) | (v_iv == 0)] = y

                    iv_curvature = np.gradient(np.gradient(v_iv))
                    self.df.loc[(ts, expiry, strikes, right), iv_col_curvature] = iv_curvature
        print(f'enrich_iv_curvature: iv_col={iv_col_curvature}, Total Values: {self.df[iv_col_curvature].count()}. # NA: {self.df[iv_col_curvature].isna().sum()}, # Zero: {(self.df[iv_col_curvature] == 0).sum()}')

    @time_it
    def enrich_skew_measures(self, skew_measure: SkewMeasure, ref_measure_col='ask_delta', ref_iv_col='mid_iv'):
        skew_col_nm = f'skew_{skew_measure}'
        ix_strike = 2
        for expiry, s_df in self.df.groupby(level='expiry'):
            for right, ss_df in s_df.groupby(level='right'):

                for ts, sss_df in ss_df.groupby(level='ts'):
                    if (sss_df[ref_measure_col] != 0).sum() < 2:
                        continue
                    measure0, measure1 = skew_measure2target_metric(skew_measure=skew_measure, right=right)

                    strike0 = (sss_df[ref_measure_col] - measure0).abs().idxmin()[ix_strike]
                    strike1 = (sss_df[ref_measure_col] - measure1).abs().idxmin()[ix_strike]

                    delta0 = sss_df.loc[(ts, expiry, strike0, right), ref_measure_col]
                    delta1 = sss_df.loc[(ts, expiry, strike1, right), ref_measure_col]

                    if delta0 == 0 or delta1 == 0 or np.isnan(delta0) or np.isnan(delta1):
                        skew = np.nan
                    else:
                        iv0 = sss_df.loc[(ts, expiry, strike0, right), ref_iv_col]
                        iv1 = sss_df.loc[(ts, expiry, strike1, right), ref_iv_col]
                        skew = (iv1 - iv0) / (delta1 - delta0)

                    self.df.loc[(ts, expiry, slice(None), right), skew_col_nm] = skew
        print(f'enrich_iv_curvature: skew_measure={skew_measure}, Total Values: {self.df[skew_col_nm].count()}. # NA: {self.df[skew_col_nm].isna().sum()}, # Zero: {(self.df[skew_col_nm] == 0).sum()}')

    def mp_enrich_div_skew_of_ds(self, v_ds_pct, iv_col='mid_iv', n_processes=8):
        # TypeError: cannot pickle 'SwigPyObject' object
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

        print(f'enrich_div_skew_of_ds: ds%={ds_pct:.2f} , Total Values: {df[col_nm].count()}. # NA: {df[col_nm].isna().sum()}, # Zero: {(df[col_nm] == 0).sum()}')

        return df[[col_nm, col_nm_rolling]]

#     @lru_cache()
#     def yield_ts(self, calculation_date_ql: ql.Date):
#         return ql.YieldTermStructureHandle(ql.FlatForward(calculation_date_ql, DiscountRateMarket, self.day_count))
#
#     @lru_cache()
#     def dividend_ts(self, calculation_date_ql: ql.Date):
#         return ql.YieldTermStructureHandle(ql.FlatForward(calculation_date_ql, DividendYield[str(self.underlying)], self.day_count))
#
#
# def get_calibration_set(df, option_right: str, iv_col='mid_iv'):
#     calibrationSet = ql.CalibrationSet()
#
#     for expiry, s_df in df.groupby('expiry'):
#         for strike, ss_df in s_df.groupby('strike'):
#             iv = ss_df[iv_col].iloc[0]
#             if np.isnan(iv):
#                 continue
#
#             payoff = ql.PlainVanillaPayoff(str2ql_option_right(option_right), float(strike))
#             exercise = ql.EuropeanExercise(to_ql_dt(expiry))
#
#             calibrationSet.push_back((ql.VanillaOption(payoff, exercise), ql.SimpleQuote(iv)))
#
#     return calibrationSet


if __name__ == '__main__':
    sym = 'ORCL'
    sym = 'ONON'
    equity = Equity(sym)
    resolution = Resolution.second
    seq_ret_threshold = 0.005
    release_date = EarningsPreSessionDates(sym)[-1]
    marker_size = 4

    option_frame = OptionFrame.load_frame(equity, resolution, seq_ret_threshold, year_quarter(release_date))
    df = option_frame.df_options.sort_index()

    print(f'Loaded df shape: {df.shape}')
    # Small frame to debug
    # df = df.loc[(slice(None), date(2024, 3, 8), slice(None), 'call'), :]

    if True:
        iv_surface = IVSurface(equity, df)
        # iv_surface.enrich_iv_ah()
        iv_surface.enrich_iv_curvature(iv_col='mid_iv')
        # iv_surface.enrich_iv_curvature(iv_col='mid_iv_ah')
        # iv_surface.enrich_skew_measures(SkewMeasure.Delta25Delta50, ref_measure_col='delta', ref_iv_col='mid_iv')
        # iv_surface.enrich_skew_measures(SkewMeasure.Delta25Delta50, ref_measure_col='delta', ref_iv_col='mid_iv_ah')

        v_dS = [-0.1, 0.1]
        # v_dS = np.linspace(0.8, 1.2, 41)
        iv_surface.mp_enrich_div_skew_of_ds(v_dS)

    # What I need to with skew? calc dSdIV to get better risk and hedge ratio estimate...
    # For scenario modelling, IV level before after were okay. That's the skew of 2 points, start & destination.
    # Then I'd get skew measures for dS 1%, ... x%. Now these would vary by expiry and strike.

    # Further, would want to experiment with multiple smoothing techniques for the skew measures.
    # Rolling EWMA, AH IV measures, outlier removal

    # Plot skew of dS on a given date. One point per option... Only selected far expiries These plots are rather noisy because they go across days.
    for ds in [0.9, 1.1]:
        figSkewSkewDs = make_subplots(rows=4, cols=2, shared_xaxes=True, subplot_titles=("Mid IV Call", 'Mid IV Put', "Mid IV Curvature call", 'curv put', f"Skew of dS {100*ds}% call"))
        col = f'div_skew_ds_{ds-1:.2f}'
        col_nm_rolling = f'div_skew_ds_{ds-1:.2f}_rolling'
        delta_skew_col_nm = f'skew_{SkewMeasure.Delta25Delta50}'
        sampled_df = df[(df['moneyness'] > 0.8) & (df['moneyness'] < 1.2)]
        for i, (dt, s_df) in enumerate(sampled_df.groupby('date')):
            if i >= 1:
                break
            for expiry, ss_df in s_df.groupby('expiry'):
                for k, (right, sss_df) in enumerate(ss_df.groupby('right')):
                    figSkewSkewDs.add_trace(go.Scatter(x=sss_df['moneyness'], y=sss_df['mid_iv'], mode='markers', name=f'IV {right} {expiry}', marker=dict(size=marker_size)), row=1, col=k+1)
                    figSkewSkewDs.add_trace(go.Scatter(x=sss_df['moneyness'], y=sss_df['mid_iv_curvature'], mode='markers', name=f'IV Curvature {right} {expiry}', marker=dict(size=marker_size)), row=2, col=k+1)
                    figSkewSkewDs.add_trace(go.Scatter(x=sss_df['moneyness'], y=sss_df[col], mode='markers', name=f'Skew {right} {expiry}', marker=dict(size=marker_size)), row=3, col=k+1)
                    figSkewSkewDs.add_trace(go.Scatter(x=sss_df['moneyness'], y=sss_df[col_nm_rolling], mode='markers', name=f'Skew {right} {expiry}', marker=dict(size=marker_size)), row=4, col=k+1)
                    figSkewSkewDs.add_vline(x=1, line_width=1, line_dash="dash", line_color="black", row=3, col=1)
        figSkewSkewDs.add_vline(x=1, line_width=1, line_dash="dash", line_color="black", row=1, col=1)
        figSkewSkewDs.update_layout(title_text=f'Skew of dS {100*ds}%', xaxis_title='Moneyness', yaxis_title='IV')
        show(figSkewSkewDs)

    # plot skew across time. rolling average, reset at SOD.
    figRollingSkew = make_subplots(rows=2, cols=2, shared_xaxes=True, subplot_titles=("Skew Rolling Call", "Skew Rolling Put"))
    for o, ds in enumerate([-0.1, 0.1]):
        for k, (right, s_df) in enumerate(df.groupby('right')):
            for expiry, ss_df in s_df.groupby('expiry'):
                ts = ss_df.index.get_level_values('ts')
                skew_rolling = ss_df[f'div_skew_ds_{ds:.2f}_rolling']
                figRollingSkew.add_trace(go.Scatter(x=ts, y=skew_rolling, mode='lines+markers', name=f'{right} {expiry} {ds:.2f}_rolling'), row=o+1, col=k+1)
    show(figRollingSkew)

    # calculate error in estimation of dSdIV combining a dS shift * skew adjusted for a measured shift in ATM(exp). So essentially removing the effect of term structure / horizontal
    # only leaving vertical!
    # benchmark - zero skew. does it improve IV estimate for which expiries and dS?

    # Conversely, adjusted for skew, does the power law apply. Do term structures equilibrate

    print(f'Done {sym}')
