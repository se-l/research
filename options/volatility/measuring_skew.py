import os
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import date, timedelta
from plotly.subplots import make_subplots

from options.helper import year_quarter, atm_iv
from options.typess.equity import Equity
from options.typess.enums import Resolution, OptionRight
from options.typess.iv_surface import enrich_mean_regressed_skew_for_ds
from options.typess.option_frame import OptionFrame
from options.volatility.estimators.earnings_iv_drop_poly_regressor import EarningsIVDropPolyRegressorV3
from shared.constants import EarningsPreSessionDates
from shared.paths import Paths
from shared.plotting import show


def run(take=-1):
    tickers = 'DG,ORCL,ONON,DLTR,CSCO,CRWD,PATH,DELL,TGT,JD,WDAY,PANW,MRVL'
    # tickers = 'TGT'
    for sym in tickers.split(','):
        sym = sym.lower()
        equity = Equity(sym)
        # start, end = earnings_download_dates(sym, take=take)
        resolution = Resolution.minute
        seq_ret_threshold = 0.005
        release_date = EarningsPreSessionDates(sym)[take]
        option_frame = OptionFrame.load_frame(equity, resolution, seq_ret_threshold, year_quarter(release_date))
        df = option_frame.df_options.sort_index()
        df['strike'] = df.index.get_level_values('strike').astype(float)

        ts = df.index.get_level_values('ts').unique()
        v_ts_pre_release = [i for i in ts if i.date() == release_date if i.hour >= 10]
        v_ts_post_release = [i for i in ts if i.date() == (release_date + timedelta(days=1)) if i.hour >= 10]

        # get_error_skew(df.loc[v_ts_pre_release], df.loc[v_ts_post_release])

        plot_horizontal_iv_curve_change(df.loc[v_ts_pre_release], df.loc[v_ts_post_release], sym)

        # plot how much vertical and horizontal iv curves have changed
        # plot dATM_IV / tenor and dIV_strike / moneyness grouped by expiry.

    # Separate script
    # Most relevant metric to estimate dIV / tenor.  Assume skew constant at first.
    # Plot it for as many release as available...
    # First model: Regress through'em. So dATM_IV = f(tenor)
    # Second model: dIV, not just ATM/level, but strike/moneyness dependent. Incorporate that steep side drops less than flat part.
    # Improvement: More features, like past IV ... IV trend, estimated diffusion IV.


def plot_horizontal_iv_curve_change(df0, df1, sym=None):
    dt_range0 = f'''{min(df0.index.get_level_values('ts').unique()).date()}-{max(df0.index.get_level_values('ts').unique()).date()}'''
    dt_range1 = f'''{min(df1.index.get_level_values('ts').unique()).date()}-{max(df1.index.get_level_values('ts').unique()).date()}'''
    level_groupby = ('expiry', 'strike', 'right')
    s0 = df0['spot'].mean()
    s1 = df1['spot'].mean()
    ds = s1 - s0

    # Term structure level change of IV by expiry and right
    for df in [df0, df1]:
        df['atm_iv'] = None
        for expiry, s_df in df.groupby(level='expiry'):
            for right, ss_df in s_df.groupby(level='right'):
                for ts, sss_df in ss_df.groupby(level='ts'):
                    s = sss_df.iloc[0]['spot']
                    v_atm_iv = atm_iv(sss_df.loc[ts], expiry, s, right=right)
                    df.loc[sss_df.index, 'atm_iv'] = v_atm_iv

    # exp_d_iv_skew_regressed_curvature = (df0['mid_iv_curvature_regressed'] * -ds).groupby(level=level_groupby).mean()
    # exp_d_iv_skew_regressed_curvature_mean = (enrich_mean_regressed_skew_for_ds(ds, df0) * -ds).groupby(level=level_groupby).mean()    # Averages all skews from K(ds=0) to K(ds).

    # The Model - expand this. Further clarify the error to minimize
    # 1. A constant d_iv tenor across all expiries and rights: -2%. 
    # df0['atm_iv_estimated'] = df0['atm_iv'] - 0.02
    # df0['mid_iv_estimated'] = df0['mid_iv'] - 0.02

    path_model = os.path.join(Paths.path_models, f'earnings_iv_drop_regressor_{date.today()}b.json')
    model = EarningsIVDropPolyRegressorV3().load_model(path_model)
    d_iv_pct = model.predict(df0)
    df0['atm_iv_estimated'] = df0['atm_iv'] + df0['atm_iv'] * d_iv_pct

    # 2. Average of all percentage drop by expiry across assets

    # Estimate delta PnL with the respect to skill of execution. Plot bids/asks/trades.
    # 3. Create feature frame, plot x/y look at things, e.g., SPY IV, assets last month's IV, !!!similar assets and their IV drops!!! similar by which metrics???

    # ix_s = lambda df: df.loc[(slice(None), expiries, slice(None))].index
    # df0_s = df0.loc[ix_s(df0)]
    # df1_s = df1.loc[ix_s(df1)]

    agg_funcs = dict(mid_iv='mean', moneyness='mean', atm_iv_estimated='mean', atm_iv='mean', tenor='mean')
    agg1_funcs = dict(mid_iv='mean', moneyness='mean', atm_iv='mean', tenor='mean')
    df0_agg = df0.groupby(level=level_groupby).agg(agg_funcs)
    df1_agg = df1.groupby(level=level_groupby).agg(agg1_funcs)

    scoped_expiries = df0[df0['tenor'] > 0.2].index.get_level_values('expiry').unique()

    figH = make_subplots(rows=1, cols=3, shared_xaxes=True, subplot_titles=['Measured', 'Estimated', 'Error'])
    for right in [OptionRight.call, OptionRight.put]:
        ix = (slice(None), scoped_expiries, slice(None), right)
        ix_agg = (scoped_expiries, slice(None), right)
        # Horizontal: Plot over tenor. Measured Before n after earnings
        figH.add_trace(go.Scatter(x=df0.loc[ix, 'tenor'], y=df0.loc[ix, 'atm_iv'], mode='markers', name=f'{right} IV Term Structure @ {dt_range0}', marker=dict(size=2)), row=1, col=1)
        figH.add_trace(go.Scatter(x=df0_agg.loc[ix_agg, 'tenor'], y=df0_agg.loc[ix_agg, 'atm_iv'], mode='lines+markers', name=f'{right} IV Term Structure @ {dt_range0}', marker=dict(size=4)), row=1, col=1)

        figH.add_trace(go.Scatter(x=df1.loc[ix, 'tenor'], y=df1.loc[ix, 'atm_iv'], mode='markers', name=f'{right} IV Term Structure @ {dt_range1}', marker=dict(size=2)), row=1, col=1)
        figH.add_trace(go.Scatter(x=df1_agg.loc[ix_agg, 'tenor'], y=df1_agg.loc[ix_agg, 'atm_iv'], mode='lines+markers', name=f'{right} IV Term Structure @ {dt_range1}', marker=dict(size=4)), row=1, col=1)

        # Horizontal: Plot over tenor. Measured vs Estimated
        figH.add_trace(go.Scatter(x=df1.loc[ix, 'tenor'], y=df1.loc[ix, 'atm_iv'], mode='markers', name=f'{right} IV Term Structure @ {dt_range1}', marker=dict(size=2)), row=1, col=2)
        figH.add_trace(go.Scatter(x=df1_agg.loc[ix_agg, 'tenor'], y=df1_agg.loc[ix_agg, 'atm_iv'], mode='lines+markers', name=f'{right} IV Term Structure @ {dt_range1}', marker=dict(size=4)), row=1, col=2)

        figH.add_trace(go.Scatter(x=df0.loc[ix, 'tenor'], y=df0.loc[ix, 'atm_iv_estimated'], mode='markers', name=f'{right} estimated IV Term Structure @ {dt_range0}', marker=dict(size=2)), row=1, col=2)
        figH.add_trace(go.Scatter(x=df0_agg.loc[ix_agg, 'tenor'], y=df0_agg.loc[ix_agg, 'atm_iv_estimated'], mode='lines+markers', name=f'{right} estimated IV Term Structure @ {dt_range0}', marker=dict(size=4)), row=1, col=2)

        # Error
        ix_intersect = df0.loc[ix].index.droplevel(0).intersection(df1.loc[ix].index.droplevel(0))
        df1_atm = df1.loc[ix, ['tenor', 'atm_iv']].groupby(level=('expiry', 'strike')).mean().loc[ix_intersect]
        df1_atm['err'] = df1_atm['atm_iv'] - df0.loc[ix, ['tenor', 'atm_iv_estimated']].groupby(level=('expiry', 'strike')).mean().loc[ix_intersect]['atm_iv_estimated']

        figH.add_trace(go.Scatter(x=df1_atm['tenor'], y=df1_atm['err'], mode='lines+markers', name=f'{right} MAE IV Term Structure', marker=dict(size=4)), row=1, col=3)
        # figH.add_trace(go.Scatter(x=df1_atm['tenor'], y=df1_atm['err']**2, mode='lines+markers', name=f'{right} RMSE IV Term Structure', marker=dict(size=4)), row=1, col=3)
    figH.update_layout(title_text=f'{sym} IV Term Structure Change {dt_range0} vs {dt_range1}')
    show(figH)

    # fig = make_subplots(rows=len(scoped_expiries), cols=3, shared_xaxes=True, subplot_titles=['Measured', 'Estimated', 'Error'])
    # # Vertical: Plot over moneyness
    # for i, expiry in enumerate(scoped_expiries):
    #     for right in [OptionRight.call, OptionRight.put]:
    #         ix = (slice(None), expiry, slice(None), right)
    #         ix_agg = (expiry, slice(None), right)
    #         try:
    #             df0_s = df0.loc[ix]
    #             df0_s_agg = df0_agg.loc[ix_agg]
    #             df1_s = df1.loc[ix]
    #             df1_s_agg = df1_agg.loc[ix_agg]
    #         except KeyError as e:
    #             print(f'KeyError: {e}')
    #             continue
    #         row = i+1
    #         # Raw IVs
    #         fig.add_trace(go.Scatter(x=df0_s['moneyness'], y=df0_s['mid_iv'], mode='markers', name=f'{right} IV Vertical Structure @ {dt_range0}', marker=dict(size=3)), row=row, col=1)
    #         fig.add_trace(go.Scatter(x=df0_s_agg['moneyness'], y=df0_s_agg['mid_iv'], mode='lines+markers', name=f'{right} IV Vertical Structure @ {dt_range0}', marker=dict(size=3)), row=row, col=1)
    #
    #         fig.add_trace(go.Scatter(x=df1_s['moneyness'], y=df1_s['mid_iv'], mode='markers', name=f'{right} IV Vertical Structure @ {dt_range1}', marker=dict(size=3)), row=row, col=1)
    #         fig.add_trace(go.Scatter(x=df1_s_agg['moneyness'], y=df1_s_agg['mid_iv'], mode='markers', name=f'{right} IV Vertical Structure @ {dt_range1}', marker=dict(size=3)), row=row, col=1)
    #
    #         # Estimated IVs
    #         fig.add_trace(go.Scatter(x=df1_s['moneyness'], y=df1_s['mid_iv'], mode='markers', name=f'{right} IV Vertical Structure @ {dt_range1}', marker=dict(size=3)), row=row, col=2)
    #         fig.add_trace(go.Scatter(x=df1_s_agg['moneyness'], y=df1_s_agg['mid_iv'], mode='markers', name=f'{right} IV Vertical Structure @ {dt_range1}', marker=dict(size=3)), row=row, col=2)
    #
    #         fig.add_trace(go.Scatter(x=df0_s['moneyness'], y=df0_s['mid_iv'], mode='markers', name=f'{right} IV Vertical Structure @ {dt_range0}', marker=dict(size=3)), row=row, col=2)
    #         fig.add_trace(go.Scatter(x=df0_s_agg['moneyness'], y=df0_s_agg['mid_iv'], mode='markers', name=f'{right} IV Vertical Structure @ {dt_range0}', marker=dict(size=3)), row=row, col=2)
    #
    #         # Error
    #         fig.add_trace(go.Scatter(x=df1_s['moneyness'], y=df1_s['mid_iv'].droplevel(0) - df0_s['mid_iv_estimated'].droplevel(0), mode='markers', name=f'{right} MAE IV Vertical Structure', marker=dict(size=3)), row=row, col=3)
    #         # fig.add_trace(go.Scatter(x=df1_s['moneyness'], y=(df1_s['mid_iv'] - df0_s['mid_iv_estimated'])**2, mode='markers', name=f'{right} RMSE IV Vertical Structure', marker=dict(size=3)), row=row, col=3)
    #
    #         # fig.add_trace(go.Scatter(x=df1_s['moneyness'], y=(df1_s['mid_iv'] - df0_s['mid_iv_estimated']).mean(axis=1), mode='lines+markers', name=f'{right} MAE IV Vertical Structure', marker=dict(size=3)), row=row, col=1)
    #         # fig.add_trace(go.Scatter(x=df1_s['moneyness'], y=((df1_s['mid_iv'] - df0_s['mid_iv_estimated']) ** 2).mean(axis=1), mode='lines+markers', name=f'{right} RMSE IV Vertical Structure', marker=dict(size=3)), row=row, col=1)
    #
    # show(fig)


def get_error_skew(df0, df1):
    level_groupby = ('expiry', 'strike', 'right')
    s0 = df0['spot'].mean()
    s1 = df1['spot'].mean()
    ds = s1 - s0

    # Term structure level change of IV by expiry and right
    for df in [df0, df1]:
        df['atm_iv'] = None
        df['d_atm_iv'] = None
        for expiry, s_df in df.groupby(level='expiry'):
            for right, ss_df in s_df.groupby(level='right'):
                for ts, sss_df in ss_df.groupby(level='ts'):
                    s = sss_df.iloc[0]['spot']
                    v_atm_iv = atm_iv(sss_df.loc[ts], expiry, s, right=right)
                    df.loc[sss_df.index, 'atm_iv'] = v_atm_iv

    exp_d_iv_skew_regressed_curvature = (df0['mid_iv_curvature_regressed'] * -ds).groupby(level=level_groupby).mean()
    exp_d_iv_skew_regressed_curvature_mean = enrich_mean_regressed_skew_for_ds(ds, df0) * -ds    # Averages all skews from K(ds=0) to K(ds).
    exp_d_iv_skew_regressed_curvature_mean = exp_d_iv_skew_regressed_curvature_mean.groupby(level=level_groupby).mean()

    expiries = df0.index.get_level_values('expiry').unique()
    scoped_expiries = expiries[expiries > date(2024, 6, 1)]

    ix_scoped = lambda df: df.loc[(slice(None), scoped_expiries, slice(None))][(df['moneyness'] > 0.8) & (df['moneyness'] < 1.2)].index
    df0_scoped = df0.loc[ix_scoped(df0)]
    df1_scoped = df1.loc[ix_scoped(df1)]

    df0_scoped_esr_mid_iv = df0_scoped['mid_iv'].dropna().groupby(level=level_groupby).mean()
    df1_scoped_esr_mid_iv = df1_scoped['mid_iv'].dropna().groupby(level=level_groupby).mean()
    d_iv = df1_scoped_esr_mid_iv - df0_scoped_esr_mid_iv.dropna()
    d_iv_ts = df1['atm_iv'].groupby(level='expiry').mean() - df0['atm_iv'].groupby(level='expiry').mean()

    # def skew_error_report():
    @dataclass
    class SkewErrorReport:
        category: str
        min: float
        max: float
        mean: float
        std: float

    results = []
    # Error of skew

    mi, ma, mean, std = val_by_expiry_right(d_iv)
    results.append(SkewErrorReport('d_iv', mi, ma, mean, std))

    # d_iv_ts = {expiry: df.loc[expiry, 'd_atm_iv'].mean() for expiry in scoped_expiries}
    # print(f'd_iv_ts: {d_iv_ts}')

    mi, ma, mean, std = val_by_expiry_right(df1_scoped_esr_mid_iv - df0_scoped_esr_mid_iv + d_iv_ts)
    results.append(SkewErrorReport('d_iv + d_iv_ts', mi, ma, mean, std))

    # mi, ma, mean, std = val_by_expiry_right(df1_scoped_esr_mid_iv - (df0_scoped_esr_mid_iv + exp_d_iv_skew_rolling))
    # results.append(SkewErrorReport('d_iv - exp_d_iv_skew_rolling', mi, ma, mean, std))
    #
    # mi, ma, mean, std = val_by_expiry_right(df1_scoped_esr_mid_iv - (df0_scoped_esr_mid_iv + d_iv_ts + exp_d_iv_skew_rolling))
    # results.append(SkewErrorReport('d_iv + d_iv_ts - exp_d_iv_skew_rolling', mi, ma, mean, std))
    #
    # mi, ma, mean, std = val_by_expiry_right(df1_scoped_esr_mid_iv - (df0_scoped_esr_mid_iv + d_iv_ts + exp_d_iv_skew))
    # results.append(SkewErrorReport('d_iv + d_iv_ts - exp_d_iv_skew', mi, ma, mean, std))

    # BEST performer in TGT test
    mi, ma, mean, std = val_by_expiry_right(df1_scoped_esr_mid_iv - df0_scoped_esr_mid_iv + d_iv_ts + exp_d_iv_skew_regressed_curvature)
    results.append(SkewErrorReport('d_iv + d_iv_ts + exp_d_iv_skew_regressed_curvature', mi, ma, mean, std))
    print('-'*100)

    mi, ma, mean, std = val_by_expiry_right(df1_scoped_esr_mid_iv - df0_scoped_esr_mid_iv + d_iv_ts + exp_d_iv_skew_regressed_curvature_mean)
    results.append(SkewErrorReport('d_iv + d_iv_ts + exp_d_iv_skew_regressed_curvature_mean', mi, ma, mean, std))
    print('-'*100)

    # More accuracy is expected when d_iv_skew is averaged over [skew0, skew1].
    # Skew is expected to become a tad more negative, meaning OTM put/ITM calls maintain their IV while ~ATM option's IVs drop.

    pd.options.display.max_columns = 100
    print(pd.DataFrame(results))


def val_by_expiry_right(df):
    vals = []
    for expiry, s_df in df.groupby(level='expiry'):
        for right, ss_df in s_df.groupby(level='right'):
            vals.append(ss_df.mean())
    vals = [v for v in vals if not pd.isna(v)]
    return min(vals), max(vals), sum(vals) / len(vals), np.std(vals)


if __name__ == '__main__':
    """
    skew measurement must start off a fairly smooth surface. present?
    Further, given a skew is ideally described as 4rd moment, there should be differentiable, consistent function. Would remove the problem with edge volas.
    So an improvement could be 2 fit a quadratic function through each skew. 
    Further, the larger the tenor, the flatter the skew mostly, If not, gotta alert.
    """
    run()
