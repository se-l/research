import pickle

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from datetime import date, datetime
from typing import Tuple, List

from options.frame_builder import get_option_frame
from estimators.earnings_iv_drop_ssvi_surface.static import get_df_xy
from options.helper import exclude_outlier_quotes, get_tenor, get_moneyness_fwd, date_to_sod
from options.surfaces.processors import get_ivs_by_date, get_df_qt, scope_df
from options.types.calibration_item import df2calibration_items
from options.types.dividend import get_dividends
from options.types.enums import Resolution
from options.types.equity import Equity

from options.types.iv_surface_essvi import IVSurface, f_essvi_iv
from options.types.option import get_price_cuda
from options.types.quote_side import QuoteSide
from shared.modules.logger import info
from shared.plotting import show, Trace
from plotly.subplots import make_subplots
from options.types.enums import OptionRight
from shared.yield_curve import YieldCurve


def plot_evaluation(ivs: IVSurface, df_q, calc_date, min_max_mny_fwd_ln = (-0.3, 0.3), max_points_per_curve=800):
    iVSurfaceModelEvaluation = ivs.evaluate(min_max_mny_fwd_ln=min_max_mny_fwd_ln)

    rights = [OptionRight.call, OptionRight.put]

    v_trc: List[Trace] = []
    subplot_titles = []
    row = 1

    v_trc.append(Trace(iVSurfaceModelEvaluation.get_trace_error_metrics_bar(), row, 1))
    subplot_titles += ['Across Surface', '']
    row += 1

    error_dct_title = [
        # ('rmse_price_right_tenor',# 'RMSE Price'),
        ('logit_err_price_right_tenor', 'Logit Err Price'),
        # ('rmse_iv_right_tenor', 'RMSE IV'
        ('mae_iv_right_tenor', 'MAE IV'),
        # ('rmse_err_over_of_spread_right_tenor', 'RMSE of Spread'
        ('mae_err_over_spread_right_tenor', 'MAE of Spread'),
    ]
    error_dct_surf_title = [
        ('me_iv_by_mny', 'Mean Error IV'),
        ('me_price_by_mny', 'Mean Error Price'),
    ]
    filter_dct = lambda dct, option_right: {k[0]: v for k, v in dct.items() if k[1] == option_right}
    for attr, metric in error_dct_title:
        for right, col in zip(rights, [1, 2]):
            dct = filter_dct(getattr(iVSurfaceModelEvaluation, attr), right)
            v_trc.append(Trace(go.Scatter(x=list(dct.keys()), y=list(dct.values()), mode='markers', marker=dict(size=4), name=attr), row, col))
            subplot_titles.append(f'{right} {metric}')
        row += 1
    filter_dct = lambda dct, option_right: {(k[0], k[2]): v for k, v in dct.items() if k[1] == option_right}
    for attr, metric in error_dct_surf_title:
        for right, col in zip(rights, [1, 2]):
            dct = filter_dct(getattr(iVSurfaceModelEvaluation, attr), right)
            unique_tenors = {k[0] for k in dct.keys()}
            for tenor in unique_tenors:
                buckets = [k[1] for k in dct.keys() if k[0] == tenor]
                errors = [v for (k, v) in dct.items() if k[0] == tenor]
                v_trc.append(Trace(go.Scatter(x=buckets, y=errors, mode='markers', marker=dict(size=4), name=f'{attr} {right} {tenor}'), row, col))
            subplot_titles.append(f'{right} {metric}')
        row += 1

    # Plot calibrated interest rate
    zero_rates = YieldCurve().get_zero_curve(calc_date)
    calibrated_rates = YieldCurve().get_last_zero_curve(calc_date, equity, -14)
    v_trc.append(Trace(go.Scatter(x=zero_rates.times, y=zero_rates.rates, mode='markers', marker=dict(size=4), name=f'Default Rates'), row, 1))
    v_trc.append(Trace(go.Scatter(x=calibrated_rates.times, y=calibrated_rates.rates, mode='markers', marker=dict(size=4), name=f'Calibrated Rates'), row, 1))
    subplot_titles.append(f'YieldCurve')
    subplot_titles.append(f'')
    row += 1


    dividends = get_dividends(str(ivs.underlying), start=calc_date)
    calibration_items_bid = df2calibration_items(df_q, calc_date, iv_col_nm='bid_iv', price_col_nm='bid_close', spot_col_nm='spot', dividends=dividends, weight_col_nm=None)
    calibration_items_ask = df2calibration_items(df_q, calc_date, iv_col_nm='ask_iv', price_col_nm='ask_close', spot_col_nm='spot', dividends=dividends, weight_col_nm=None)

    # Prices & IV
    # for tenor_dt in list(ivs.calibration_items.keys())[-1:]:
    for tenor_dt in list(ivs.calibration_items.keys()):
        info(f'plot_evaluation(): {equity} {tenor_dt}')
        params = ivs.params[tenor_dt]
        v_mny_fwd_ln = ivs.calibration_items[tenor_dt].mny_fwd_ln
        ix_scoped_mny = (min_max_mny_fwd_ln[0] < v_mny_fwd_ln) & (v_mny_fwd_ln < min_max_mny_fwd_ln[1])
        v_iv_mid = ivs.calibration_items[tenor_dt].iv[ix_scoped_mny]
        v_mid_price = ivs.calibration_items[tenor_dt].price[ix_scoped_mny]
        v_spot = ivs.calibration_items[tenor_dt].spot[ix_scoped_mny]
        v_strike = ivs.calibration_items[tenor_dt].strike[ix_scoped_mny]
        v_rights = ivs.calibration_items[tenor_dt].rights[ix_scoped_mny]
        v_tenor = np.array([ivs.calibration_items[tenor_dt].tenor] * len(v_mny_fwd_ln))

        v_is_call = np.array([1 if r == OptionRight.call else 0 for r in v_rights])
        y_iv_pred = f_essvi_iv(v_mny_fwd_ln, *params, tenor=get_tenor(tenor_dt, calc_date))
        tenor = get_tenor(tenor_dt, calc_date)
        v_mny_fwd_ln = np.log(get_moneyness_fwd(equity, v_strike, v_spot, v_tenor, date_to_sod(calc_date), yield_curve_in=calibrated_rates))

        y_price_pred = get_price_cuda(s=v_spot, k=v_strike, t=np.asarray([tenor] * len(v_spot)), v_is_call=v_is_call,
                                      iv=y_iv_pred, dividends=dividends, calculation_date=calc_date, equity=ivs.underlying,
                                      yield_curve=calibrated_rates)

        # Price
        for right, col in zip(rights, [1, 2]):
            ix_right = v_rights == right
            v_mny_fwd_ln_right = v_mny_fwd_ln[ix_right]
            if calibration_items_bid is None:
                v_trc.append(Trace(go.Scatter(x=v_mny_fwd_ln_right, y=v_mid_price[ix_right], mode='markers', marker=dict(size=4), name=f'Price Mid {right} {tenor_dt}'), row, col))

            # Quotes if any
            if calibration_items_bid is not None and (i_bid := next(iter([i for i in calibration_items_bid if i.tenor_dt == tenor_dt]), None)):
                ix_right_ba = (i_bid.rights == right) & (min_max_mny_fwd_ln[0] < i_bid.mny_fwd_ln) & (i_bid.mny_fwd_ln < min_max_mny_fwd_ln[1])
                i_bid_y = i_bid.price[ix_right_ba]
                ix_sample = pd.Series(range(len(i_bid_y))).sample(min(len(i_bid_y), max_points_per_curve)).values
                x = i_bid.mny_fwd_ln[ix_right_ba][ix_sample]
                v_trc.append(Trace(go.Scatter(x=x, y=i_bid_y[ix_sample], mode='markers', marker=dict(size=2), name=f'Price Bid {right} {tenor_dt}'), row, col))
                if calibration_items_ask is not None and (i_ask := next(iter([i for i in calibration_items_ask if i.tenor_dt == tenor_dt]), None)):
                    i_ask_price_r = i_ask.price[ix_right_ba]
                    v_trc.append(Trace(go.Scatter(x=x, y=i_ask_price_r[ix_sample], mode='markers', marker=dict(size=2), name=f'Price Ask {right} {tenor_dt}'), row, col))

            ix_sample = pd.Series(range(len(v_mny_fwd_ln_right))).sample(min(len(v_mny_fwd_ln_right), max_points_per_curve)).values
            v_trc.append(Trace(go.Scatter(x=v_mny_fwd_ln_right[ix_sample], y=y_price_pred[ix_right][ix_sample], mode='markers', marker=dict(size=3), name=f'Y Price {right} {tenor_dt}'), row, col))
            subplot_titles.append(f'Price {right} {ivs.underlying} expiry: {tenor_dt}')
        row += 1

        # IV
        for right, col in zip(rights, [1, 2]):
            ix_right = v_rights == right
            v_mny_fwd_ln_right = v_mny_fwd_ln[ix_right]

            if calibration_items_bid is None:
                v_trc.append(Trace(go.Scatter(x=v_mny_fwd_ln_right, y=v_iv_mid[ix_right], mode='markers', marker=dict(size=4), name=f'IV Mid {right} {tenor_dt}'), row, col))

            # Quotes if any. The ask/bid IV would need to recalculate with the calibrated rates...
            if calibration_items_bid is not None and (i_bid := next(iter([i for i in calibration_items_bid if i.tenor_dt == tenor_dt]), None)):
                ix_right_ba = (i_bid.rights == right) & (min_max_mny_fwd_ln[0] < i_bid.mny_fwd_ln) & (i_bid.mny_fwd_ln < min_max_mny_fwd_ln[1])
                i_bid_y = i_bid.iv[ix_right_ba]
                ix_sample = pd.Series(range(len(i_bid_y))).sample(min(len(i_bid_y), max_points_per_curve)).values
                x = i_bid.mny_fwd_ln[ix_right_ba][ix_sample]
                v_trc.append(Trace(go.Scatter(x=x, y=i_bid_y[ix_sample], mode='markers', marker=dict(size=2), name=f'IV Bid {right} {tenor_dt}'), row, col))
                if calibration_items_ask is not None and (i_ask := next(iter([i for i in calibration_items_ask if i.tenor_dt == tenor_dt]), None)):
                    i_ask_iv_r = i_ask.iv[ix_right_ba]
                    v_trc.append(Trace(go.Scatter(x=x, y=i_ask_iv_r[ix_sample], mode='markers', marker=dict(size=2), name=f'IV Ask {right} {tenor_dt}'), row, col))

            ix_sample = pd.Series(range(len(v_mny_fwd_ln_right))).sample(min(len(v_mny_fwd_ln_right), max_points_per_curve)).values
            v_trc.append(Trace(go.Scatter(x=v_mny_fwd_ln_right[ix_sample], y=y_iv_pred[ix_right][ix_sample], mode='markers', marker=dict(size=3), name=f'Y IV {right} {tenor_dt}'), row, col))
            subplot_titles.append(f'IV {right} {ivs.underlying} expiry: {tenor_dt}')

            # v_weights = ivs.calibration_items[tenor_dt].weights
            # fig.add_trace(go.Scatter(x=v_mny_fwd_ln_right, y=v_weights[ix_right], mode='markers', marker=dict(size=4), name='weights'), row=2, col=col)
            # subplot_titles.append(f'Weights {right}')
            # v_vega = ivs.calibration_items[tenor_dt].vega
            # if v_vega is not None:
            #     fig.add_trace(go.Scatter(x=v_mny_fwd_ln_right, y=v_vega, mode='markers', marker=dict(size=4), name='vega'), row=3, col=col)
            # fig.add_trace(go.Histogram(x=v_mny_fwd_ln_right, nbinsx=int((max(v_mny_fwd_ln_right) - min(v_mny_fwd_ln_right)) * 100), name='count'), row=3, col=col)
            # subplot_titles.append(f'Sample Count Hist {right}')
        row += 1

    n_cols = max((trc.col for trc in v_trc))
    n_rows = max((trc.row for trc in v_trc))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
    fig.update_layout(
        height=350 * n_rows,
        margin=dict(t=40, b=20),
    )
    for trc in v_trc:
        fig.add_trace(trc.trace, row=trc.row, col=trc.col)

    ts_start = None
    ts_end = None
    for t, items in ivs.calibration_items.items():
        ts_start = items.ts[0] if ts_start is None else min(items.ts[0], ts_start)
        ts_end = items.ts[-1] if ts_end is None else max(items.ts[-1], ts_end)

    fig.update_layout(
        title=dict(
            text=f"IVS {ivs.underlying}: {ts_start.isoformat()} - {ts_end.isoformat()}",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
        margin=dict(t=80)  # make room for the title
    )
    show(fig)


def get_dfs(equity, dates, resolution, seq_ret_threshold) -> Tuple[pd.DataFrame, pd.DataFrame]:
    option_frame = get_option_frame(equity, dates, resolution=resolution, seq_ret_threshold=seq_ret_threshold)
    df = option_frame.df_options.sort_index()
    df_trades = option_frame.df_option_trades.sort_index()
    # A little cleaning
    df_trades = exclude_outlier_quotes(df_trades, [], equity)
    df_q = get_df_xy(df, dates)
    df_t = get_df_xy(df_trades, dates)
    return df_q, df_t

if __name__ == '__main__':
    equity = Equity('FDX')
    dates = [date(2024, 3, 20)]
    resolution = Resolution.second
    seq_ret_threshold = 0.002
    seq_ret_threshold_surface = None
    side = 'mid'
    arb_free = False
    calc_date = dates[0]

    equity = Equity("DELL")
    n_samples_per_tenor = 500
    dates = [date(2024, 8, 29)]
    side = QuoteSide.mid
    resolution = Resolution.second
    seq_ret_threshold = 0.002

    option_frame = get_option_frame(equity, dates, resolution=resolution, seq_ret_threshold=seq_ret_threshold)
    df_q, df_t = get_df_qt(option_frame, equity, dates)

    start = datetime(1970, 1, 1)
    end = datetime(2099, 1, 1)
    calc_date = df_q.index.get_level_values('ts')[0].date()
    df_q, df_t = scope_df(df_q, start, end), scope_df(df_t, start, end)

    # with open(r'D:\ivs.pkl', 'rb') as f:
    #     ivs = pickle.load(f)
    # plot_evaluation(ivs, df_q, calc_date)
    # with open(r'D:\ivs_fp_jl.pkl', 'rb') as f:
    #     ivs_fp_jl = pickle.load(f)
    # plot_evaluation(ivs_fp_jl, df_q, calc_date)
    with open(r'D:\ivs_jl.pkl', 'rb') as f:
        ivs_jl = pickle.load(f)
    plot_evaluation(ivs_jl, df_q, calc_date)
    # v_ivs = get_ivs_by_date(equity, dates, resolution, seq_ret_threshold, side, arb_free, seq_ret_threshold_surface=seq_ret_threshold_surface)
    # df_q, df_t = get_dfs(equity, dates, resolution, seq_ret_threshold=seq_ret_threshold)
    # plot_evaluation(v_ivs[-1], df_q, calc_date)
