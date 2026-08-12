import numpy as np
import pickle
import pandas as pd

from multiprocessing import Pool
from typing import List
from datetime import timedelta, date
from connector.api_minlp.common import exclude_outlier_quotes
from options.frame_builder import get_option_frame, check_data_presence
from options.helper import get_dividend_yield
from options.surfaces.processors import get_df_xy
from options.types.calibration_item import df2calibration_items, union_calibration_items
from options.types.enums import Resolution, OptionRight
from options.types.equity import Equity
from options.types.sym_date import SymDate
from shared.constants import EarningsPreSessionDates, DiscountRateMarket
from options.types.iv_surface_essvi import IVSurface


xy_store_fn = 'earnings_iv_drop_srf.pkl'
moneyness_fit = 'moneyness_fwd_ln'
iv_col_nm = 'mid_price_iv'
weight_col_nm = 'volume'
min_mny = -0.3
max_mny = 0.3


def store_xy_mp(sym_dates: List[SymDate], plot=True, mp=True):
    payloads = []
    for sym_date in sym_dates:
        sym, release_date = sym_date.symbol, sym_date.date
        if not check_data_presence(SymDate(sym, release_date)):
            continue
        sym = sym.lower()
        equity = Equity(sym)
        # start, end = earnings_download_dates(sym, take=take)
        resolution = Resolution.second
        seq_ret_threshold = 0.002

        if release_date >= date.today():
            print(f'{sym} earnings date is in the future.')
            continue
        payloads.append((equity, release_date, resolution, seq_ret_threshold, plot))

    if mp:
        with Pool(4) as pool:
            v_xy = pool.starmap(store_xy, payloads)
    else:
        v_xy = [store_xy(*p) for p in payloads]
    v_xy = [x for x in v_xy if x is not None]

    xy = pd.concat(v_xy)
    # scale_vega(xy, weight_col_nm)

    with open(xy_store_fn, 'wb') as f:
        pickle.dump(xy, f)


def scope_df(x: pd.DataFrame):
    return x[(x[moneyness_fit] < max_mny) & (x[moneyness_fit] > min_mny)]


def store_xy(equity: Equity, release_date: date, resolution: Resolution, seq_ret_threshold: float, plot=True) -> pd.DataFrame | None:
    sym = equity.symbol.lower()
    rate = DiscountRateMarket
    dividend_yield = get_dividend_yield(sym)

    dates = [release_date]  # - timedelta(days=28)]
    dates += [d.date() for d in (pd.date_range(release_date, release_date + timedelta(days=1), freq='D'))]
    dates = sorted(set(dates))
    release_date_p1 = dates[-1]

    option_frame = get_option_frame(equity, dates, resolution=resolution, seq_ret_threshold=seq_ret_threshold)
    df = option_frame.df_options.sort_index()
    df_trades = option_frame.df_option_trades.sort_index()
    df_trades = exclude_outlier_quotes(df_trades, [], equity)

    df_q_pre = get_df_xy(df, [release_date])
    df_q_post = get_df_xy(df, [release_date_p1])
    df_t_pre = get_df_xy(df_trades, [release_date])
    df_t_post = get_df_xy(df_trades, [release_date_p1])
    ivs_0 = IVSurface(equity)

    # weight_fills_over_quotes = 1
    # cnt_t_by_expiry_right = df_t_pre.groupby(['expiry', 'right']).count()['close'].to_dict()
    # cnt_q_by_expiry_right = df_q_pre.groupby(['expiry', 'right']).count()['moneyness'].to_dict()

    calibration_items_bid = df2calibration_items(scope_df(df_q_pre), release_date, iv_col_nm='bid_iv', price_col_nm='bid_close', spot_col_nm='spot', rf=rate, dividend_yield=dividend_yield, weight_col_nm='vega_mid_iv')
    calibration_items_ask = df2calibration_items(scope_df(df_q_pre), release_date, iv_col_nm='ask_iv', price_col_nm='ask_close', spot_col_nm='spot', rf=rate, dividend_yield=dividend_yield, weight_col_nm='vega_mid_iv')
    calibration_items_quotes = df2calibration_items(scope_df(df_q_pre), release_date, iv_col_nm='mid_price_iv', price_col_nm='mid_price', spot_col_nm='spot', rf=rate, dividend_yield=dividend_yield, weight_col_nm='vega_mid_iv')
    calibration_items_fill = df2calibration_items(scope_df(df_t_pre), release_date, iv_col_nm='fill_iv', price_col_nm='close', spot_col_nm='spot', rf=rate, dividend_yield=dividend_yield, weight_col_nm='vega_fill_iv', vega_col_nm='vega_fill_iv')

    # Weight them somewhat making fills equal to quotes
    calibration_items = union_calibration_items(calibration_items_quotes, calibration_items_fill)

    # Fit entire surface
    for right in [OptionRight.call, OptionRight.put]:
        samples = {item.tenor_dt: item for item in calibration_items if item.right == right}
        # ivs_0.calibrate_surface(right, samples)
        ivs_0.calibrate_surface_arb_free(right, samples)
    ivs_0.plot_model_params(release_date)
    ivs_0.evaluation.plot()
    # ivs_0.plot_all_prices(release_date, calibration_items_bid=calibration_items_bid, calibration_items_ask=calibration_items_ask)
    # ivs_0.plot_all_smiles(release_date, calibration_items_bid=calibration_items_bid, calibration_items_ask=calibration_items_ask)
    ivs_0.plot_surface_calibration_items()

    # For article
    # ix_worst_slice = ivs_0.ps_rmse().index[ivs_0.ps_rmse().argmax()]
    # ix_worst_slice = (date(2024, 7, 5), 'put')
    # ivs_0.plot_smile(*ix_worst_slice, release_date, calibration_items_bid=calibration_items_bid, calibration_items_ask=calibration_items_ask)
    # ix_best_slice = ivs_0.ps_rmse().index[ivs_0.ps_rmse().argmax()]
    # ix_best_slice = (date(2024, 9, 20), 'call')
    # ivs_0.plot_smile(*ix_best_slice, release_date, calibration_items_bid=calibration_items_bid, calibration_items_ask=calibration_items_ask)

    # T1
    ivs_1 = IVSurface(equity)

    calibration_items_post_fill = df2calibration_items(scope_df(df_t_post), release_date_p1, iv_col_nm='fill_iv', price_col_nm='close', spot_col_nm='spot', rf=rate, dividend_yield=dividend_yield, weight_col_nm='vega_fill_iv', vega_col_nm='vega_fill_iv')
    calibration_items_post_bid = df2calibration_items(scope_df(df_q_post), release_date_p1, iv_col_nm='bid_iv', price_col_nm='bid_close', spot_col_nm='spot', rf=rate, dividend_yield=dividend_yield, weight_col_nm='vega_mid_iv')
    calibration_items_post_ask = df2calibration_items(scope_df(df_q_post), release_date_p1, iv_col_nm='ask_iv', price_col_nm='ask_close', spot_col_nm='spot', rf=rate, dividend_yield=dividend_yield, weight_col_nm='vega_mid_iv')
    calibration_items_post_quotes = df2calibration_items(scope_df(df_q_post), release_date_p1, iv_col_nm='mid_price_iv', price_col_nm='mid_price', spot_col_nm='spot', rf=rate, dividend_yield=dividend_yield, weight_col_nm='vega_mid_iv')
    calibration_items_post = calibration_items_post_fill + calibration_items_post_quotes

    # For article
    # ix_worst_slice = ivs_1.ps_rmse().index[ivs_1.ps_rmse().argmax()]
    # ivs_1.plot_smile(*ix_worst_slice, release_date, calibration_items_bid=calibration_items_bid, calibration_items_ask=calibration_items_ask)
    # ix_best_slice = ivs_1.ps_rmse().index[ivs_1.ps_rmse().argmax()]
    # ivs_1.plot_smile(*ix_best_slice, release_date, calibration_items_bid=calibration_items_bid, calibration_items_ask=calibration_items_ask)

    # Fit entire surface
    for right in [OptionRight.call, OptionRight.put]:
        samples = {item.tenor_dt: item for item in calibration_items_post if item.right == right}
        ivs_1.calibrate_surface(equity, right, samples)
        # ivs_1.calibrate_surface_arb_free(right, samples)
    ivs_1.plot_model_params(release_date_p1)
    # ivs_1.plot_all_prices(release_date_p1, calibration_items_bid=calibration_items_post_bid, calibration_items_ask=calibration_items_post_ask)
    # ivs_1.plot_all_smiles(release_date_p1, calibration_items_bid=calibration_items_post_bid, calibration_items_ask=calibration_items_post_ask)
    ivs_1.plot_surface_calibration_items()


if __name__ == "__main__":
    """
    1. Compared to previous model, calculate the pct drop off the estimated jump IV.
        Use data from 1 month ago as most simple example.
        If promising, load more data to to estimated front month diffusion IV.
    2. Test using moneyness_fwd and moneyness_fwd_ln.
    3. Then may wanna focus on market IV, volume, open interest and another modeling strategy.
    """
    import os
    tickers = ','.join(os.listdir(r'D:\trade\data\option\usa\second')).upper()
    exclude = ['UNH', 'GS', 'ON']
    exclude += ['ORCL', 'DELL', 'ONON']
    tickers = ','.join([e for e in tickers.split(',') if e not in exclude])

    print(tickers)
    sym_dates = []
    for sym in tickers.split(','):
        for release_date in EarningsPreSessionDates(sym)[:3]:
            if check_data_presence(SymDate(sym, release_date)):
                sym_dates.append(SymDate(sym, release_date))

    store_xy_mp(sym_dates, plot=True, mp=True)
