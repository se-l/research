import QuantLib as ql
import datetime
import pandas as pd
import numpy as np

from importlib import reload
from options.helper import to_ql_dt, str2ql_option_right, get_tenor

import options.client as mClient
reload(mClient)
client = mClient.Client()


def get_calibration_set(df, ts, option_right: str, col_iv: str):
    calibrationSet = ql.CalibrationSet()
    for expiry, sub_df in df.groupby(level='expiry'):
         for strike, sub_sub_df in sub_df.groupby(level='strike'):
             ix = (ts, expiry, strike, option_right)
             if ix not in sub_sub_df.index:
                 continue
             iv = sub_sub_df.loc[ix, col_iv]
             if np.isnan(iv):
                 continue
             payoff = ql.PlainVanillaPayoff(str2ql_option_right(option_right), float(strike))
             exercise = ql.EuropeanExercise(to_ql_dt(expiry))
             calibrationSet.push_back((ql.VanillaOption(payoff, exercise), ql.SimpleQuote(iv)))
    return calibrationSet


def ah_vol_interpolation(df: pd.DataFrame, ts: datetime.datetime, option_right: str, spot: float, rate: float, dividend: float) -> ql.AndreasenHugeVolatilityInterpl:
    spot_quote = ql.QuoteHandle(ql.SimpleQuote(spot))
    calibration_set = get_calibration_set(df, ts, option_right, 'mid_iv')
    yield_ts = ql.YieldTermStructureHandle(ql.FlatForward(to_ql_dt(ts), rate, day_count))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(to_ql_dt(ts), dividend, day_count))
    return ql.AndreasenHugeVolatilityInterpl(calibration_set, spot_quote, yield_ts, dividend_ts)


def ah_vol_surface(df: pd.DataFrame, ts: datetime.datetime, option_right: str, spot: float, rate: float, dividend: float) -> ql.AndreasenHugeVolatilityAdapter:
    ah_vol_surface = ah_vol_interpolation(df, ts, option_right, spot, rate, dividend)
    return ql.AndreasenHugeVolatilityAdapter(ah_vol_surface)


def ah_lv_surface(df: pd.DataFrame, ts: datetime.datetime, option_right: str, spot: float, rate: float, dividend: float) -> ql.AndreasenHugeLocalVolAdapter:
    ah_vol_surface = ah_vol_interpolation(df, ts, option_right, spot, rate, dividend)
    return ql.AndreasenHugeLocalVolAdapter(ah_vol_surface)


def safe_black_vol(srf: ql.AndreasenHugeVolatilityAdapter, t: float, k: float) -> float:
    try:
        return srf.blackVol(t, k)
    except RuntimeError as e:
        # print(e)
        return np.nan


def ah_dVoldK(srf: ql.AndreasenHugeVolatilityAdapter, t: float, k: float, step=0.5) -> float:
    try:
        return (srf.blackVol(t, k+step) - srf.blackVol(t, k-step)) / 2*step
    except RuntimeError as e:
        # print(e)
        return np.nan


def df2ah_iv_and_lv(df: pd.DataFrame, ps_spot, rate: float, dividend: float, min_dte=14) -> pd.DataFrame:
    for ts, sub_df in df.groupby(level='ts'):
        spot = ps_spot.loc[ts]['close']
        for right in ['call', 'put']:
            ah_vol_interp = ah_vol_interpolation(sub_df, ts.to_pydatetime(), right, spot, rate, dividend)
            ah_vol_srf = ql.AndreasenHugeVolatilityAdapter(ah_vol_interp)
            ah_lv_srf = ql.AndreasenHugeLocalVolAdapter(ah_vol_interp)

            idx = df.loc[(ts, slice(None), slice(None), right)].index
            vols = [safe_black_vol(ah_vol_srf, get_tenor(ix[0], ts.date()), float(ix[1])) for ix in idx]
            dIVdS = [-ah_dVoldK(ah_vol_srf, get_tenor(ix[0], ts.date()), float(ix[1])) for ix in idx]
            lv = ah_lv_srf.localVol(0, spot)
            lvs = [lv] * len(idx)
            df.loc[(ts, slice(None), slice(None), right), 'ah_iv'] = vols
            df.loc[(ts, slice(None), slice(None), right), 'ah_lv'] = lvs
            df.loc[(ts, slice(None), slice(None), right), 'ah_dIVdS'] = dIVdS
    return df


def df2ah_div_ds(df: pd.DataFrame, ps_spot, rate: float, dividend: float, min_dte=14) -> pd.DataFrame:
    for ts, sub_df in df.groupby(level='ts'):
        spot = ps_spot.loc[ts]['close']
        for right in ['call', 'put']:
            ah_vol_interp = ah_vol_interpolation(sub_df, ts.to_pydatetime(), right, spot, rate, dividend)
            ah_vol_srf = ql.AndreasenHugeVolatilityAdapter(ah_vol_interp)
            idx = df.loc[(ts, slice(None), slice(None), right)].index
            dIVdS = [-ah_dVoldK(ah_vol_srf, get_tenor(ix[0], ts.date()), float(ix[1])) for ix in idx]
            df.loc[(ts, slice(None), slice(None), right), 'ah_dIVdS'] = dIVdS
    return df
