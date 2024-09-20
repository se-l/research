from itertools import chain

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, List, Dict, Tuple

from options.helper import get_tenor
from options.typess.enums import OptionRight


@dataclass
class CalibrationItem:
    mny_fwd_ln: np.ndarray
    strike: np.ndarray
    calculation_date: date
    tenor_dt: date
    right: str | OptionRight
    iv: np.ndarray
    price: np.ndarray
    spot: np.ndarray
    rf: float
    dividend_yield: float
    weights: np.ndarray = None
    vega: np.ndarray = None
    ts: Iterable[datetime] = None
    bid_price: np.ndarray = None
    ask_price: np.ndarray = None

    def __post_init__(self):
        if self.bid_price is None:
            self.bid_price = np.empty_like(self.price)
            self.bid_price[:] = np.nan
        if self.ask_price is None:
            self.ask_price = np.empty_like(self.price)
            self.ask_price[:] = np.nan

    @property
    def tenor(self):
        return get_tenor(self.tenor_dt, self.calculation_date)

    def __add__(self, other):
        if not isinstance(other, CalibrationItem):
            raise NotImplemented
        if self.tenor_dt != other.tenor_dt:
            raise NotImplemented
        if self.right != other.right:
            raise NotImplemented
        if self.calculation_date != other.calculation_date:
            raise NotImplemented

        mny_fwd_ln = np.concatenate([self.mny_fwd_ln, other.mny_fwd_ln])
        strike = np.concatenate([self.strike, other.strike])
        iv = np.concatenate([self.iv, other.iv])
        price = np.concatenate([self.price, other.price])
        spot = np.concatenate([self.spot, other.spot])
        weights = np.concatenate([self.weights, other.weights]) if self.weights is not None else None
        vega = np.concatenate([self.vega, other.vega]) if self.vega is not None else None
        ts = np.concatenate([self.ts, other.ts]) if self.ts is not None else None
        bid_price = np.concatenate([self.bid_price, other.bid_price])
        ask_price = np.concatenate([self.ask_price, other.ask_price])

        return CalibrationItem(
            mny_fwd_ln=mny_fwd_ln,
            strike=strike,
            calculation_date=self.calculation_date,
            tenor_dt=self.tenor_dt,
            right=self.right,
            iv=iv,
            price=price,
            spot=spot,
            rf=self.rf,
            dividend_yield=self.dividend_yield,
            weights=weights,
            vega=vega,
            ts=ts,
            bid_price=bid_price,
            ask_price=ask_price
        )


def df2calibration_items(df_in: pd.DataFrame, calc_date: date, iv_col_nm, price_col_nm, spot_col_nm, rf, dividend_yield, weight_col_nm='vega_mid_price_iv', vega_col_nm=None) -> List[CalibrationItem]:
    """2) Refactor to include quotes, at least during plotting."""
    df = df_in[df_in[price_col_nm].notna() & df_in[iv_col_nm].notna()]
    if weight_col_nm:
        df = df[(df[weight_col_nm].notna()) & (df[weight_col_nm] > 0)]
    calibration_items = []
    for right, s_df in df.groupby('right'):
        for expiry, ss_df in s_df.groupby('expiry'):
            calibration_items.append(
                CalibrationItem(
                    mny_fwd_ln=ss_df['moneyness_fwd_ln'].values.astype(float),
                    strike=np.array(ss_df.index.get_level_values('strike').astype(float)),
                    calculation_date=calc_date,
                    tenor_dt=expiry,
                    right=right,
                    iv=ss_df[iv_col_nm].values.astype(float),
                    price=ss_df[price_col_nm].values.astype(float),
                    spot=ss_df[spot_col_nm].values.astype(float),
                    rf=rf,
                    dividend_yield=dividend_yield,
                    weights=ss_df[weight_col_nm].values.astype(float),
                    vega=ss_df[vega_col_nm].values.astype(float) if vega_col_nm else None,
                    ts=pd.to_datetime(ss_df.index.get_level_values('ts')),
                    bid_price=ss_df['bid_close'].values if 'bid_close' in df_in.columns else None,
                    ask_price=ss_df['ask_close'].values if 'ask_close' in df_in.columns else None,
                ))
    return calibration_items


def union_calibration_items(*ci: List[CalibrationItem]) -> List[CalibrationItem]:
    """Join on key tenor_dt, right"""
    dct_ci: Dict[Tuple[date, OptionRight], CalibrationItem] = {}
    for c in chain(*ci):
        key = (c.tenor_dt, c.right)
        if key not in dct_ci:
            dct_ci[key] = c
        else:
            dct_ci[key] += c
    return sorted(dct_ci.values(), key=lambda x: (x.tenor_dt, x.right))
