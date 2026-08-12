import numpy as np
import pandas as pd

from itertools import chain
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Dict
from options.helper import get_tenor
from options.types.dividend import Dividend


@dataclass
class CalibrationItem:
    mny_fwd_ln: np.ndarray
    strike: np.ndarray
    calculation_date: date
    tenor_dt: date
    rights: np.ndarray
    iv: np.ndarray
    price: np.ndarray
    spot: np.ndarray
    dividends: List[Dividend] = ()
    weights: np.ndarray = None
    vega: np.ndarray = None
    ts: np.ndarray[datetime] = None
    bid_price: np.ndarray = None
    ask_price: np.ndarray = None
    dividend_yield: float = None  # not needed anymore for American options.

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
        if self.calculation_date != other.calculation_date:
            raise NotImplemented

        mny_fwd_ln = np.concatenate([self.mny_fwd_ln, other.mny_fwd_ln])
        strike = np.concatenate([self.strike, other.strike])
        rights = np.concatenate([self.rights, other.rights])
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
            rights=rights,
            iv=iv,
            price=price,
            spot=spot,
            dividends=self.dividends,
            weights=weights,
            vega=vega,
            ts=ts,
            bid_price=bid_price,
            ask_price=ask_price
        )

    def downsample(self, n: int) -> 'CalibrationItem':
        ix_all = np.arange(len(self.price))
        ix_sample = np.random.choice(ix_all, size=n, replace=False)
        return CalibrationItem(
            mny_fwd_ln=self.mny_fwd_ln[ix_sample],
            strike=self.strike[ix_sample],
            calculation_date=self.calculation_date,
            tenor_dt=self.tenor_dt,
            rights=self.rights[ix_sample],
            iv=self.iv[ix_sample],
            price=self.price[ix_sample],
            spot=self.spot[ix_sample],
            dividends=self.dividends,
            weights=self.weights[ix_sample] if self.weights is not None else None,
            vega=self.vega[ix_sample] if self.vega is not None else None,
            ts=self.ts[ix_sample],
            bid_price=self.bid_price[ix_sample],
            ask_price=self.ask_price[ix_sample],
        )


def df2calibration_items(df_in: pd.DataFrame, calc_date: date, iv_col_nm, price_col_nm, spot_col_nm, dividends: List[Dividend], weight_col_nm='vega_mid_price_iv', vega_col_nm=None, max_samples_by_tenor=None, seed=1234) -> List[CalibrationItem]:
    """2) Refactor to include quotes, at least during plotting."""
    df = df_in[df_in[price_col_nm].notna() & df_in[iv_col_nm].notna()]
    if weight_col_nm:
        df = df[(df[weight_col_nm].notna()) & (df[weight_col_nm] > 0)]
    calibration_items = []
    for expiry, s_df in df.groupby('expiry'):
        df_sample = s_df.sample(n=min(max_samples_by_tenor, len(s_df)), random_state=seed) if max_samples_by_tenor else s_df
        calibration_items.append(
            CalibrationItem(
                mny_fwd_ln=df_sample['moneyness_fwd_ln'].values.astype(float),
                strike=np.array(df_sample.index.get_level_values('strike').astype(float)),
                calculation_date=calc_date,
                tenor_dt=expiry,
                rights=np.array(df_sample.index.get_level_values('right').astype(str)),
                iv=df_sample[iv_col_nm].values.astype(float),
                price=df_sample[price_col_nm].values.astype(float),
                spot=df_sample[spot_col_nm].values.astype(float),
                dividend_yield=0,
                dividends=dividends,
                weights=df_sample[weight_col_nm].values.astype(float) if weight_col_nm else None,
                vega=df_sample[vega_col_nm].values.astype(float) if vega_col_nm else None,
                ts=pd.to_datetime(df_sample.index.get_level_values('ts')),
                bid_price=df_sample['bid_close'].values if 'bid_close' in df_in.columns else None,
                ask_price=df_sample['ask_close'].values if 'ask_close' in df_in.columns else None,
            ))
    return calibration_items


def union_calibration_items(*ci: List[CalibrationItem]) -> List[CalibrationItem]:
    """Unions quotes and trades into a single calibration item. Grouped by tenor_dt"""
    dct_ci: Dict[date, CalibrationItem] = {}
    for c in chain(*ci):
        key = c.tenor_dt
        if key not in dct_ci:
            dct_ci[key] = c
        else:
            dct_ci[key] += c
    return sorted(dct_ci.values(), key=lambda x: x.tenor_dt)
