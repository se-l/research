import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, List, Callable

from options.helper import get_tenor
from options.typess.enums import OptionRight
from options.typess.option import Option


@dataclass
class CalibrationItem:
    mny_fwd_ln: Iterable[float] | np.ndarray
    strike: Iterable[float] | np.ndarray
    calculation_date: date
    tenor_dt: date
    right: str | OptionRight
    iv: Iterable[float] | np.ndarray
    price: Iterable[float] | np.ndarray
    spot: Iterable[float] | np.ndarray
    rf: float
    dividend_yield: float
    weights: Iterable[float] = None
    vega: Iterable[float] = None
    ts: Iterable[datetime] = None

    @property
    def tenor(self):
        return get_tenor(self.tenor_dt, self.calculation_date)


def df2calibration_items(df_in: pd.DataFrame, calc_date: date, iv_col_nm, price_col_nm, spot_col_nm, rf, dividend_yield, weight_col_nm='vega_mid_price_iv', vega_col_nm=None) -> List[CalibrationItem]:
    """2) Refactor to include quotes, at least during plotting."""
    df = df_in[df_in[price_col_nm].notna() & df_in[iv_col_nm].notna()]
    calibration_items = []
    for right, s_df in df.groupby('right'):
        for expiry, ss_df in s_df.groupby('expiry'):
            calibration_items.append(
                CalibrationItem(
                    mny_fwd_ln=ss_df['moneyness_fwd_ln'].values.astype(float),
                    strike=ss_df.index.get_level_values('strike').astype(float),
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
                    # bid=ss_df['bid_close'].values if 'bid_close' in df_in.columns else None,
                    # ask=ss_df['ask_close'].values if 'ask_close' in df_in.columns else None,
                ))
    return calibration_items
