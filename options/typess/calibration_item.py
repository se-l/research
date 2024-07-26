import pandas as pd
from dataclasses import dataclass
from datetime import date
from typing import Iterable, List

from options.helper import get_tenor
from options.typess.enums import OptionRight


@dataclass
class CalibrationItem:
    mny_fwd_ln: Iterable[float]
    calculation_date: date
    tenor_dt: date
    right: str | OptionRight
    iv: Iterable[float]
    weights: Iterable[float] = None
    vega: Iterable[float] = None

    @property
    def tenor(self):
        return get_tenor(self.tenor_dt, self.calculation_date)


def df2calibration_items(df_in: pd.DataFrame, calc_date: date, y_col_nm, weight_col_nm='vega_mid_price_iv', vega_col_nm=None) -> List[CalibrationItem]:
    """2) Refactor to include quotes, at least during plotting."""
    calibration_items = []
    for right, s_df in df_in.groupby('right'):
        for expiry, ss_df in s_df.groupby('expiry'):
            calibration_items.append(
                CalibrationItem(
                    ss_df['moneyness_fwd_ln'].values.astype(float),
                    calc_date,
                    expiry,
                    right,
                    ss_df[y_col_nm].values.astype(float),
                    ss_df[weight_col_nm].values.astype(float),
                    vega=ss_df[vega_col_nm].values.astype(float) if vega_col_nm else None,
                ))
    return calibration_items
