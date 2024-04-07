import numpy as np
import datetime
import pandas as pd

from collections import defaultdict
from collections.abc import Mapping

from options.helper import val_from_df, atm_iv
from options.typess.equity import Equity


class Portfolio(Mapping):
    holdings: defaultdict

    def __init__(self, holdings: dict = None):
        self.holdings = defaultdict(int, holdings or {})

    @property
    def underlying(self):
        return next(iter(self.holdings.keys())).underlying_symbol.upper() if len(self.holdings) > 0 else None

    def add_holding(self, security, quantity):
        self.holdings[security] += quantity

    def remove_security(self, security):
        self.holdings.pop(security) if security in self.holdings else None

    def deltaTotal(self, calcDate: datetime.date, s: float, df: pd.DataFrame, iv_col: str = 'mid_iv') -> float:
        deltaTotal = 0
        for sec, q in self.holdings.items():
            if isinstance(sec, Equity):
                deltaTotal += q
            else:
                iv = val_from_df(df, sec.expiry, sec.optionContract.strike, sec.right, iv_col)
                if np.isnan(iv) or iv == 0:
                    price = val_from_df(df, sec.expiry, sec.optionContract.strike, sec.right, 'mid_price')
                    spot = val_from_df(df, sec.expiry, sec.optionContract.strike, sec.right, 'spot')
                    iv = sec.iv(price, spot, calcDate)
                if np.isnan(iv) or iv == 0:
                    iv = atm_iv(df, sec.expiry, s)
                iv = atm_iv(df, sec.expiry, s) if iv == 0 else iv
                deltaTotal += sec.delta(iv, s, calcDate) * q * sec.multiplier
        return deltaTotal

    def print_backtest_holdings(self):
        print('Symbol,Quantity,FillPrice')
        for sec, q in self.holdings.items():
            print(f'{str(sec)},{q},0')

    def print_entry_holdings(self, df0: pd.DataFrame, calc_date: datetime.date):
        s = df0.iloc[0]['spot']
        print('Symbol,Quantity,IV')
        for sec, q in self.holdings.items():
            if isinstance(sec, Equity):
                print(f'{str(sec)},{q},{s}')
            else:
                price = val_from_df(df0, sec.expiry, sec.optionContract.strike, sec.right, 'mid_price')
                print(f'{str(sec)},{q},{sec.iv(price, s, calc_date):.4f}')

    def keys(self):
        return self.holdings.keys()

    def securities(self):
        return self.holdings.keys()

    def __iter__(self):
        return iter(self.holdings)

    def __getitem__(self, __key):
        return self.holdings[__key]

    def __len__(self):
        return len(self.holdings)

    def __repr__(self):
        return ', '.join([f'{str(o)}: {q}' for o, q in self.holdings.items()])


if __name__ == '__main__':
    pass
