import numpy as np
import pandas as pd

from datetime import date
from collections.abc import Mapping
from typing import Dict, Set, List

from google.protobuf.internal.containers import MessageMap
from options.typess.cash import Cash
from options.typess.equity import Equity
from options.typess.holding import Holding
from options.typess.option import Option
from options.typess.security import Security, SecurityDataSnap
from options.typess.scenario import Scenario
from shared.modules.logger import warning


class Portfolio(Mapping):
    holdings: Dict[Security, float]

    def __init__(self, holdings: Dict[Security, float] = None):
        self.holdings = holdings or {}

    @classmethod
    def from_message_map(cls, msg_map: MessageMap[str, object], calculation_date: date = None):
        return cls.from_holdings({Holding.from_holding_pb(v, calculation_date) for v in msg_map.values()})

    @classmethod
    def from_holdings(cls, holdings: Set[Holding]):
        return cls({h.symbol: h.quantity for h in holdings})

    @property
    def underlying(self) -> str:
        return next(iter(self.holdings.keys())).underlying_symbol.upper() if len(self.holdings) > 0 else None

    def add_holding(self, security: Security, quantity: float):
        if security not in self.holdings:
            self.holdings[security] = 0
        self.holdings[security] += quantity

    def remove_security(self, security):
        self.holdings.pop(security) if security in self.holdings else None

    def delta_total(self, calc_date: date, s: float, df: pd.DataFrame, iv_col: str = 'mid_iv') -> float:
        from options.helper import val_from_df
        _delta_total = 0
        for sec, q in self.holdings.items():
            if isinstance(sec, Equity):
                _delta_total += q
            elif isinstance(sec, Option):
                try:
                    iv = val_from_df(df, sec.expiry, sec.optionContract.strike, sec.right, iv_col)
                    if np.isnan(iv) or iv == 0:
                        price = val_from_df(df, sec.expiry, sec.optionContract.strike, sec.right, 'mid_price')
                        spot = val_from_df(df, sec.expiry, sec.optionContract.strike, sec.right, 'spot')
                        iv = sec.iv(price, spot, calc_date)
                except (IndexError, KeyError):
                    # Missing price data. Illiquid. Can fill with model IV, interpolated IV.
                    warning(f'No {iv_col} for {sec} on {calc_date}. Setting IV to 0.2. To be improved. Some model IV better in this case')
                    iv = 0.2
                _delta_total += sec.delta(iv, s, calc_date) * q * sec.multiplier
            elif isinstance(sec, Cash):
                continue
            else:
                raise ValueError(f'Invalid security type: {type(sec)}')
        return _delta_total

    def print_backtest_holdings(self):
        print('Symbol,Quantity,FillPrice')
        for sec, q in self.holdings.items():
            print(f'{str(sec)},{q},0')

    def get_holdings(self) -> List[Holding]:
        return [Holding(sec, q) for sec, q in self.holdings.items()]

    def nlv(self, market_data: Dict[Security, SecurityDataSnap], scenario: Scenario | str = Scenario.mid):
        return sum((security.nlv(market_data, quantity, scenario) for security, quantity in self.holdings.items()))
        # {(security, quantity): security.nlv(market_data, quantity, scenario) for security, quantity in self.holdings.items()}
        # sum({(security, quantity): security.nlv(market_data, quantity, scenario) for security, quantity in self.holdings.items()}.values())

    def keys(self):
        return self.holdings.keys()

    def securities(self):
        return self.holdings.keys()

    def __iter__(self):
        return iter(self.get_holdings())

    def __getitem__(self, __key):
        return self.holdings[__key] if __key in self.holdings else 0

    def __len__(self):
        return len(self.holdings)

    def __repr__(self):
        return ', '.join([f'{str(o)}: {q}' for o, q in self.holdings.items()])

    def __copy__(self):
        return Portfolio(self.holdings)

    def __bool__(self):
        return bool(self.holdings)


if __name__ == '__main__':
    pf = Portfolio({Equity('AAPL'): 100})
    for h in pf:
        print(h)
