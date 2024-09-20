import datetime

from decimal import Decimal
from dataclasses import dataclass
from typing import Dict

from options.typess.enums import OptionRight, OptionStyle, TickType, Resolution
from options.typess.equity import Equity
from options.typess.security import Security, SecurityDataSnap
from options.typess.scenario import Scenario
from shared.constants import dt_fmt_ymd


@dataclass
class OptionContract(Security):
    # Ideally refactor to combine with Option class
    def __init__(self, symbol: str, underlying_symbol: str, expiry: datetime.date, strike: Decimal, right: OptionRight | str, issue_date: datetime.date = None, option_style: str = OptionStyle.american, multiplier: int = 100, equity: Equity = None):
        self.symbol = symbol.upper()
        self._underlying_symbol = underlying_symbol.upper()
        self.expiry = expiry
        self.strike = strike
        self.right = right
        self.issue_date = issue_date
        self.option_style = option_style
        self.multiplier = multiplier
        self.equity = equity

        self.equity = Equity(self._underlying_symbol)

    @property
    def underlying_symbol(self):
        return self.equity.symbol

    @classmethod
    def from_filename(cls, filename: str, issue_date=None):
        if filename[:8].isdigit():  # Minute: 20230627_akam_minute_quote_american_put_1800000_20240119.csv"
            dt, symbol, resolution, tick_type, option_style, option_right, strike, expiry = filename.split('.')[0].split('_')
        else:  # Daily: are_quote_american_put_1950000_20230721.csv
            symbol, tick_type, option_style, option_right, strike, expiry = filename.split('.')[0].split('_')
        expiry = datetime.datetime.strptime(expiry, '%Y%m%d').date()
        option_right = OptionRight.from_string(option_right)
        option_style = OptionStyle.from_string(option_style)
        strike = Decimal(strike) / 10000
        return cls(symbol, symbol, expiry, strike, option_right, issue_date, option_style)

    @classmethod
    def from_contract_nm(cls, contract_nm: str, issue_date=None):
        symbol, exercise, option_right, strike, expiry = contract_nm.split('_')
        expiry = datetime.datetime.strptime(expiry, '%Y%m%d').date()
        option_right = OptionRight.from_string(option_right)
        strike = Decimal(strike) / 10000
        return cls(symbol, symbol, expiry, strike, option_right, issue_date)

    def csv_name(self, tick_type: TickType, resolution: Resolution, dt: datetime.date = None):
        if resolution in (Resolution.minute, Resolution.second, Resolution.tick) and dt:  # for minute, second, tick
            return f'{dt.strftime(dt_fmt_ymd)}_{self.symbol}_{resolution}_{tick_type}_{self.option_style}_{self.right}_{int(self.strike * 10000)}_{self.expiry.strftime(dt_fmt_ymd)}.csv'.lower()
        else:
            return f'{self.underlying_symbol}_{tick_type}_{self.option_style}_{self.right}_{int(self.strike * 10000)}_{self.expiry.strftime(dt_fmt_ymd)}.csv'.lower()

    def zip_name(self, tick_type: TickType, resolution: Resolution, dt: datetime.date = None):
        if dt is None:
            raise ValueError('date must be provided')
        return self.get_zip_name(self.underlying_symbol, tick_type, resolution, dt)

    def nlv(self, market_data: Dict[str, SecurityDataSnap], q: float = 1, scenario=Scenario.mid):
        if scenario == Scenario.mid:
            price = (market_data[self.symbol].bid + market_data[self.symbol].ask) / 2
        elif scenario == Scenario.best:
            price = market_data[self.symbol].bid if q > 0 else market_data[self.symbol].ask
        elif scenario == Scenario.worst:
            price = market_data[self.symbol].ask if q > 0 else market_data[self.symbol].bid
        else:
            raise ValueError(f'Invalid scenario: {scenario}')
        return price * q * self.multiplier

    @staticmethod
    def get_zip_name(underlying_symbol: str, tick_type: TickType, resolution: Resolution, date: datetime.date):
        if resolution in (Resolution.daily, Resolution.hour):
            return f'{underlying_symbol}_{date.year}_{tick_type}_american.zip'.lower()
        else:
            return f'{date.strftime(dt_fmt_ymd)}_{tick_type}_american.zip'.lower()

    def ib_symbol(self):
        # print('WARNING: strike may not be accurate')
        sym = self.underlying_symbol
        sym += ' ' * (6 - len(sym))
        return f'{sym}{self.expiry.strftime("%y%m%d")}{self.right[0]}{str(int(self.strike*1000)).zfill(8)}'.upper()

    @classmethod
    def from_ib_symbol(cls, ib_symbol: str):
        """PANW  250117C00540000"""
        underlying_symbol = ib_symbol.split(' ')[0]
        # use regex to extract expiry, right, strike
        expiry = datetime.datetime.strptime(ib_symbol.split(' ')[-1][:6], '%y%m%d').date()
        right = OptionRight.from_string(ib_symbol.split(' ')[-1][6])
        # print('WARNING: strike may not be accurate')
        strike = Decimal(ib_symbol.split(' ')[-1][7:]) / 1000
        return cls(ib_symbol, underlying_symbol, expiry, strike, right)

    def __repr__(self):
        return f'{self.underlying_symbol}_{self.option_style}_{self.right}_{int(self.strike * 10000)}_{self.expiry.strftime(dt_fmt_ymd)}'.lower()

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def __hash__(self):
        return hash(self.__repr__())


# if __name__ == '__main__':
#     OptionContract.from_filename('are_quote_american_put_1950000_20230721.csv')
