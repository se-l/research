import datetime

from dataclasses import dataclass
from options.typess.enums import TickType, Resolution
from options.typess.symbol import Symbol


@dataclass
class Equity(Symbol):
    symbol: str
    multiplier: int = 1

    @classmethod
    def from_filename(cls, filename: str):
        """hpe_quote.csv"""
        symbol = filename
        return cls(symbol)

    def csv_name(self, tick_type: TickType, resolution: Resolution, date: datetime.date = None):
        if resolution in (Resolution.minute, Resolution.second, Resolution.tick) and date:  # for minute, second, tick
            return f'{date.strftime("%Y%m%d")}_{self.symbol}_{resolution}_{tick_type}.csv'
        else:
            return f'{self.symbol}.csv'

    def zip_name(self, tick_type: TickType, resolution: Resolution, date: datetime.date = None):
        if resolution in (Resolution.daily, Resolution.hour):
            return f'{self.symbol}.zip'

    @staticmethod
    def delta(*args, **kwargs):
        return 1

    @staticmethod
    def iv(*args, **kwargs):
        return 0

    @staticmethod
    def npv():
        return 1

    def __repr__(self):
        return f'{self.symbol}'

    def __hash__(self):
        return hash(self.__repr__())

    def __str__(self):
        return self.__repr__()
