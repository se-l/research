import datetime

from dataclasses import dataclass
from typing import Dict

from options.typess.enums import TickType, Resolution
from options.typess.security import Security, SecurityDataSnap
from options.typess.scenario import Scenario


@dataclass
class Cash(Security):
    symbol: str
    multiplier: int = 1

    def __post_init__(self):
        self.symbol = self.symbol.upper()

    @property
    def underlying_symbol(self) -> str:
        return self.symbol

    @classmethod
    def from_filename(cls, filename: str):
        """hpe_quote.csv"""
        symbol = filename
        return cls(symbol)

    def csv_name(self, tick_type: TickType, resolution: Resolution, date: datetime.date = None):
        raise NotImplementedError

    def zip_name(self, tick_type: TickType, resolution: Resolution, date: datetime.date = None):
        raise NotImplementedError

    @staticmethod
    def delta(*args, **kwargs):
        return 1

    @staticmethod
    def iv(*args, **kwargs):
        return 0

    @staticmethod
    def npv():
        return 1

    def nlv(self, market_data: Dict[str, SecurityDataSnap], q: float = 1, scenario=Scenario.mid):
        return self.multiplier * q

    def __repr__(self):
        return f'{self.symbol.upper()}'

    def __hash__(self):
        return hash(self.__repr__())

    def __str__(self):
        return self.__repr__()
