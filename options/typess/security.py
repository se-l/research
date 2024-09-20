from datetime import date, datetime

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict

from options.typess.enums import TickType, Resolution
from options.typess.scenario import Scenario


@dataclass
class Security:
    symbol: str

    @property
    @abstractmethod
    def underlying_symbol(self): pass

    @abstractmethod
    def csv_name(self, tick_type: TickType, resolution: Resolution, dt: date = None): pass

    @abstractmethod
    def zip_name(self, tick_type: TickType, resolution: Resolution, dt: date = None): pass

    @abstractmethod
    def nlv(self, market_data: Dict['Security', 'SecurityDataSnap'], q: float = 1, scenario: Scenario | str = Scenario.mid): pass

    @abstractmethod
    def __repr__(self): pass

    def __hash__(self):
        return self.__repr__()

    def __lt__(self, other):
        return self.__repr__() < other.__repr__()

    def __le__(self, other):
        return self.__repr__() <= other.__repr__()

    def __ge__(self, other):
        return self.__repr__() >= other.__repr__()

    def __gt__(self, other):
        return self.__repr__() > other.__repr__()

    def __str__(self):
        return self.__repr__()


@dataclass
class SecurityDataSnap:
    security: Security
    ts: datetime
    spot: float
    bid: float = None
    ask: float = None

    def __hash__(self):
        return hash((self.security.symbol, self.ts))
