import datetime

from abc import abstractmethod
from dataclasses import dataclass

from options.typess.enums import TickType, Resolution


@dataclass
class Symbol:
    symbol: str

    @abstractmethod
    def csv_name(self, tick_type: TickType, resolution: Resolution, date: datetime.date = None): pass

    @abstractmethod
    def zip_name(self, tick_type: TickType, resolution: Resolution, date: datetime.date = None): pass

    @abstractmethod
    def __repr__(self): pass

    def __hash__(self):
        return self.__repr__()
