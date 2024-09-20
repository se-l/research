from datetime import date
from dataclasses import dataclass


@dataclass
class SymDate:
    symbol: str
    date: date

    def __post_init__(self):
        self.symbol = self.symbol.upper()

    def __eq__(self, other):
        return self.symbol == other.symbol.upper() and self.date == other.date

    def __hash__(self):
        return hash((self.symbol, self.date))
