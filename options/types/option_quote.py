from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal


@dataclass
class OptionQuote:
    ts: datetime
    expiry: date
    strike: Decimal
    right: str
    bid_close: float
    ask_close: float
