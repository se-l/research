from dataclasses import dataclass
from common.utils.dc_base import DCBase
from connector.ib.enums import ExchangeCode


@dataclass
class TickBidAsk(DCBase):
    time: int  # Milliseconds since midnight in the timezone of the data format.
    bid_sale: int  # Deci-cents bid price of the bid quote tick.
    bid_size: int  # Number of shares in the bid quote tick.
    ask_sale: int  # Deci-cents ask price of the ask quote tick.
    ask_size: int  # Number of shares in the ask quote tick.
    exchange: ExchangeCode = ""  # Location of the sale.
    suspicious: int = 0
    quote_sale_condition: str = ""  # Notes on the sale.
