from dataclasses import dataclass

from common.utils.dc_base import DCBase
from connector.ib.enums import ExchangeCode


@dataclass
class TickTrade(DCBase):
    time: int  # Milliseconds since midnight in the timezone of the data format.
    trade_sale: int  # Deci-cents price of the tick sale.
    volume: int  # Number of shares in the sale.
    exchange: ExchangeCode = ""  # Location of the sale.
    trade_sale_condition: int = 1  # Notes on the sale.
    suspicious: int = 0  # Boolean indicating the tick is flagged as suspicious. This generally indicates
                         # the trade is far from other market prices and may be reversed. TradeBar data excludes suspicious ticks.
