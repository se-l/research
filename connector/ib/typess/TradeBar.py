from dataclasses import dataclass
from common.utils.dc_base import DCBase


@dataclass
class TradeBar(DCBase):
    time: int  # Milliseconds since midnight in the timezone of the data format.
    open: int  # Deci-cents Open Price for TradeBar.
    high: int  # Deci-cents High Price for TradeBar.
    low: int  # Deci-cents Low Price for TradeBar.
    close: int  # Deci-cents Close Price for TradeBar.
    volume: int  # Number of shares traded in this TradeBar.
