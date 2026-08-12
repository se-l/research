from dataclasses import dataclass
from common.utils.dc_base import DCBase


@dataclass
class QuoteBar(DCBase):
    time: int  # Milliseconds since midnight in the timezone of the data format.
    open: int  # Deci-cents Open Price for Quote Bid/Ask Bar.
    high: int  # Deci-cents High Price for Quote Bid/Ask Bar.
    low: int  # Deci-cents Low Price for Quote Bid/Ask Bar.
    close: int  # Deci-cents Close Price for Quote Bid/Ask Bar.
    volume: int  # Number of shares traded in this Quote Bid/Ask Bar.
