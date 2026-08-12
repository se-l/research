from .enum_utils import EnumStr
from enum import Enum


class Series(EnumStr, Enum):
    trade = 'trade'
    quote = 'quote'
    ask = 'ask'
    bid = 'bid'
