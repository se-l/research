from .enum_utils import EnumStr
from enum import Enum


class OrderType(EnumStr, Enum):
    limit = 'limit'
    market = 'market'
