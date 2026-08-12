from .enum_utils import EnumStr
from enum import Enum


class Side(EnumStr, Enum):
    short = 'short'
    long = 'long'
    hold = 'hold'
