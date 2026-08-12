from common.modules.enum_utils import EnumStr
from enum import Enum


class Direction(EnumStr, Enum):
    short = 'short'
    long = 'long'
    buy = 1
    sell = -1
    flat = 0
