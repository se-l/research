from .enum_utils import EnumStr
from enum import Enum


class TickDirection(EnumStr, Enum):
    hold = 'hold'
    up = 'up'
    down = 'down'
