from .enum_utils import EnumStr
from enum import Enum


class Resolution(EnumStr, Enum):
    second = '1S'
    day = '1D'
