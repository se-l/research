from .enum_utils import EnumStr
from enum import Enum


class Timing(EnumStr, Enum):
    entry = 'entry'
    exit = 'exit'
