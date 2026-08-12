from .enum_utils import EnumStr
from enum import Enum


class AlgoObj(EnumStr, Enum):
    classification = 'classification'
    regression = 'regression'
