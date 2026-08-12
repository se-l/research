from .enum_utils import EnumStr
from enum import Enum


class EstimatorType(EnumStr, Enum):
    lgb = 'lgb'
    xgb = 'xgb'
