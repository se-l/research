from .enum_utils import EnumStr
from enum import Enum


class EstimatorModes(EnumStr, Enum):
    classification = 'classification'
    regression = 'regression'
    classification_exit = 'classification_exit'
