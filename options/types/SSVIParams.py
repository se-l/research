from dataclasses import dataclass
from datetime import date
from typing import Dict

class SSVIMetric:
    theta = 'theta'
    rho = 'rho'
    psi = 'psi'

@dataclass
class SSVITenorParams:
    theta: float
    rho: float
    psi: float

class SSVISurfParams(Dict[date, SSVITenorParams]):
    def __setitem__(self, key: date, value: SSVITenorParams) -> None:
        if not isinstance(key, date):
            raise TypeError("Key must be of type datetime.date")
        if not isinstance(value, SSVITenorParams):
            raise TypeError("Value must be of type Parameters")
        super().__setitem__(key, value)

    def __getitem__(self, key: date) -> SSVITenorParams:
        if not isinstance(key, date):
            raise TypeError("Key must be of type datetime.date")
        return super().__getitem__(key)