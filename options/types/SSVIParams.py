from dataclasses import dataclass
from datetime import date
from typing import Dict, Tuple


class SSVIMetric:
    theta = 'theta'
    rho = 'rho'
    psi = 'psi'

@dataclass
class SSVITenorParams:
    theta: float
    rho: float
    psi: float

    def to_tuple(self) -> Tuple[float, float, float]:
        return self.theta, self.rho, self.psi

    def __iter__(self):
        return iter(self.to_tuple())

class SSVISurfParams(Dict[date, SSVITenorParams]):
    def __setitem__(self, key: date, value: SSVITenorParams) -> None:
        if not isinstance(key, date):
            raise TypeError("Key must be of type datetime.date")
        if not isinstance(value, SSVITenorParams):
            raise TypeError("Value must be of type SSVITenorParams")
        super().__setitem__(key, value)

    def __getitem__(self, key: date) -> SSVITenorParams:
        if not isinstance(key, date):
            raise TypeError("Key must be of type datetime.date")
        return super().__getitem__(key)