from dataclasses import dataclass, field
from typing import Dict

from options.typess.portfolio import Portfolio
from options.typess.scenario import Scenario
from options.typess.security import Security


@dataclass
class StressTestDsResult:
    holdings: Portfolio
    ds_dnlv: Dict[float, float]
    delta_total: float
    delta_total_across_ds: float
    weighted_dnlv: float = 0
    marginal_weighted_dnlv_by_holding: Dict[Security, float] = field(default_factory=dict)
    total_objective: float = 0
    marginal_scaled_objective_by_holding: Dict[Security, float] = field(default_factory=dict)
    tag: Scenario | str = ''
