import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Dict
from options.types.portfolio import Portfolio
from options.types.scenario import Scenario
from options.types.security import Security
from shared.modules.logger import info


@dataclass
class StressTestDsResult:
    holdings: Portfolio
    ds_dnlv: Dict[float, float]
    delta_total: float
    delta_total_across_ds: float
    weighted_dnlv: float = 0
    marginal_utility_by_holding: Dict[Security, float] = field(default_factory=dict)
    total_objective: float = 0
    marginal_weighted_objective_by_holding: Dict[Security, float] = field(default_factory=dict)
    tag: Scenario | str = ''

    def log_marginal_utility(self):
        lst = [{'Security': h.symbol, 'Quantity': np.sign(h.quantity)*1, 'Utility': self.marginal_utility_by_holding.get(h.symbol)} for h in self.holdings.get_holdings()]
        info(pd.DataFrame(lst))

    def log_marginal_objective(self):
        lst = [{'Security': h.symbol, 'Quantity': 1, 'Objective': self.marginal_weighted_objective_by_holding.get(h.symbol)} for h in self.holdings.get_holdings()]
        info(pd.DataFrame(lst))