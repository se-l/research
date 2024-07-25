import os
from dataclasses import dataclass, field
from typing import Tuple, Dict, Union

import numpy as np
import pandas as pd

from options.typess.enums import Resolution
from options.typess.equity import Equity
from options.typess.option import Option
from options.volatility.estimators.earnings_iv_drop_poly_regressor import EarningsIVDropPolyRegressorV3
from shared.constants import model_nm_earnings_iv_drop_regressor
from shared.paths import Paths


@dataclass
class EarningsConfig:
    sym: str
    take: int = -1
    plot: bool = True
    plot_last: bool = True
    max_scoped_options: int = 400
    resolution: Resolution = Resolution.minute
    seq_ret_threshold: float = 0.005
    min_tenor: float = 0.0
    max_tenor: float = 999
    moneyness_limits: Tuple[float, float] = (0.8, 1.2)
    abs_delta_limits: Tuple[float, float] = (0.05, 0.95)
    add_equity_holdings: bool = True
    v_dIVT1: np.array = field(default_factory=lambda: np.linspace(0.0, -0.04, 2))
    earnings_iv_drop_regressor: EarningsIVDropPolyRegressorV3 = None
    ts_time_power: float = 0.5
    v_ds_ret: np.array = field(default_factory=lambda: np.linspace(0.8, 1.2, 21))
    holdings: Dict[Union[Option, Equity], float] = field(default_factory=dict)
    run_solver: bool = True
    solver_t_params: Tuple[float, float, float] = (3.9179707006161415, 0.32111046535136556, 18.689304507870794)
    n_contracts: int = 20


def get_earnings_cfg(sym) -> EarningsConfig:
    path_model = os.path.join(Paths.path_models, model_nm_earnings_iv_drop_regressor)
    earnings_iv_drop_regressor = EarningsIVDropPolyRegressorV3().load_model(path_model)
    cfg = EarningsConfig(sym, take=-2, plot=False, plot_last=False, earnings_iv_drop_regressor=earnings_iv_drop_regressor,
                         moneyness_limits=(0.8, 1.2), abs_delta_limits=(0.1, 0.9),
                         min_tenor=0.0, max_tenor=1, add_equity_holdings=False)
    return cfg


if __name__ == '__main__':
    cfg = get_earnings_cfg('FDX')
    x = pd.DataFrame([[0, 0.1]], columns=['moneyness_fwd_ln', 'tenor'])
    print(cfg.earnings_iv_drop_regressor.predict(x))
