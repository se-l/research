import numpy as np

from dataclasses import dataclass, field
from datetime import date
from typing import Tuple
from numpy._typing import NDArray

from options.types.enums import Resolution
from options.types.portfolio import Portfolio
from options.volatility.estimators.earnings_iv_drop_poly_regressor import EarningsIVDropPolyRegressorV3


@dataclass
class EarningsConfig:
    sym: str
    release_date: date
    plot: bool = True
    plot_last: bool = True
    max_scoped_options: int = 400
    resolution: Resolution = Resolution.minute
    seq_ret_threshold: float = 0.002
    min_tenor: float = 0.0
    max_tenor: float = 999
    moneyness_limits: Tuple[float, float] = (0.75, 1.25)
    abs_delta_limits: Tuple[float, float] = (0.05, 0.95)
    add_equity_holdings: bool = True
    v_dIVT1: NDArray = field(default_factory=lambda: np.linspace(0.0, -0.04, 2))
    earnings_iv_drop_regressor: EarningsIVDropPolyRegressorV3 = None
    ts_time_power: float = 0.5
    v_ds_ret: NDArray = field(default_factory=lambda: np.linspace(0.8, 1.2, 21))
    portfolio: Portfolio = field(default_factory=lambda: Portfolio())
    run_solver: bool = True
    solver_t_params: Tuple[float, float, float, float] = (2.46172191, 2.7678077519107513, -0.058245258591748644, 5.577267709644770)
    n_contracts: int = 20
    earnings_iv_drop_regressor_model_name_version: str = 'f_20260516-230556'
    var_neg_ratio: float = 1.5
    weight_max_t_curve = 5.0
    weight_wing_lift: float = 0.2

    def __hash__(self):
        return hash((self.sym, self.release_date, self.plot, self.plot_last, self.max_scoped_options, self.resolution, self.seq_ret_threshold, self.min_tenor, self.max_tenor, self.moneyness_limits, self.abs_delta_limits, self.add_equity_holdings, self.ts_time_power, ','.join(str(k)+str(v) for k, v in self.portfolio.items()), self.run_solver, self.solver_t_params, self.n_contracts))


def get_earnings_cfg(sym: str, release_date: date) -> EarningsConfig:
    # path_model = os.path.join(Paths.path_models, model_nm_earnings_iv_drop_regressor)
    # earnings_iv_drop_regressor = EarningsIVDropPolyRegressorV3().load_model(path_model)
    cfg = EarningsConfig(sym, release_date, plot=False, plot_last=False,
                         moneyness_limits=(0.8, 1.2), abs_delta_limits=(0.1, 0.9),
                         min_tenor=0.0, max_tenor=1, add_equity_holdings=False)
    return cfg


# if __name__ == '__main__':
#     import pandas as pd
#     cfg = get_earnings_cfg('FDX')
#     hash(cfg)
#     x = pd.DataFrame([[0, 0.1]], columns=['moneyness_fwd_ln', 'tenor'])
#     print(cfg.earnings_iv_drop_regressor.predict(x))
