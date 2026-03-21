import numpy as np
import plotly.graph_objs as go

from datetime import date
from dataclasses import dataclass
from typing import Dict, Tuple
from options.typess.enums import OptionRight


@dataclass
class IVSurfaceModelEvaluation:
    rmse_iv: float
    rmse_iv_right_tenor: Dict[Tuple[date, OptionRight], np.ndarray]
    rmse_price: float
    rmse_price_right_tenor: Dict[Tuple[date, OptionRight], np.ndarray]
    rmse_err_over_spread: float
    rmse_err_over_spread_right_tenor: Dict[Tuple[date, OptionRight], np.ndarray]

    mae_iv: float
    mae_iv_right_tenor: Dict[Tuple[date, OptionRight], np.ndarray]
    mae_price: float
    mae_price_right_tenor: Dict[Tuple[date, OptionRight], np.ndarray]
    logit_err_price: float
    logit_err_price_right_tenor: Dict[Tuple[date, OptionRight], np.ndarray]
    mae_err_over_spread: float
    mae_err_over_spread_right_tenor: Dict[Tuple[date, OptionRight], np.ndarray]
    me_iv_by_mny: Dict[Tuple[date, OptionRight, float], np.ndarray]
    me_price_by_mny: Dict[Tuple[date, OptionRight, float], np.ndarray]


    def get_trace_error_metrics_bar(self):
        bar_titles = [
            # (self.rmse_iv, 'RMSE IV'),
            (self.mae_iv, 'MAE IV'),
            # (self.rmse_price, 'RMSE Price'),
            (self.mae_price, 'MAE Price'),
            # (self.rmse_err_over_spread, 'RMSE of err over spread'),
            (self.mae_err_over_spread, 'MAE of err over spread'),
        ]
        return go.Bar(
            x=[i[1] for i in bar_titles],
            y=[i[0] for i in bar_titles],
        )