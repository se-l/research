from datetime import date

import numpy as np
import plotly.graph_objs as go

from dataclasses import dataclass
from itertools import chain
from typing import Dict, Tuple
from plotly.subplots import make_subplots
from options.typess.enums import OptionRight
from shared.plotting import show


@dataclass
class IVSurfaceModelEvaluation:
    rmse_iv: float
    rmse_iv_right_tenor: Dict[Tuple[date, OptionRight], np.ndarray]
    rmse_price: float
    rmse_price_right_tenor: Dict[Tuple[date, OptionRight], np.ndarray]
    rmse_pc_of_spread: float
    rmse_pc_of_spread_right_tenor: Dict[Tuple[date, OptionRight], np.ndarray]

    mae_iv: float
    mae_iv_right_tenor: Dict[Tuple[date, OptionRight], np.ndarray]
    mae_price: float
    mae_price_right_tenor: Dict[Tuple[date, OptionRight], np.ndarray]
    mae_pc_of_spread: float
    mae_pc_of_spread_right_tenor: Dict[Tuple[date, OptionRight], np.ndarray]

    def plot(self):
        rights = [OptionRight.call, OptionRight.put]
        fig = make_subplots(rows=7, cols=2, subplot_titles=['Across Surface', ''] + list(chain(*[[f'{right} {metric}' for right in rights] for metric in ['RMSE Price', 'MAE Price', 'RMSE IV', 'MAE IV', 'RMSE % of Spread', 'MAE % of Spread']])))

        fig.add_trace(go.Bar(
                x=['RMSE IV', 'MAE IV', 'RMSE Price', 'MAE Price', 'RMSE % of Spread', 'MAE % of Spread'],
                y=[self.rmse_iv, self.mae_iv, self.rmse_price, self.mae_price, self.rmse_pc_of_spread, self.mae_pc_of_spread]
            ),
            row=1, col=1
        )

        filter_dct = lambda dct, option_right: {k[0]: v for k, v in dct.items() if k[1] == option_right}

        for col, right in enumerate(rights):
            ix_col = col + 1
            for row, attr in enumerate(['rmse_price_right_tenor', 'mae_price_right_tenor', 'rmse_iv_right_tenor', 'mae_iv_right_tenor', 'rmse_pc_of_spread_right_tenor', 'mae_pc_of_spread_right_tenor']):
                ix_row = row + 2
                dct = filter_dct(getattr(self, attr), right)
                fig.add_trace(go.Scatter(x=list(dct.keys()), y=list(dct.values()), mode='markers', marker=dict(size=4), name=attr), row=ix_row, col=ix_col)
        show(fig)
