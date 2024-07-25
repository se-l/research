import datetime
import os
import pickle
import pandas as pd

from dataclasses import dataclass
from shared.paths import Paths
from options.typess.enums import Resolution
from options.typess.equity import Equity


@dataclass
class OptionFrame:
    equity: Equity
    df_options: pd.DataFrame
    df_equity: pd.DataFrame
    df_option_trades: pd.DataFrame
    start: datetime.date
    end: datetime.date
    resolution: Resolution
    seq_ret_threshold: float
    version: str
    ts_created: datetime.datetime

    def filename(self) -> str:
        return self.fn(self.equity.symbol, self.resolution, self.seq_ret_threshold, self.version)

    def store(self):
        with open(os.path.join(Paths.path_analysis_frames, self.filename()), 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def fn(equity: str, resolution: Resolution, seq_ret_threshold: float, version: str) -> str:
        return f'{equity.lower()}_option_analysis_frame_{resolution}_{seq_ret_threshold}_ret_threshold_{version}.pkl'

    @classmethod
    def load_frame(cls, equity: Equity, resolution: Resolution, seq_ret_threshold: float, v: str = '1'):
        with open(os.path.join(Paths.path_analysis_frames, cls.fn(equity.symbol.lower(), resolution, seq_ret_threshold, v)), 'rb') as f:
            option_frame = pickle.load(f)
        return option_frame
