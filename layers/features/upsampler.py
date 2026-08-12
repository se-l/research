import pandas as pd


class Upsampler:

    def __init__(self, ps: pd.Series):
        self.ps = ps[ps.notna()]

    def upsample(self, window, aggregator='sum', method='periods') -> pd.Series:
        """either n periods of by timestamp. given event based is goal, only former"""
        if method == 'periods':
            ps = self.ps.rolling(window=window, min_periods=window).__getattribute__(aggregator)()
            ps.name += f'|aggWindow-{window}|aggAggregator-{aggregator}'
            return ps
        elif method == 'time':
            raise NotImplementedError
        else:
            raise ValueError

