import typing
import pandas as pd

from shared.modules.logger import logger


class EventExtractor:
    def __init__(self, thresholds=None, method='std'):
        self.method = method
        self.thresholds = thresholds or [None, None]

    def __call__(self, ps: pd.Series, max_events: typing.Union[int, float] = None) -> pd.Index:
        ps2 = ps[~ps.isna()]
        if self.method == 'std':
            threshold = ps2.std() * self.thresholds[-1]
            ix = ps2.index[ps2.abs() >= threshold]
            logger.info(f'Sampling {len(ix)} events for {ps.name}')
            if len(ix) > max_events:
                return pd.Index([])
            else:
                return ix
        else:
            raise NotImplementedError('Unclear method how to establish a range')
