import datetime
import pandas as pd
import numpy as np
import re

from common.modules.assets import Assets
from shared.modules.logger import logger
from common.modules.enums import Exchange
from connector.tsdb.client import upsert
from layers.exchange_reader import ExchangeDataReader


class LabelReturn:
    """Volatilty dependent symmetric threshold to determins the side of the bet. Volatility potentially determined by volumne
    instead of price series for better statistical properties.
    3 barrier:
        1) min long return with stop loss
        2) min short return with stop-loss
        3) ewm_span
    Volatility measrued as moving ewa with span equal ewm_span period for now.
    """

    def __init__(self, exchange: Exchange, sym, start, end, resampling_rule, ewm_span):
        self.exchange = exchange
        self.sym = sym
        self.start = start
        self.end = end
        self.resampling_rule = resampling_rule
        self.ewm_span = ewm_span

    def label(self) -> pd.DataFrame:
        df = ExchangeDataReader.load_trades(self.exchange, self.sym, self.start, self.end).set_index('timestamp')
        dfs = []
        dt_range = self.dt_range(df)
        for (start, end) in dt_range:
            print(start, end)
            ps = df.loc[df.index[(start < df.index) & (df.index < end)], 'price']
            ps = ps.resample(rule=self.resampling_rule).last().ffill()  # close
            # return
            ps = (ps - ps.shift(1)).fillna(0) / ps
            alpha = 2 / (self.int_span + 1)
            weights = np.array([(1 - alpha) ** i for i in range(self.int_span)])

            forward_return = [(arr * weights + 1).prod() for arr in np.lib.stride_tricks.sliding_window_view(ps.values, window_shape=self.int_span)]
            dfs.append(pd.DataFrame(forward_return[1:], index=ps.index[:-self.int_span], columns=['forward_return_ewm']))
        return pd.concat(dfs)

    def dt_range(self, df) -> [list]:
        dates = sorted(set([x.date() for x in df.index]))
        dt_range = [[dates[0]]]
        for i, dt in enumerate(dates):
            if i == 0:
                continue
            elif dates[i - 1] + datetime.timedelta(days=1) == dt:
                continue
            else:
                dt_range[-1].append(dates[i - 1])
                dt_range.append([dt])
        dt_range[-1].append(dt)
        assert set([len(el) for el in dt_range]) == {2}
        return [[pd.Timestamp(el).tz_localize('UTC') for el in lst] for lst in dt_range]

    @property
    def int_span(self) -> int:
        freq_resampled = int(re.search(r'(\d*)', self.resampling_rule).group(1))
        ewm_span = int(re.search(r'(\d*)', self.ewm_span).group(1))
        return ewm_span // freq_resampled

    def to_disk(self, df):
        upsert(meta={
            'measurement_name': 'label',
            'exchange': exchange,
            'asset': sym,
            'resampling_rule': bar.resampling_rule,
            'ewm_span': bar.ewm_span,
        },
            data=df
        )


if __name__ == '__main__':
    information = 'EWM Span return label'
    exchange = Exchange.bitfinex
    sym = Assets.ethusd
    for ewm_span in [2 ** span for span in range(7)]:
        logger.info(f'{information} - {sym.upper()} - ewm_span min: {ewm_span}')
        bar = LabelReturn(
            exchange=exchange,
            sym=sym,
            start=datetime.datetime(2022, 2, 7),
            # end=datetime.datetime(2022, 4, 19),
            # start=datetime.datetime(2022, 4, 24),
            # end=datetime.datetime(2022, 5, 29),
            # start=datetime.datetime(2022, 7, 13),
            end=datetime.datetime(2022, 9, 3),
            resampling_rule='1min',
            ewm_span=f'{ewm_span}min'  # letter must match resampling letter
        )
        df = bar.label()
        logger.info(f'Resampled df of shape: {df.shape}')
        # should reference the underlying volatilty curve somewhere and add as parameter
        assert len(df.index.unique()) == len(df), 'Timestamp is not unique. Group By time first before uploading to influx.'
        bar.to_disk(df)
    logger.info('Done')
