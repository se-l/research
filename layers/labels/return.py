import datetime
import pandas as pd
import numpy as np
import re

from common.modules.assets import Assets
from shared.modules.logger import logger
from common.modules.enums import Exchange
from connector.tsdb.client import upsert, query
from layers.exchange_reader import ExchangeDataReader


def label_point(x: np.array) -> int:
    """
    0 price
    1 long up profit
    2 long down stopout
    3 short down profit
    4 short up stopout
    :param x:
    :return:
    """
    r = x[:, 0] / x[0, 0]
    iloc_long_touch_up = np.argmax(r >= x[:, 1])
    iloc_long_touch_down = np.argmax(r <= x[:, 2])
    iloc_short_touch_down = np.argmax(r <= x[:, 3])
    iloc_short_touch_up = np.argmax(r >= x[:, 4])

    if 0 < iloc_short_touch_down < min([ix for ix in (iloc_long_touch_up, iloc_short_touch_up) if ix > 0] + [iloc_short_touch_down + 1] * 2):
        return -1
    elif 0 < iloc_long_touch_up < min([ix for ix in (iloc_long_touch_down, iloc_short_touch_down) if ix > 0] + [iloc_long_touch_up + 1] * 2):
        return 1
    else:
        return 0


class LabelReturn:
    """Volatilty dependent symmetric threshold to determins the side of the bet. Volatility potentially determined by volumne
    instead of price series for better statistical properties.
    3 barrier:
        1) min long return with stop loss
        2) min short return with stop-loss
        3) expiration
    Volatility measrued as moving ewa with span equal expiration period for now.
    """

    def __init__(self, exchange: Exchange, sym, start, end, resampling_rule, expiration_window,
                 min_long_return_sig_f,
                 max_long_stopout_sig_f,
                 min_short_return_sig_f,
                 max_short_stopout_sig_f):
        self.exchange = exchange
        self.sym = sym
        self.start = start
        self.end = end
        self.resampling_rule = resampling_rule
        self.expiration_window = expiration_window
        self.min_long_return_sig_f = min_long_return_sig_f
        self.max_long_stopout_sig_f = max_long_stopout_sig_f
        self.min_short_return_sig_f = min_short_return_sig_f
        self.max_short_stopout_sig_f = max_short_stopout_sig_f

    def label(self) -> pd.DataFrame:
        df = ExchangeDataReader.load_trades(self.exchange, self.sym, self.start, self.end).set_index('timestamp')
        ps = df['price'].resample(rule=self.resampling_rule).last().ffill()
        df = pd.DataFrame(ps[~ps.isna()])
        df = self.add_volatilty(df)
        df = df[~df['volatility'].isna()]
        df['barrier_long_up'] = 1 + df['volatility'] * min_long_return_sig_f  # profit
        df['barrier_long_down'] = 1 - df['volatility'] * max_long_stopout_sig_f  # stopout
        df['barrier_short_down'] = 1 - df['volatility'] * min_short_return_sig_f  # profit
        df['barrier_short_up'] = 1 + df['volatility'] * max_short_stopout_sig_f  # stopout
        df['label'] = None
        df.loc[:(len(df) - self.expiration_periods + 1), 'label'] = [label_point(a) for a in np.lib.stride_tricks.sliding_window_view(
            df[['price', 'barrier_long_up', 'barrier_long_down', 'barrier_short_down', 'barrier_short_up']].values, window_shape=(self.expiration_periods, 5))
        .reshape(-1, self.expiration_periods, 5)]
        df = df[~df['label'].isna()]
        df = df[(df['label'] - df['label'].shift(1).fillna(0)) != 0]
        # print(df[['label', 'price']])
        return df[['label']]

    @property
    def expiration_periods(self) -> int:
        freq_resampled = int(re.search(r'(\d*)', self.resampling_rule).group(1))
        expiration_window = int(re.search(r'(\d*)', self.expiration_window).group(1))
        return expiration_window // freq_resampled

    def add_volatilty(self, df: pd.DataFrame):
        res = query(meta={
            'measurement_name': "trade bars",
            'weighting': "ewm",
            'information': "volatility"
        },
            start=self.start,
            to=self.end
        )
        df_vol = pd.DataFrame(res)
        df_vol.columns = ['ts', 'volatility']
        df_vol = df_vol.set_index('ts')
        df = df.join(df_vol, on='ts', how='left')
        df['volatility'] = df['volatility'].ffill()
        return df

    def to_disk(self, df: pd.DataFrame):
        upsert(meta={
            'measurement_name': 'label',
            'exchange': self.exchange,
            'asset': sym,
            'resampling_rule': self.resampling_rule,
            'expiration_window': self.expiration_window,
            'min_long_return_vol_f': min_long_return_sig_f,
            'max_long_stopout_sig_f': max_long_stopout_sig_f,
            'min_short_return_vol_f': min_short_return_sig_f,
            'max_short_stopout_sig_f': max_short_stopout_sig_f,
        },
            data=df
        )


if __name__ == '__main__':
    exchange = Exchange.bitfinex
    sym = Assets.ethusd
    min_long_return_sig_f = 0.2
    max_long_stopout_sig_f = min_long_return_sig_f / 2
    min_short_return_sig_f = 0.2
    max_short_stopout_sig_f = min_short_return_sig_f / 2

    bar = LabelReturn(
        exchange=exchange,
        sym=sym,
        start=datetime.datetime(2022, 2, 9),
        end=datetime.datetime(2022, 3, 13),
        resampling_rule='1min',
        expiration_window='60min',  # letter must match resampling letter
        min_long_return_sig_f=min_long_return_sig_f,
        max_long_stopout_sig_f=max_long_stopout_sig_f,
        min_short_return_sig_f=min_short_return_sig_f,
        max_short_stopout_sig_f=max_short_stopout_sig_f,
    )
    df = bar.label()
    logger.info(f'Resampled df of shape: {df.shape}')
    print(df.head())
    print(df.tail())
    # should reference the underlying volatilty curve somewhere and add as parameter
    assert len(df.index.unique()) == len(df), 'Timestamp is not unique. Group By time first before uploading to influx.'
    df['label'] = df['label'].astype(float)
    bar.to_disk(df)
    logger.info('Done')
