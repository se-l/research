import datetime
import pandas as pd

from common.modules.assets import Assets
from shared.modules.logger import logger
from common.modules.enums import Exchange

from layers.bars.base_bar import BaseBar
from layers.exchange_reader import ExchangeDataReader


class Feature(BaseBar):
    information = 'imbalance_per_plus_tick'

    def __init__(self, exchange: Exchange, sym, start, end, unit, **kwargs):
        super().__init__(exchange, sym, start, end, unit, **kwargs)

    def resample(self) -> pd.DataFrame:
        df = ExchangeDataReader.load_trades(self.exchange, self.sym, self.start, self.end)
        self.add_measurable(df)
        # df['imbalance_size'] = df['measurable'].cumsum()
        df['next_plus_minus_tick'] = (df['price'] - df['price'].shift(-1).ffill()) != 0
        df['key'] = None
        df.loc[df.index[df['next_plus_minus_tick'] == True], 'key'] = df.loc[df.index[df['next_plus_minus_tick'] == True], 'price'].astype(str) + 'abc' + \
                                                                      df.loc[df.index[df['next_plus_minus_tick'] == True]]['timestamp'].astype(str)
        df['key'] = df['key'].bfill()
        df = df[df['key'].notna()]
        df = df[['key', 'measurable']].groupby('key').sum()
        df = df.reset_index()
        df['timestamp'] = pd.to_datetime(df['key'].str.split('abc', expand=True)[1])
        return df[['timestamp', 'measurable']].groupby('timestamp').mean().rename(columns={'measurable': 'imbalance_size'}).astype(int)

    def add_measurable(self, df):
        if self.unit == 'tick':
            df['measurable'] = df['side']
        elif self.unit == self.sym:
            df['measurable'] = df['side'] * df['size']
        elif self.unit == 'usd':
            df['measurable'] = df['side'] * df['grossValue']
        elif self.unit == 'xbt':
            df['measurable'] = df['side'] * df['homeNotional']
        else:
            raise NotImplemented
        return df


if __name__ == '__main__':
    """tick: 500, usd: 100M, ethusd: 500k"""
    unit = 'ethusd'

    bar = Feature(
        exchange=Exchange.bitmex,
        sym=Assets.ethusd,
        start=datetime.datetime(2022, 2, 7),
        end=datetime.datetime(2022, 3, 2),
        unit=unit
    )
    df = bar.resample()
    logger.info(f'Resampled df of shape: {df.shape}')
    print(df.head())
    print(df.tail())
    bar.to_influx(df)
    logger.info('Done')
