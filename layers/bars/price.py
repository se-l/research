import datetime
import yaml
import numpy as np

from shared.modules.logger import logger
from common.paths import Paths
from layers.bars.base_bar import BaseBar
from layers.exchange_reader import ExchangeDataReader


class PriceBars(BaseBar):
    information = 'price'

    def __init__(self, exchange, sym, start, end, unit, unit_size):
        self.unit_size = unit_size
        super().__init__(exchange, sym, start, end, unit, unit_size=unit_size)

    def resample(self):
        """Total volume across buy sell or single side volume when side is set"""
        df = ExchangeDataReader.load_trades(self.exchange, self.sym, self.start, self.end)
        if self.unit_size:
            df.loc[:, 'price'] = np.around(df['price'].values / self.unit_size, decimals=0) * self.unit_size
        df['price_changed'] = (df['price'] - df['price'].shift(1)).fillna(0)
        df = df[df['price_changed'] != 0]
        return df[['timestamp', 'price']].groupby('timestamp').last()

    def add_measurable(self, df): pass


if __name__ == '__main__':
    with open(Paths.layer_settings, "r") as stream:
        settings = yaml.safe_load(stream)
    for exchange in settings.keys():
        for asset, params in settings[exchange].items():
            try:
                logger.info(f'Loading price for {exchange} - {asset}')
                params = params.get('price', {})
                bar = PriceBars(
                    exchange=exchange,
                    sym=asset,
                    start=datetime.datetime(2022, 2, 7),
                    # end=datetime.datetime(2022, 13, 13),
                    # start=datetime.datetime(2022, 3, 14),
                    # end=datetime.datetime(2022, 4, 19),
                    # start=datetime.datetime(2022, 4, 24),
                    # end=datetime.datetime(2022, 5, 29),
                    # start=datetime.datetime(2022, 7, 13),
                    end=datetime.datetime(2022, 9, 3),
                    unit=asset,
                    unit_size=params.get('unit_size'),
                )
                df = bar.resample()
                logger.info(f'Resampled df of shape: {df.shape}')
                # print(df.head())
                # print(df.tail())
                bar.to_npy(df)
                logger.info('Done')
            except Exception as e:
                logger.warning(f'{exchange} - {asset} - Price')
                logger.warning(f'{e}')
    logger.info('Done')

