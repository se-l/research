import datetime
import pandas as pd
import yaml

from shared.modules.logger import logger
from common.modules.enums import Exchange
from common.paths import Paths
from layers.bars.base_bar import BaseBar
from layers.exchange_reader import ExchangeDataReader


class Imbalance(BaseBar):
    information = 'imbalance'

    def __init__(self, exchange: Exchange, sym, start, end, unit, unit_size):
        super().__init__(exchange, sym, start, end, unit, **{'unit_size': unit_size})

    def resample(self) -> pd.DataFrame:
        df = ExchangeDataReader.load_trades(self.exchange, self.sym, self.start, self.end)
        self.add_measurable(df)
        df['imbalance_bar'] = df['measurable'].cumsum() // self.unit_size
        df['imbalance_size'] = (df['imbalance_bar'] - df['imbalance_bar'].shift(1)).fillna(0).astype(int)
        df = df[df['imbalance_size'] != 0]
        return df[['timestamp', 'imbalance_size']].groupby('timestamp').sum()

    def add_measurable(self, df):
        if self.unit == 'tick':
            df['measurable'] = df['side']
        elif self.unit == self.sym:
            df['measurable'] = df['side'] * df['size']
        elif self.unit == 'usd' and self.exchange == Exchange.bitmex:
            df['measurable'] = df['side'] * df['grossValue']
        elif self.unit == 'usd' and self.sym[-3:].lower() == 'usd':
            df['measurable'] = df['size'] * df['price']
        elif self.unit == 'xbt':
            df['measurable'] = df['side'] * df['homeNotional']
        else:
            raise NotImplemented(self.unit)
        return df


if __name__ == '__main__':
    information = 'trade imbalance'
    with open(Paths.layer_settings, "r") as stream:
        settings = yaml.safe_load(stream)
    for exchange in settings.keys():
        for asset, params in settings[exchange].items():
            if not params.get(information):
                continue
            for unit in ['tick', 'usd', asset]:
                unit_size = params[information].get(unit, {})
                if not unit_size:
                    continue
                logger.info(f'{information} - {asset.upper()} - {unit} - {unit_size}')
                bar = Imbalance(
                    exchange=exchange,
                    sym=asset,
                    # start=datetime.datetime(2022, 2, 7),
                    # end=datetime.datetime(2022, 3, 13),
                    # start=datetime.datetime(2022, 3, 14),
                    # end=datetime.datetime(2022, 4, 19),
                    # start=datetime.datetime(2022, 4, 24),
                    # end=datetime.datetime(2022, 5, 29),
                    start=datetime.datetime(2022, 7, 13),
                    end=datetime.datetime(2022, 9, 3),
                    unit=unit,
                    unit_size=unit_size
                )
                df = bar.resample()
                if df.shape[0] / (bar.end - bar.start).days < 1000:
                    logger.warning(f'Decrease threshold for {asset.upper()} - {unit} - {unit_size}')
                logger.info(f'Resampled df of shape: {df.shape}. Points per day: {df.shape[0] / (bar.end - bar.start).days}')
                bar.to_npy(df)
    logger.info('Done')
