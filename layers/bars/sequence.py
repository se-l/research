import yaml
import numpy as np
import datetime
import pandas as pd

from shared.modules.logger import logger
from common.paths import Paths
from common.modules.enums import Exchange
from layers.bars.base_bar import BaseBar
from layers.exchange_reader import ExchangeDataReader


class SequenceBar(BaseBar):
    information = 'sequence'

    def __init__(self, exchange, sym, start, end, unit, unit_size):
        super().__init__(exchange, sym, start, end, unit, **dict(unit_size=unit_size))

    def resample(self) -> pd.DataFrame:
        df = ExchangeDataReader.load_trades(self.exchange, self.sym, self.start, self.end)

        df_up = df[df['side'] == 1].copy()
        self.add_measurable(df_up)
        df_up['measurable_cum_up'] = df_up['measurable'].cumsum()

        df_down = df[df['side'] == -1].copy()
        self.add_measurable(df_down)
        df_down['measurable_cum_down'] = df_down['measurable'].cumsum()

        df = pd.concat((df_up, df_down), sort=False).sort_values('timestamp')
        df['measurable_cum_down'] = df['measurable_cum_down'].ffill().fillna(0).abs()
        df['measurable_cum_up'] = df['measurable_cum_up'].ffill().fillna(0)
        bars = []
        while True:
            # refactor to Julia and stop repeatedly mutating the entire array
            iloc_down = np.argmax(df['measurable_cum_down'].values >= self.unit_size)
            iloc_up = np.argmax(df['measurable_cum_up'].values >= self.unit_size)
            if iloc_up == iloc_down == 0:
                break
            iloc_up = 10 ** 99 if iloc_up == 0 else iloc_up
            iloc_down = 10 ** 99 if iloc_down == 0 else iloc_down

            if iloc_down == iloc_up:
                direction = 0
            elif iloc_up < iloc_down:
                direction = 1
            elif iloc_down < iloc_up:
                direction = -1
            else:
                raise KeyError

            iloc = min((iloc_up, iloc_down))
            df['measurable_cum_down'] -= df.iloc[iloc]['measurable_cum_down']
            df['measurable_cum_up'] -= df.iloc[iloc]['measurable_cum_up']

            bars.append({'sequence_direction': direction, 'timestamp': df.iloc[iloc]['timestamp']})
            df = df.iloc[iloc:]

        return pd.DataFrame(bars).set_index('timestamp')[['sequence_direction']]

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
            raise NotImplemented
        return df


if __name__ == '__main__':
    information = 'sequence'
    with open(Paths.layer_settings, "r") as stream:
        settings = yaml.safe_load(stream)
    for exchange in settings.keys():
        for asset, params in settings[exchange].items():
            # if not params.get('trade sequence'):
            #     continue
            if asset != 'xrpusd':
                continue
            for unit in ['tick', 'usd', asset]:
                unit_size = params['trade sequence'].get(unit, {})
                if not unit_size:
                    continue
                logger.info(f'{information} - {asset.upper()} - {unit} - {unit_size}')
                bar = SequenceBar(
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
                    unit_size=unit_size,
                )
                df = bar.resample()
                df = df['sequence_direction'].groupby(level=0).sum()
                bar.to_npy(df[df != 0])
                if df.shape[0] / (bar.end - bar.start).days < 500:
                    logger.warning(f'Decrease threshold for {asset.upper()} - {unit} - {unit_size}')
                logger.info(f'Resampled df of shape: {df.shape}. Points per day: {df.shape[0] / (bar.end - bar.start).days}')
    logger.info('Done')
    # redo ADAUSD - adausd - 50
    # Injected (124353, 6) dataframe records
    # 2022-03-20 19:33:11,958 - INFO Resampled df of shape: (124374,). Points per day: 3658.0588235294117

    # equence - XRPUSD - xrpusd - 100
    # Injected (376693, 6) dataframe records
    # 2022-03-20 20:17:03,331 - INFO Resampled df of shape: (376741,). Points per day: 11080.617647058823
