import datetime
import yaml

from shared.modules.logger import logger
from common.paths import Paths
from layers.bars.base_bar import BaseBar
from layers.exchange_reader import ExchangeDataReader


class TradeVolumeBars(BaseBar):
    information = 'volume'

    def __init__(self, exchange, sym, start, end, unit, side=None, **kwargs):
        self.side = side
        super().__init__(exchange, sym, start, end, unit, side=side, **kwargs)

    def resample(self):
        """Total volume across buy sell or single side volume when side is set"""
        df = ExchangeDataReader.load_trades(self.exchange, self.sym, self.start, self.end)
        self.add_measurable(df)
        if self.side:
            df = df[df['side'] == self.side]
            df['measurable'] = df['measurable'].abs()
        df['measurable_cumsum'] = df['measurable'].cumsum() // self.unit_size
        df['volume_size'] = (df['measurable_cumsum'] - df['measurable_cumsum'].shift(1)).fillna(0).astype(int)
        df = df[df['volume_size'] != 0]
        return df[['timestamp', 'volume_size']].groupby('timestamp').sum()

    def add_measurable(self, df):
        if self.unit == 'tick':
            df['measurable'] = df['side'].abs()  # for volume interested in abs # cnt ticks, unlike imbalance measures
        elif self.unit == self.sym:
            df['measurable'] = df['side'] * df['size']
        elif self.unit == 'usd' and self.sym[-3:].lower() == 'usd':
            df['measurable'] = df['side'] * df['size'] * df['price']
        else:
            raise NotImplemented
        return df


if __name__ == '__main__':
    information = 'volume'
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
                for side in [None, -1, 1]:
                    logger.info(f'{information} - side: {side} - {asset.upper()} - {unit} - {unit_size}')
                    # if asset != 'btcusd' and unit != 'btcusd' and side is not None:
                    #     continue
                    bar = TradeVolumeBars(
                        exchange=exchange,
                        sym=asset,
                        # start=datetime.datetime(2022, 2, 7),
                        # # end=datetime.datetime(2022, 3, 13),
                        # # start=datetime.datetime(2022, 3, 14),
                        # # end=datetime.datetime(2022, 4, 19),
                        # # start=datetime.datetime(2022, 4, 24),
                        # end=datetime.datetime(2022, 5, 29),
                        start=datetime.datetime(2022, 7, 13),
                        end=datetime.datetime(2022, 9, 3),
                        unit=unit,
                        unit_size=unit_size,
                        side=side,
                    )
                    df = bar.resample()
                    if side is None:
                        if df.shape[0] / (bar.end - bar.start).days < 500:
                            logger.warning(f'Decrease threshold for {asset.upper()} - {unit} - {unit_size}')
                        logger.info(f'Resampled df of shape: {df.shape}. Points per day: {df.shape[0] / (bar.end - bar.start).days}')
                    bar.to_npy(df)
    logger.info('Done')

