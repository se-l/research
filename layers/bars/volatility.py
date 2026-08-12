import datetime
import pandas as pd
import re

from common.modules.assets import Assets
from shared.modules.logger import logger
from common.modules.enums import Exchange
from layers.bars.base_bar import BaseBar
from layers.exchange_reader import ExchangeDataReader


class Volatility(BaseBar):
    information = 'volatility'

    def __init__(self, exchange, sym, start, end, unit, resampling_rule, span, weighting):
        super().__init__(exchange, sym, start, end, unit, resampling_rule=resampling_rule, span=span, weighting=weighting)

    def resample(self):
        df = ExchangeDataReader.load_trades(self.exchange, self.sym, self.start, self.end)
        ps = df.set_index('timestamp')['price'].resample(rule=self.resampling_rule).last().ffill()
        df = pd.DataFrame(ps[~ps.isna()])
        df['return'] = 1 + (df['price'] - df['price'].shift(1)) / df['price']

        if self.weighting == 'ewm':
            df['volatility'] = df['return'].ewm(span=self.std_periods, min_periods=0).std()
        else:
            df['volatility'] = df['return'].rolling(window=self.std_periods, min_periods=0).std()
        return df[~df['volatility'].isna()]['volatility']

    def add_measurable(self, df): pass

    @property
    def std_periods(self):
        freq_resampled = int(re.search(r'(\d*)', self.resampling_rule).group(1))
        span_window = int(re.search(r'(\d*)', self.span).group(1))
        return span_window // freq_resampled


if __name__ == '__main__':
    unit = 'return_close'

    bar = Volatility(
        exchange=Exchange.bitfinex,
        sym=Assets.ethusd,
        start=datetime.datetime(2022, 2, 7),
        # end=datetime.datetime(2022, 3, 13),
        # start=datetime.datetime(2022, 3, 14),
        end=datetime.datetime(2022, 4, 19),
        unit=unit,
        resampling_rule='1D',
        span='30D',  # Must align by letter for now D
        weighting='ewm'
    )
    df = bar.resample()
    logger.info(f'Resampled df of shape: {df.shape}')
    print(df.head())
    print(df.tail())
    bar.to_npy(df)
    logger.info('Done')

