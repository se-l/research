"""
Convert zipped files containing tick data quote to qc bar data.
"""
import io
import multiprocessing
import os
import zipfile
from pathlib import Path

import pandas as pd

from collections import defaultdict
from shared.constants import file_root
from connector.ib.enums import TradeType
from options.client import Client

client = Client()


def header(trade_type: TradeType, write=False):
    if trade_type == TradeType.quote:
        names = [
            'time',
            'bid_open', 'bid_high', 'bid_low', 'bid_close', 'bid_volume',
            'ask_open', 'ask_high', 'ask_low', 'ask_close', 'ask_volume'
        ]
    elif TradeType.trade == trade_type:
        names = ['time', 'open', 'high', 'low', 'close', 'volume']
    elif trade_type == TradeType.iv_trade:
        names = ['time', 'close_underlying', 'price', 'iv', 'delta']
    elif trade_type == TradeType.iv_quote and not write:
        names = ['time', 'close_underlying', 'bid_price', 'bid_iv', 'ask_price', 'ask_iv', 'delta', '_']
    elif trade_type == TradeType.iv_quote and write:
        names = ['time', 'close_underlying', 'bid_price', 'bid_iv', 'ask_price', 'ask_iv']
    else:
        raise ValueError(f'Invalid trade_type: {trade_type}')
    return names


if __name__ == '__main__':
        print('Done.')
