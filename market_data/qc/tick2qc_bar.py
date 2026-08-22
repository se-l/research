"""
Convert zipped files containing tick data quote to qc bar data.
"""
import io
import os
from datetime import date
from typing import List, Dict

import pandas as pd
import multiprocessing

from shared.constants import file_root, resampleFactor
from pathlib import Path
from zipfile import ZipFile

from connector.ib.enums import TradeType, Resolution
from connector.ib.typess.TickBidAsk import TickBidAsk
from connector.ib.typess.TickTrade import TickTrade
from dataclasses import fields
import numpy as np

from options.types.enums import SecurityType, TickType
from shared.modules.logger import info

market = 'usa'


def first_non_zero(ps: pd.Series):
    """too slow. use another language"""
    for v in ps:
        if v != 0 and not np.isnan(v):
            return v
    return np.nan


def min_non_zero(ps: pd.Series):
    """to0 slow. use another language"""
    return ps.min() or ps[~(ps.isna()) & (ps != 0)].min()


def convert_to_qc_quote_bar(bytes_io, resolution):
    df = pd.read_csv(bytes_io, names=[f.name for f in fields(TickBidAsk)])
    df.index = pd.DatetimeIndex(df['time'] * 10 ** 6)
    for c in ['open_bid', 'high_bid', 'low_bid', 'close_bid']:
        df[c] = df['bid_sale']
    for c in ['open_ask', 'high_ask', 'low_ask', 'close_ask']:
        df[c] = df['ask_sale']
    # convert price data to OHLC

    # It's slow. Refactor to a pure numpy solution for 10-20x speedup.
    # function first = should rather be first-non-null-value
    df = df.resample(resampleFactor[resolution], label='left').agg({
        # 'open_bid': first_non_zero, 'high_bid': 'max', 'low_bid': min_non_zero, 'close_bid': 'last',
        'open_bid': 'first', 'high_bid': 'max', 'low_bid': 'min', 'close_bid': 'last',
        'bid_size': 'last',
        # 'open_ask': first_non_zero, 'high_ask': 'max', 'low_ask': min_non_zero, 'close_ask': 'last',
        'open_ask': 'first', 'high_ask': 'max', 'low_ask': 'min', 'close_ask': 'last',
        'ask_size': 'last',
    })

    df = df[~df['close_bid'].isna()].astype(int)  # due to resampling, there may be rows with all NaN values
    # remove any rows with a 0 value
    df = df[(df[['open_bid', 'high_bid', 'low_bid', 'close_bid'] + ['open_ask', 'high_ask', 'low_ask', 'close_ask']] != 0).all(1)]  # due to performance issue

    # may have zero ticks at beginning of stream that have been converted to zero by integer. setting to close.

    # assert (df[['open_bid', 'high_bid', 'low_bid', 'close_bid'] + ['open_ask', 'high_ask', 'low_ask', 'close_ask']] == 0).sum().sum() == 0, 'Bad zero price values'
    # assert df.isna().sum() == 0, 'Bad NaN values'

    df.insert(0, 'time', [int(dt.timestamp() * 1000) for dt in df.index])
    buffer = io.StringIO()
    df.to_csv(buffer, header=False, index=False)
    return buffer.getvalue()


def convert_to_qc_trade_bar(bytes_io, resolution):
    df = pd.read_csv(bytes_io, names=[f.name for f in fields(TickTrade)])
    df.index = pd.DatetimeIndex(df['time'] * 10 ** 6)
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df['trade_sale']
    # convert price data to OHLC
    df = df.resample(resampleFactor[resolution], label='left').agg({'open': first_non_zero, 'high': 'max', 'low': min_non_zero, 'close': 'last', 'volume': 'sum'})
    df = df[~df['close'].isna()].astype(int)

    df = df[(df[['open', 'high', 'low', 'close']] != 0).all(1)]  # due to performance issue
    # assert (df[['open', 'high', 'low', 'close']] == 0).sum().sum() == 0, 'Bad zero price values'

    df.insert(0, 'time', [int(dt.timestamp() * 1000) for dt in df.index])
    buffer = io.StringIO()
    df.to_csv(buffer, header=False, index=False)
    return buffer.getvalue()


def csv_name(tick_csv_name, resolution, tick_type, sec_type):
    if sec_type == 'option':
        return tick_csv_name.replace("tick", resolution)
    elif sec_type == 'equity':
        # '20230627_hpe_Quote_Tick.csv'
        # '20230627_hpe_Trade_Tick.csv'
        # '20000101_ibm_second_trade.csv'
        dt, sym, qt, res = tick_csv_name.split('_')
        return f"{dt}_{sym}_{resolution}_{tick_type}.csv"


def underlying_namelists(sec_type, market, symbol) -> Dict[str, List[str]]:
    """return a list of all the namelists for the underlying symbol"""
    dct = {}
    tick_path = os.path.join(file_root, sec_type, market, Resolution.tick, symbol)
    if sec_type == 'option':
        for directory, subdirectories, files in os.walk(tick_path):
            for fn in files:
                if fn.endswith(".zip"):
                    zip_path = os.path.join(directory, fn)
                    with ZipFile(zip_path, 'r') as zipObj:
                        for member in zipObj.namelist():
                            if member.endswith('.csv'):
                                if fn not in dct:
                                    dct[fn] = []
                                dct[fn].append(member)
    else:
        raise NotImplementedError(sec_type)


def upsample_ticks_with_args(
        sec_type: SecurityType | str,
        market: str,
        resolution: Resolution | str,
        symbol: str,
        tick_type: TickType | str,
        start_date: date = None,
        end_date: date = None,
        overwrite: bool = True,
        skip_zip: bool = False
    ):
    tick_path = os.path.join(file_root, sec_type, market, Resolution.tick, symbol)
    for directory, subdirectories, files in os.walk(tick_path):
        for fn in files:
            if start_date and fn[:len(start_date)] < start_date:
                continue
            if end_date and fn[:len(end_date)] > end_date:
                continue
            if fn.endswith(".zip") and tick_type in fn:
                zip_member_value = []

                path_zip_tick = os.path.join(directory, fn)
                path_zip_upsampled = path_zip_tick.replace("tick", resolution)
                if skip_zip and os.path.exists(path_zip_upsampled):
                    print(f'\rSkipping {path_zip_upsampled}. Already exists.', end='', flush=True)
                    continue
                try:
                    with ZipFile(path_zip_tick, 'r') as zipObj:
                        for member in zipObj.namelist():
                            # print(member)
                            path_csv = os.path.join(directory, member)
                            path_csv_new = Path(csv_name(path_csv, resolution, tick_type, sec_type))
                            if start_date and path_csv_new.name[:len(start_date)] < start_date:
                                continue
                            if end_date and path_csv_new.name[:len(end_date)] > end_date:
                                continue

                            if not overwrite and os.path.exists(path_zip_upsampled):
                                with ZipFile(path_zip_upsampled, 'r') as zipObjUpsampled:
                                    if path_csv_new.name in zipObjUpsampled.namelist():
                                        print(f'\rSkipping {path_csv_new.name} in {path_zip_upsampled}. Already exists.', end='', flush=True)
                                        continue

                            if tick_type == TradeType.quote:
                                get_bars = convert_to_qc_quote_bar
                            elif tick_type == TradeType.trade:
                                get_bars = convert_to_qc_trade_bar
                            else:
                                raise ValueError(f"tick_type {tick_type} not supported.")
                            buffer_value = get_bars(
                                io.BytesIO(zipObj.read(member)),
                                resolution
                            )
                            zip_member_value.append((path_csv_new.name, buffer_value))

                    if zip_member_value:
                        Path(path_zip_upsampled).parent.mkdir(parents=True, exist_ok=True)
                        mode = 'w' if overwrite else 'a'
                        with ZipFile(path_zip_upsampled, mode) as zipObjUpsampled:
                            for member_name, value in zip_member_value:
                                zipObjUpsampled.writestr(member_name, value)
                        info(f"\nSaved {path_zip_upsampled}.")
                except Exception as e:
                    raise type(e)(f'{e} FileName: {fn} Path: {path_zip_tick}')


# def upsample_ticks_with_args(args):
#     security_type, tick_type, resolution, symbol, start_date, end_date, overwrite = args
#     run(security_type, market, resolution, symbol.lower(), tick_type, start_date=start_date, end_date=end_date, overwrite=overwrite)



def move_timestamp_by_x_seconds(sec_type, market, resolution, symbol, tick_type, start_date=None, end_date=None):
    """changed pd.resample label from 'right' to 'left'"""
    seconds = {'second': 1, 'minute': 60, 'hour': 3600, 'daily': 86400}[resolution] * -1
    resolution_path = os.path.join(file_root, sec_type, market, resolution, symbol)
    for directory, subdirectories, files in os.walk(resolution_path):
        for fn in files:
            if start_date and fn[:len(start_date)] < start_date:
                continue
            if end_date and fn[:len(end_date)] > end_date:
                continue
            if fn.endswith(".zip") and tick_type in fn:
                zip_member_value = []

                path_zip_old = os.path.join(directory, fn)
                path_zip_new = path_zip_old.replace("tick", resolution)
                with ZipFile(path_zip_old, 'r') as zipObj:
                    for member in zipObj.namelist():
                        # print(member)
                        path_csv = os.path.join(directory, member)
                        path_csv_new = Path(csv_name(path_csv, resolution, tick_type, sec_type))
                        if start_date and path_csv_new.name[:len(start_date)] < start_date:
                            continue
                        if end_date and path_csv_new.name[:len(end_date)] > end_date:
                            continue

                        bytes_io = io.BytesIO(zipObj.read(member))
                        try:
                            df = pd.read_csv(bytes_io, header=None)
                        except pd.errors.EmptyDataError:
                            # print(f"Empty data error for {path_csv_new.name}.")
                            zip_member_value.append((path_csv_new.name, ""))
                            continue
                        df.iloc[:, 0] = df.iloc[:, 0] + seconds * 1000
                        buffer = io.StringIO()
                        df.to_csv(buffer, header=False, index=False)

                        zip_member_value.append((path_csv_new.name, buffer.getvalue()))

                if zip_member_value:
                    Path(path_zip_new).parent.mkdir(parents=True, exist_ok=True)
                    with ZipFile(path_zip_new, 'w') as zipObjUpsampled:
                        for member_name, value in zip_member_value:
                            zipObjUpsampled.writestr(member_name, value)
                    print(f"Saved {path_zip_new}.")


if __name__ == '__main__':
    start_date = '20231013'
    end_date = '20241231'

    arg_list = []
    dt = start_date
    # for date in pd.date_range(start_date, end_date):
    #     dt = date.date().strftime('%Y%m%d')
    for security_type in ['option', 'equity']:
        for tick_type in ['quote', 'trade']:
            for resolution in ['second', 'minute']:
                # for symbol in ['ORCL', 'DELL', 'CSCO', 'PFE', 'HPE', 'IPG', 'AKAM', 'AOS', 'A', 'MO', 'FL', 'ALL', 'ARE', 'ZBRA', 'AES', 'APD', 'ALLE', 'LNT', 'ZTS', 'ZBH']:
                for symbol in ['ADI']:
                    arg_list.append((security_type, tick_type, resolution, symbol.lower(), dt, end_date, True))
                    # run(security_type, market, resolution, symbol.lower(), tick_type, start_date=start_date, end_date=end_date)
    # Create a pool of worker processes and map the function onto the list of arguments
    # for args in arg_list:
    #     run_with_args(args)
    with multiprocessing.Pool() as pool:
        pool.starmap(upsample_ticks_with_args, arg_list)

    print('Done.')
