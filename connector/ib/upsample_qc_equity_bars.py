"""
Convert zipped files containing tick data quote to qc bar data.
"""
import io
import os
import zipfile
import pandas as pd

from collections import defaultdict
from pathlib import Path
from typing import List, Dict

from shared.constants import file_root, dt_fmt_ymd
from connector.ib.enums import TradeType
from connector.ib.upsample_qc_bars import header
from options.client import Client
from options.types.enums import Resolution

client = Client()


def load(sec_type, market, resolution_from, symbol, trade_type, start_date, end_date=None):
    path_from = os.path.join(file_root, sec_type, market, resolution_from, symbol)

    dt_contract_df = defaultdict(dict)

    for directory, subdirectories, files in os.walk(path_from):
        for fn in files:  # Open the ZIP archive in read mode
            fn_dt_str = fn[:8]
            if start_date and fn_dt_str < start_date:
                continue
            if end_date and fn_dt_str > end_date:
                continue
            if fn.endswith(f'_{trade_type}.zip'):
                with zipfile.ZipFile(os.path.join(directory, fn), 'r') as archive:
                    # Get a list of all the member filenames in the archive
                    for csv_nm in archive.namelist():
                        df = pd.read_csv(io.BytesIO(archive.read(csv_nm)), names=header(trade_type))
                        if not df.empty:
                            dt_contract_df[fn_dt_str] = df
    print(f'Loaded {resolution_from} {symbol}.')
    return dt_contract_df


def transform(dt_contract_df, trade_type: TradeType, resolution_to) -> List[Dict]:
    out = []
    # Transform - because it's not resampling, but relying on QC saving low-res data by day day, below code works for daily only currently..
    if resolution_to == Resolution.daily:
        for dt, df in dt_contract_df.items():
            if trade_type == TradeType.trade:
                out += [{
                    'time': f'{dt} 00:00',
                    'open': df['open'].iloc[0],
                    'high': df['high'].max(),
                    'low': df['low'].min(),
                    'close': df['close'].iloc[-1],
                    'volume': df['volume'].sum(),
                }]
            else:
                raise ValueError(f'Invalid trade_type: {trade_type}')

    elif resolution_to == Resolution.hour:
        for dt, contract_df in dt_contract_df.items():
            delta_days = pd.to_datetime(dt, format=dt_fmt_ymd) - pd.to_datetime('19700101', format=dt_fmt_ymd)
            df_resampled = contract_df.set_index(pd.to_datetime(contract_df['time'] / 1000, unit='s') + delta_days)
            df_resampled = client.resample(df_resampled, resolution='60min')
            df_resampled['time'] = df_resampled.index.strftime('%Y%m%d %H:%M')
            out += df_resampled.to_dict('records')

    print(f'Transformed {trade_type}.')
    return sorted(out, key=lambda x: x['time'])


def upsample_equity_bars(sec_type, market, resolution_from, resolution_to, symbol, start_date, end_date=None):
    if resolution_to not in [Resolution.daily, Resolution.hour]:
        raise ValueError(f'Invalid resolution_to: {resolution_to}')

    trade_type = TradeType.trade

    path_to = os.path.join(file_root, sec_type, market, resolution_to)
    dt_contract_df = load(sec_type, market, resolution_from, symbol, trade_type, start_date, end_date)

    records = transform(dt_contract_df, trade_type, resolution_to)
    if not records:
        print(f'No records found for {symbol} {resolution_to} {trade_type}.')
        return

    path_zip = os.path.join(path_to, f'{symbol}.zip')
    if os.path.exists(path_zip):
        try:
            with zipfile.ZipFile(path_zip, 'r') as archive:
                new_times = [r['time'] for r in records]
                for zip_csv_nm in set(archive.namelist()):
                    # Keep existing
                    df_archive = pd.read_csv(io.BytesIO(archive.read(zip_csv_nm)), names=header(trade_type, write=True))
                    df_archive.drop(df_archive[(df_archive['time'] >= new_times[0]) & (df_archive['time'] <= new_times[-1])].index, inplace=True)
                    # Union
                    records += df_archive.to_dict('records')
        except Exception as e:
            print(f'Error: {e} {symbol} {resolution_to} {trade_type} {path_zip}')
            raise e

        records = sorted(records, key=lambda x: x['time'])
        print(f'Unioned with existing archive {symbol} {resolution_to}.')

    Path(path_zip).parent.mkdir(parents=True, exist_ok=True)
    csv_nm = f'{symbol.lower()}.csv'
    with zipfile.ZipFile(path_zip, 'w') as archive:
        df = pd.DataFrame(records).sort_values(by=['time'])
        df = df[~df['time'].duplicated()]
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, header=False)
        archive.writestr(csv_nm, buffer.getvalue())
    print(f'Done {resolution_to} {trade_type} {symbol}')


if __name__ == '__main__':
    start_date = '20240607'
    end_date = '20240607'

    upsample_equity_bars('equity', 'usa', 'minute', Resolution.hour, 'ORCL', start_date, end_date)
    upsample_equity_bars('equity', 'usa', 'minute', Resolution.daily, 'ORCL', start_date, end_date)

    print('Done.')
