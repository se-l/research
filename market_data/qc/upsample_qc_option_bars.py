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
from connector.ib.enums import TradeType
from market_data.qc.upsample_qc_bars import header
from options.client import Client
from options.types.enums import SecurityType, Resolution
from shared.modules.logger import info

client = Client()

def load(sec_type, market, resolution_from, symbol, trade_type, start_date, end_date=None):
    path_from = os.path.join(Paths.path_data, sec_type, market, resolution_from, symbol)

    dt_contract_df = defaultdict(dict)
    yr_contract_dct = defaultdict(lambda: defaultdict(list))

    for directory, subdirectories, files in os.walk(path_from):
        for fn in files:  # Open the ZIP archive in read mode
            if start_date and fn[:len(start_date)] < start_date:
                continue
            if end_date and fn[:len(end_date)] > end_date:
                continue
            if fn.endswith(f'{trade_type}_american.zip'):
                with zipfile.ZipFile(os.path.join(path_from, fn), 'r') as archive:
                    # Get a list of all the member filenames in the archive
                    for csv_nm in archive.namelist():
                        df = pd.read_csv(io.BytesIO(archive.read(csv_nm)), names=header(trade_type))
                        if not df.empty:
                            dt_contract_df[fn[:8]][csv_nm[9:].replace(f'_{resolution_from}', '')] = df
    print(f'Loaded {resolution_from} {symbol}.')
    return dt_contract_df, yr_contract_dct


def transform(dt_contract_df, yr_contract_dct, trade_type: TradeType, resolution_to):
    # Transform - because it's not resampling, but relying on QC saving low-res data by day day, below code works for daily only currently..
    if resolution_to == 'daily':
        for dt, contract_df in dt_contract_df.items():
            for contract, df in contract_df.items():
                if trade_type == TradeType.quote:
                    yr_contract_dct[dt[:4]][contract].append({
                        'time': f'{dt} 00:00',
                        'bid_open': df['bid_open'].iloc[0],
                        'bid_high': df['bid_high'].max(),
                        'bid_low': df['bid_low'].min(),
                        'bid_close': df['bid_close'].iloc[-1],
                        'bid_volume': df['bid_volume'].sum(),
                        'ask_open': df['ask_open'].iloc[0],
                        'ask_high': df['ask_high'].max(),
                        'ask_low': df['ask_low'].min(),
                        'ask_close': df['ask_close'].iloc[-1],
                        'ask_volume': df['ask_volume'].sum(),
                    })
                elif trade_type == TradeType.trade:
                    yr_contract_dct[dt[:4]][contract].append({
                        'time': f'{dt} 00:00',
                        'open': df['open'].iloc[0],
                        'high': df['high'].max(),
                        'low': df['low'].min(),
                        'close': df['close'].iloc[-1],
                        'volume': df['volume'].sum(),
                    })
                elif trade_type == TradeType.iv_quote:  # ['time', 'close_underlying', 'bid_price', 'bid_iv', 'ask_price', 'ask_iv']
                    yr_contract_dct[dt[:4]][contract].append({
                        'time': f'{dt} 00:00',
                        'close_underlying': df['close_underlying'].iloc[-1],
                        'bid_price': df['bid_price'].iloc[-1],
                        'bid_iv': df['bid_iv'].iloc[-1],
                        'ask_price': df['ask_price'].iloc[-1],
                        'ask_iv': df['ask_iv'].iloc[-1],
                    })
                elif trade_type == TradeType.iv_trade:  # ['time', 'close_underlying', 'price', 'iv', 'delta']
                    yr_contract_dct[dt[:4]][contract].append({
                        'time': f'{dt} 00:00',
                        'close_underlying': df['close_underlying'].iloc[-1],
                        'price': df['price'].iloc[-1],
                        'iv': df['iv'].iloc[-1],
                        'delta': df['delta'].iloc[-1],
                    })
                else:
                    raise ValueError(f'Invalid trade_type: {trade_type}')
    elif resolution_to == 'hour':
        for dt, contract_df in dt_contract_df.items():
            delta_days = pd.to_datetime(dt, format='%Y%m%d') - pd.to_datetime('19700101', format='%Y%m%d')
            for contract, df in contract_df.items():
                df_resampled = df.set_index(pd.to_datetime(df['time'] / 1000, unit='s') + delta_days)
                df_resampled = client.resample(df_resampled, resolution='60min')
                df_resampled['time'] = df_resampled.index.strftime('%Y%m%d %H:%M')
                if 'quote' in trade_type:
                    yr_contract_dct[dt[:4]][contract] += df_resampled.to_dict('records')
                elif 'trade' in trade_type:
                    yr_contract_dct[dt[:4]][contract] += df_resampled.to_dict('records')
                else:
                    raise ValueError(f'Invalid trade_type: {trade_type}')

    print(f'Transformed {trade_type}.')
    return yr_contract_dct


def upsample_option_bars(
        sec_type: SecurityType | str,
        market,
        resolution_from: Resolution | str,
        resolution_to: Resolution | str,
        symbol,
        trade_type: TradeType,
        start_date: str,
        end_date=None
    ):
    if resolution_to not in ['daily', 'hour']:
        raise ValueError(f'Invalid resolution_to: {resolution_to}')

    path_to = os.path.join(Paths.path_data, sec_type, market, resolution_to)
    dt_contract_df, yr_contract_dct = load(sec_type, market, resolution_from, symbol, trade_type, start_date, end_date)

    yr_contract_dct = transform(dt_contract_df, yr_contract_dct, trade_type, resolution_to)

    for yr, contract_dct in yr_contract_dct.items():
        path_zip = os.path.join(path_to, f'{symbol}_{yr}_{trade_type}_american.zip')
        if not os.path.exists(path_zip):
            continue
        try:
            with zipfile.ZipFile(path_zip, 'r') as archive:
                for zip_csv_nm in archive.namelist():
                    # Keep existing
                    df_archive = pd.read_csv(io.BytesIO(archive.read(zip_csv_nm)), names=header(trade_type, write=True))

                    # Union
                    if zip_csv_nm in contract_dct:
                        dct_lst = contract_dct[zip_csv_nm]
                        # union of two lists without duplicates while records in dct_lst are newer
                        df_archive.drop(df_archive[(df_archive['time'] >= dct_lst[0]['time']) & (df_archive['time'] <= dct_lst[-1]['time'])].index, inplace=True)
                        contract_dct[zip_csv_nm] = df_archive.to_dict('records') + dct_lst
                    else:
                        contract_dct[zip_csv_nm] = df_archive.to_dict('records')
                # Append New
                for new_csv_nm in [nm for nm in contract_dct.keys() if nm not in archive.namelist()]:
                    dct_lst = contract_dct[new_csv_nm]
                    contract_dct[new_csv_nm] = dct_lst
        except Exception as e:
            info(f'Error: {e} {symbol} {resolution_to} {trade_type} {path_zip}')
            raise e

        info(f'upsample_option_bars(): Unioned with archive {yr}: {symbol} {trade_type} {path_zip}.')

    # Save dt_contract_ohlcv to resolution_to
    for yr, contract in yr_contract_dct.items():
        path_zip_new = os.path.join(path_to, f'{symbol}_{yr}_{trade_type}_american.zip')
        Path(path_zip_new).parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path_zip_new, 'w') as archive:
            for contract, dct_lst in contract.items():
                df = pd.DataFrame(dct_lst).sort_values(by=['time'])
                df = df[~df['time'].duplicated()]
                buffer = io.StringIO()
                df.to_csv(buffer, index=False, header=False)
                archive.writestr(contract, buffer.getvalue())
    info(f'Done {resolution_to} {trade_type} {symbol}')


def cleanup(end_date='20230712'):
    for a, b, fns in os.walk(r'C:\repos\trade\data\option\usa\second'):
        # if 'tick' in a:
        #     continue
        for fn in fns:
            if '_iv_quote_american' in fn or '_iv_trade_american' in fn:
                info(os.path.join(a, fn))
                os.remove(os.path.join(a, fn))
            # if fn[:8] > end_date:
            #     os.remove(os.path.join(a, fn))


def rn():
    for a, b, fns in os.walk(r'C:\repos\trade\data\option\usa\second'):
        for fn in fns:
            if fn.endswith('trade.zip'):
                os.rename(os.path.join(a, fn), os.path.join(a, fn.replace('trade.zip', 'trade_american.zip')))
            elif fn.endswith('quote.zip'):
                os.rename(os.path.join(a, fn), os.path.join(a, fn.replace('quote.zip', 'quote_american.zip')))


if __name__ == '__main__':
    start_date = '20231018'
    # start_date = '20230501'
    end_date = '20241231'

    arg_list = []
    # for symbol in ['CSCO', 'DELL', 'ORCL', 'AKAM', 'PFE', 'HPE', 'IPG', 'AOS', 'A', 'MO', 'FL', 'ALL', 'ARE', 'ZBRA', 'AES', 'APD', 'ALLE', 'LNT', 'ZTS', 'ZBH']:
    #     for res in ['hour', 'daily']:
    #         arg_list.append((SecurityType.option, 'usa', 'minute', res, symbol.lower(), 'trade', start_date, end_date))
    #         arg_list.append((SecurityType.option, 'usa', 'minute', res, symbol.lower(), 'quote', start_date, end_date))

    for symbol in ['DELL', 'HPE']:
        for res in ['daily']:
            arg_list.append((SecurityType.option, 'usa', 'second', res, symbol.lower(), TradeType.iv_quote, start_date, end_date))
            arg_list.append((SecurityType.option, 'usa', 'second', res, symbol.lower(), TradeType.iv_trade, start_date, end_date))
    with multiprocessing.Pool() as pool:
        pool.map(upsample_option_bars, arg_list)
    info('Done.')
