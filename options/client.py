import numpy as np
import pandas as pd
import datetime
import os

from collections import defaultdict, Counter
from decimal import Decimal
from functools import reduce
from itertools import chain
from typing import List, Dict, Union, Set, Iterable
from zipfile import ZipFile

from shared.constants import file_root
from options.typess.enums import TickType, CsvHeader, Resolution, SecurityType
from options.typess.equity import Equity
from options.typess.option_contract import OptionContract

bp = 10_000


class Client:
    """
    Client for reading data saved in QuantConnect format
    """
    root = file_root
    market = 'usa'

    def __init__(self, root: str = None, market: str = None):
        self.root = root or self.root
        self.market = market or self.market

    def option_contracts(self, symbol: str, as_of: datetime.date = None, include_expired=False, start: datetime.date = None) -> Set[OptionContract]:
        """
        Use the daily option zip files to get a list of option contracts
        """
        path_zips = []
        path_zips.append(os.path.join(self.root, 'option', 'usa', 'daily', f'{symbol.lower()}_{as_of.year}_quote_american.zip'))
        if start:
            path_zips.append(os.path.join(self.root, 'option', 'usa', 'daily', f'{symbol.lower()}_{start.year}_quote_american.zip'))
            path_zips = list(sorted(path_zips))

        contracts = set()
        for path_zip in path_zips:
            if not os.path.exists(path_zip):
                raise FileNotFoundError(f'Missing daily file at {path_zip}')

            with ZipFile(path_zip, 'r') as zipObj:
                for csv_nm in zipObj.namelist():
                    df: pd.DataFrame = pd.read_csv(zipObj.open(csv_nm), names=getattr(CsvHeader, TickType.quote))
                    df = df[df['bid_close'] != 0]
                    if not df.empty:
                        contracts.add(OptionContract.from_filename(csv_nm, issue_date=datetime.datetime.strptime(df.iloc[0, 0], '%Y%m%d %M:%S').date()))  # '20230505 00:00'
        if include_expired:
            return contracts
        else:
            return {c for c in contracts if c.expiry >= as_of or c.issue_date <= as_of}

    def history(self, symbols: Iterable[Union[Equity, OptionContract]], start: datetime.date, end: datetime.date, resolution: Resolution.minute, tick_type: TickType.quote, security_type) -> Dict[str, pd.DataFrame]:
        output = {}
        for symbol in symbols:
            if isinstance(symbol, Equity) and tick_type == TickType.quote and resolution in (Resolution.daily, Resolution.hour):
                raise ValueError('There is no quote equity data for daily or hourly resolution.')
            start_ = max(start, symbol.issue_date) if isinstance(symbol, OptionContract) and symbol.issue_date else start
            end_ = min(end, symbol.expiry) if isinstance(symbol, OptionContract) and symbol.expiry else end
            if resolution in (Resolution.minute, Resolution.second):
                for dt in pd.date_range(start_, end_, freq='D'):
                    if dt.weekday() in (5, 6):
                        continue
                    date = dt.strftime('%Y%m%d')
                    underlying_folder = symbol.underlying_symbol.lower() if isinstance(symbol, OptionContract) else str(symbol).lower()
                    for directory, subdirectories, files in os.walk(os.path.join(self.root, security_type, self.market, resolution, underlying_folder)):
                        for file in files:
                            if file.endswith('.zip') and file[:8] == date and tick_type in file:
                                csv_name = symbol.csv_name(tick_type, resolution, dt)
                                # print('Opening', os.path.join(directory, file), csv_name)
                                # expected_contracts = {fn for fn in contracts if fn.split('.')[0][-8:] > date}
                                with ZipFile(os.path.join(directory, file), 'r') as zipObj:
                                    if csv_name not in zipObj.namelist():
                                        print(f'Missing {csv_name} in {file}')
                                        continue
                                    try:
                                        df: pd.DataFrame = pd.read_csv(zipObj.open(csv_name), names=getattr(CsvHeader, tick_type), index_col=False)
                                    except Exception as e:
                                        print(f'Error reading file: fn: {os.path.join(directory, file)}, csv: {csv_name}. {e}')
                                        raise e
                                    if df.empty:
                                        continue
                                    if tick_type in (TickType.iv_quote, TickType.iv_trade):
                                        df = df[~df['mid_price_underlying'].isna()]
                                    df['time'] = df['time'] / 1000
                                    # dt_tz = datetime.fromtimestamp(gp_ticks[0].time, tz=pytz.UTC).astimezone(pytz.timezone(TZ_USEASTERN))
                                    df['time'] += (dt.date() - datetime.date(1970, 1, 1)).total_seconds()
                                    ix_out_of_bound_ts = df.index[df['time'] > 3605962001]
                                    if ix_out_of_bound_ts.any():
                                        print(f'Out of bound timestamps in {file}: {len(ix_out_of_bound_ts)}')
                                        df.drop(ix_out_of_bound_ts, inplace=True)
                                    df['time'] = pd.to_datetime(df['time'], unit='s')
                                    df.set_index('time', inplace=True)

                                    for c in [c for c in df.columns if any((n in c for n in ('open', 'high', 'low', 'close')))]:
                                        df[c] = df[c] / bp

                                    if str(symbol) in output:
                                        output[str(symbol)] = pd.concat([output[str(symbol)], df])
                                    else:
                                        output[str(symbol)] = df
            else:
                # ignoring year for now
                directory = os.path.join(self.root, security_type, self.market, resolution)
                fns = set()
                for dt in [start, end]:
                    fns.add(symbol.zip_name(tick_type, resolution, dt))
                for fn in list(sorted(fns)):
                    if not os.path.exists(os.path.join(directory, fn)):
                        continue

                    csv_name = symbol.csv_name(tick_type, resolution, start)
                    # print('Opening', os.path.join(directory, file), csv_name)
                    with ZipFile(os.path.join(directory, fn), 'r') as zipObj:
                        if csv_name not in zipObj.namelist():
                            print(f'Missing {csv_name} in {fn}')
                            continue
                        df: pd.DataFrame = pd.read_csv(zipObj.open(csv_name), names=getattr(CsvHeader, tick_type), index_col=False)
                        if df.empty:
                            continue
                        if tick_type in (TickType.iv_trade, TickType.iv_quote):
                            df = df[~df['mid_price_underlying'].isna()]
                        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d %H:%M')
                        df.set_index('time', inplace=True)
                        df = df.loc[start:end+datetime.timedelta(days=1)]
                        for c in [c for c in df.columns if any((n in c for n in ('open', 'high', 'low', 'close')))]:
                            df[c] = df[c] / bp

                        if str(symbol) in output:
                            output[str(symbol)] = pd.concat([output[str(symbol)], df])
                        else:
                            output[str(symbol)] = df
        return output

    @staticmethod
    def union_vertically(dfs: List[pd.DataFrame], fill=False) -> pd.DataFrame:
        """cuts of time going beyond min max index intersection"""
        min_ts = max([df.index[0] for df in dfs])
        max_ts = min([df.index[-1] for df in dfs])
        df = reduce(lambda x, y: pd.merge(x, y.loc[min_ts: max_ts], how='outer', on=['time']), dfs)
        if fill:
            df.fillna(method='ffill', inplace=True)
        return df

    @staticmethod
    def resample(df: pd.DataFrame, resolution: str = '60min') -> pd.DataFrame:
        def method(col):
            if any(c in col for c in ('volume', 'size')):
                return 'sum'
            elif 'open' in col:
                return 'first'
            elif 'high' in col:
                return 'max'
            elif 'low' in col:
                return 'min'
            elif 'close' in col:
                return 'last'
            else:
                return 'last'
        df = df.resample(resolution, label='left').agg({c: method(c) for c in df.columns})
        close_col = next(iter([c for c in df.columns if 'close' in c]), None)
        return df[~df[close_col].isna()] if close_col else df

    def central_volatility_contracts(self, symbol: Equity, start: datetime.date, end: datetime.date = None, n=1) -> Dict[datetime.date, Set[OptionContract]]:
        """
        Return the contracts, necessary to derive the point in time ATM implied volatility. Returns more contracts because over a given time frame, the price varies
        hence more contracts are needed to be able to always have the ATM contract.
        """
        out_contracts = defaultdict(set)
        contracts = self.option_contracts(str(symbol), end, include_expired=True, start=start)
        df = self.history([symbol], start, end, Resolution.minute, TickType.quote, SecurityType.equity)[str(symbol)]
        ps = (df['bid_close'] + df['ask_close']) / 2
        for expiry in {c.expiry for c in contracts}:
            mat_contracts = {c for c in contracts if c.expiry == expiry}
            strikes = np.array(list({c.strike for c in mat_contracts}))

            for c in mat_contracts:
                ps_c = ps[(ps.index.date >= c.issue_date) & (ps.index.date <= c.expiry)]
                if ps_c.empty:
                    continue
                ps_high = Decimal(ps_c.max()).quantize(Decimal('0.0001'))
                ps_low = Decimal(ps_c.min()).quantize(Decimal('0.0001'))

                # get the contracts whose strike prices are next above and below the underlying price
                try:
                    max_n_strike = max(sorted([s for s in (strikes - ps_high) if s > 0])[:n]) + ps_high
                    min_n_strike = min(sorted([s for s in (strikes - ps_low) if s < 0], reverse=True)[:n]) + ps_low
                except ValueError as e:
                    print(f'Error: {e}')
                    continue
                if max_n_strike >= c.strike >= min_n_strike:
                    out_contracts[c.expiry].add(c)

        return out_contracts

    def get_contracts(self, symbol: Equity, as_of: datetime.date = None, strike_range=(0, 999999)) -> Dict[datetime.date, List[OptionContract]]:
        as_of = as_of or datetime.date.today()
        out_contracts: Dict[datetime.date, List[OptionContract]] = defaultdict(list)
        contracts = self.option_contracts(str(symbol), as_of)
        for mat_date in {c.expiry for c in contracts}:
            out_contracts[mat_date] += [c for c in contracts if c.expiry == mat_date and strike_range[0] <= c.strike <= strike_range[1]]
        return out_contracts

    @staticmethod
    def strike_to_atm_distance(price_underlying: pd.Series, strikes: np.ndarray) -> pd.DataFrame:
        """
            x0: time; y...: strikes; z...: distances from ATM
            x0: time; y: priceUnderlying
            return: x0: Time; y[strike]: distance from ATM
        """
        strikes_ = sorted(set(strikes))
        strike_distance = Counter(np.array(strikes_[:-1]) - np.array(strikes_[1:])).most_common()[0][0]
        df = pd.DataFrame({strike: (price_underlying - strike) for strike in strikes_}, index=price_underlying.index)
        df = ((df.abs() // strike_distance) + 1) * np.sign(df)
        return df

    def list_missing_contracts(self, symbol: Equity, resolution, tick_type, security_type=SecurityType.option) -> Dict[datetime.date, List[OptionContract]]:
        missing_contracts = defaultdict(list)
        contracts = self.option_contracts(str(symbol))
        for contract in contracts:
            for dt in pd.date_range(start, end, freq='D'):
                if dt.weekday() in (5, 6) or dt.date() >= contract.expiry or dt.date() < contract.issue_date:
                    continue
                date = dt.strftime('%Y%m%d')
                underlying_folder = symbol.underlying_symbol.lower() if isinstance(symbol, OptionContract) else str(symbol).lower()
                for directory, subdirectories, files in os.walk(os.path.join(self.root, security_type, self.market, resolution, underlying_folder)):
                    for file in files:
                        if file.endswith('.zip') and file[:8] == date and tick_type in file:
                            csv_name = contract.csv_name(tick_type, resolution, dt)
                            with ZipFile(os.path.join(directory, file), 'r') as zipObj:
                                if csv_name not in zipObj.namelist():
                                    print(f'Missing {csv_name} in {file}')
                                    missing_contracts[dt].append(contract)
        return missing_contracts


if __name__ == '__main__':
    start = datetime.date(2023, 9, 5)
    end = datetime.date(2023, 10, 11)
    client = Client()
    sym = 'dell'
    equity = Equity(sym)

    # ivs = client.history([equity], start, end, Resolution.minute, TickType.iv_quote, SecurityType.equity)
    # print(ivs)
    contracts = client.central_volatility_contracts(equity, start, end, n=8)
    # contracts = contracts[datetime.date(2023, 8, 18)]
    # contracts = [c for c in contracts if c.strike == 16]
    contracts = list(chain(*contracts.values()))
    ivs = client.history(contracts, start, end, Resolution.second, TickType.iv_quote, SecurityType.option)
    # print(contracts)
    # ivs = client.history(contracts, start, end, Resolution.second, TickType.iv_quote, SecurityType.option)
    # ivs = client.history(list(chain(*contracts.values())), start, end, Resolution.second, TickType.iv_quote, SecurityType.option)
    # quotes = client.history(contracts, start, end, Resolution.minute, TickType.quote, SecurityType.option)
    # trades = client.history(contracts, start, end, Resolution.minute, TickType.trade, SecurityType.option)
    # underlying_quotes = client.history([Equity('hpe')], start, end, Resolution.minute, TickType.quote, SecurityType.equity)
    # print(underlying_quotes)

    # missing_contracts = client.list_missing_contracts(Equity('hpe'), Resolution.minute, TickType.quote)
    # print(missing_contracts)
    print('Done.')
