import datetime

from decimal import Decimal
from dataclasses import dataclass
from options.typess.enums import OptionRight, OptionStyle, TickType, Resolution
from options.typess.symbol import Symbol


@dataclass
class OptionContract(Symbol):
    symbol: str
    underlying_symbol: str
    expiry: datetime.date
    strike: Decimal
    right: str
    issue_date: datetime.date = None
    option_style: str = OptionStyle.american
    multiplier: int = 100

    @classmethod
    def from_filename(cls, filename: str, issue_date=None):
        if filename[:8].isdigit():  # Minute: 20230627_akam_minute_quote_american_put_1800000_20240119.csv"
            dt, symbol, resolution, tick_type, option_style, option_right, strike, expiry = filename.split('.')[0].split('_')
        else:  # Daily: are_quote_american_put_1950000_20230721.csv
            symbol, tick_type, option_style, option_right, strike, expiry = filename.split('.')[0].split('_')
        expiry = datetime.datetime.strptime(expiry, '%Y%m%d').date()
        option_right = OptionRight.from_string(option_right)
        option_style = OptionStyle.from_string(option_style)
        strike = Decimal(strike) / 10000
        return cls(symbol, symbol, expiry, strike, option_right, issue_date, option_style)

    @classmethod
    def from_contract_nm(cls, contract_nm: str, issue_date=None):
        symbol, exercise, option_right, strike, expiry = contract_nm.split('_')
        expiry = datetime.datetime.strptime(expiry, '%Y%m%d').date()
        option_right = OptionRight.from_string(option_right)
        strike = Decimal(strike) / 10000
        return cls(symbol, symbol, expiry, strike, option_right, issue_date)

    def csv_name(self, tick_type: TickType, resolution: Resolution, date: datetime.date = None):
        if resolution in (Resolution.minute, Resolution.second, Resolution.tick) and date:  # for minute, second, tick
            return f'{date.strftime("%Y%m%d")}_{self.symbol}_{resolution}_{tick_type}_{self.option_style}_{self.right}_{int(self.strike * 10000)}_{self.expiry.strftime("%Y%m%d")}.csv'
        else:
            return f'{self.underlying_symbol}_{tick_type}_{self.option_style}_{self.right}_{int(self.strike * 10000)}_{self.expiry.strftime("%Y%m%d")}.csv'

    def zip_name(self, tick_type: TickType, resolution: Resolution, date: datetime.date = None):
        if resolution in (Resolution.daily, Resolution.hour):
            return f'{self.underlying_symbol}_{date.year}_{tick_type}_american.zip'
        else:
            return f'{date.strftime("%Y%m%d")}_{tick_type}_american.zip'

    def ib_symbol(self):
        # print('WARNING: strike may not be accurate')
        return f'{self.underlying_symbol.upper()}  {self.expiry.strftime("%y%m%d")}{self.right[0].upper()}{str(int(self.strike*1000)).zfill(8)}'

    @classmethod
    def from_ib_symbol(cls, ib_symbol: str):
        """PANW  250117C00540000"""
        underlying_symbol = ib_symbol.split(' ')[0]
        # use regex to extract expiry, right, strike
        expiry = datetime.datetime.strptime(ib_symbol.split(' ')[-1][:6], '%y%m%d').date()
        right = OptionRight.from_string(ib_symbol.split(' ')[-1][6])
        # print('WARNING: strike may not be accurate')
        strike = Decimal(ib_symbol.split(' ')[-1][7:]) / 1000
        return cls(underlying_symbol, underlying_symbol, expiry, strike, right)

    def __repr__(self):
        return f'{self.underlying_symbol.lower()}_{self.option_style}_{self.right}_{int(self.strike * 10000)}_{self.expiry.strftime("%Y%m%d")}'

    def __hash__(self):
        return hash(self.__repr__())


def cleanup_low_res_files(symbol: Symbol):
    """
    1. Reading in all annual option quote files for a given symbol
    2. Derive option contracts from name
    3. For each set boundary of tradeable dates by whether by min max date with close bid or ask in daily quote files
    4. Loop through all low min, sec, tick files and delete corresponding contract if not in tradeable dates
    """
    pass
    # import os
    # import pandas as pd
    # from connector.quantconnect.typess.enums import TickType, Resolution
    # from connector.quantconnect.typess.symbol import Symbol
    #
    # option_contracts = set()
    # for filename in os.listdir('data/option/minute/ARE'):
    #     option_contracts.add(OptionContract.from_filename(filename))
    # for filename in os.listdir('data/option/second/ARE'):
    #     option_contracts.add(OptionContract.from_filename(filename))
    # for filename in os.listdir('data/option/tick/ARE'):
    #     option_contracts.add(OptionContract.from_filename(filename))
    # option_contracts = list(option_contracts)
    # option_contracts.sort(key=lambda x: x.maturity_date)
    # option_contracts.sort(key=lambda x: x.strike)
    # option_contracts.sort(key=lambda x: x.right)
    # option_contracts.sort(key=lambda x: x.option_style)
    # option_contracts.sort(key=lambda x: x.symbol)
    #
    # # Get min max date for each contract
    # for option_contract in option_contracts:
    #     print(option_contract)
    #     min_date = None
    #     max_date = None
    #     for filename in os.listdir(f'data/option/daily/{symbol.symbol}'):
    #         if option_contract.symbol in filename:
    #             df = pd.read_csv(f'data/option/daily/{symbol.symbol}/{filename}')
    #             df['date'] = pd.to_datetime(df['date'])
    #             df = df.set_index('date')
    #             df = df.loc[df['close_ask'] > 0]
    #             df = df.loc[df['close_bid'] > 0]
    #             if min_date is None:
    #                 min_date = df.index.min()
    #                 max_date = df.index.max()
    #             else:
    #                 min_date = min(min_date, df.index.min())
    #                 max_date = max(max_date, df.index.max())
    #     print(min_date, max_date)
    #
    #     # Delete files not in tradeable dates
    #     for filename in os.listdir(f'data/option/minute/{symbol.symbol}'):
    #         if option_contract.symbol in filename:
    #             date = datetime.datetime.strptime(filename.split('_')[0], '%Y%m%d').date()
    #             if date < min_date or date > max_date:
    #                 print(f'deleting {filename}')
    #                 os.remove(f'data/option/minute/{symbol.symbol}/{filename}')


if __name__ == '__main__':
    OptionContract.from_filename('are_quote_american_put_1950000_20230721.csv')
