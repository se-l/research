import datetime
import os
from typing import Set
from zipfile import ZipFile

import pandas as pd

root = r'C:\repos\trade\data'
market = 'usa'


def check(security_type, start, to):
    """Checks for data completeness in all zip files under
    root
    for symbols HPE,IPG,AKAM,AOS,A,MO,FL,ALL,ARE,ZBRA,AES,APD,ALLE,LNT,ZTS,ZBH,SPY
    for resolution daily, minute, tick, second, hour
    option and equity
    Create set of contracts expected on each day
    defined date range May 8th till today
    """
    pull_files = set()
    pull_zips = set()
    for resolution in expected_resolutions(security_type):
        for symbol in ['HPE', 'IPG', 'AKAM', 'AOS', 'A', 'MO', 'FL', 'ALL', 'ARE', 'ZBRA', 'AES', 'APD', 'ALLE', 'LNT', 'ZTS', 'ZBH']:
            # iterate over all dates in range
            for dt in pd.date_range(datetime.datetime.strptime(str(start), '%Y%m%d'), datetime.datetime.strptime(str(to), '%Y%m%d')):
                if dt.weekday() in (5, 6):
                    continue
                date = dt.strftime('%Y%m%d')
                contracts = set()
                for directory, subdirectories, files in os.walk(os.path.join(root, security_type, market, resolution, symbol.lower())):
                    if missing_zips := {os.path.join(directory, f) for f in expected_filenames(security_type, resolution, date) - set(files)}:
                        pull_zips = pull_zips.union(missing_zips)
                        print(f'Missing .zip file: {missing_zips}')
                    if security_type == 'option':
                        for file in files:
                            print(file)
                            # Extract the .zip files entries and add entry names to contracts set
                            if file.endswith('.zip') and file[:8] >= date and not '_trade_' in file and not '_quote_' in file:
                                print('Opening', os.path.join(directory, file))
                                expected_contracts = {fn for fn in contracts if fn.split('.')[0][-8:] > date}
                                with ZipFile(os.path.join(directory, file), 'r') as zipObj:
                                    files_contracts = set([f.split('.')[0] for f in zipObj.namelist()])
                                    contracts.union(files_contracts)
                                    missing_files = expected_contracts - files_contracts
                                    if missing_files:
                                        pull_files = pull_files.union(missing_files)
                                        print(f'Missing .csv: {date} {symbol} {resolution} {security_type} {missing_files}')
    return pull_files, pull_zips


def expected_resolutions(security_type) -> Set[str]:
    if security_type == 'option':
        return {'daily', 'tick'}
    else:
        return {'daily', 'second'}


def expected_filenames(security_type, resolution, date) -> Set[str]:
    if security_type == 'option':
        return {f'{date}_{cat}_american.zip' for cat in ['quote', 'trade', 'openinterest']}
    else:
        if resolution in ('daily', 'minute'):
            return {f'{date}_{cat}.zip' for cat in ['quote', 'trade']}
        else:
            return {f'{date}_{cat}.zip' for cat in ['trade']}


if __name__ == '__main__':
    for sec_type in ['option', 'equity']:
        print(check(sec_type, 20230601, 20230606))
