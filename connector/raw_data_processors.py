import shutil
import datetime
import multiprocessing
import pandas as pd

from itertools import groupby
from dataclasses import dataclass
from pathlib import Path
from typing import List

from connector.ib.enums import TradeType, Resolution
from connector.ib.generate_open_interest_files import gen_openinterest_files
from connector.ib.tick2qc_bar import upsample_ticks_with_args
from shared.constants import file_root, dt_fmt_ymd, EarningsPreSessionDates
from connector.ib.upsample_qc_equity_bars import upsample_equity_bars
from connector.ib.upsample_qc_option_bars import upsample_option_bars
from options.helper import add_trade_days
from options.types.enums import SecurityType, TickType
from shared.modules.logger import info
from shared.paths import Paths


@dataclass
class RawDataConfig:
    start: datetime.date
    end: datetime.date
    tickers: str


def clean_up(yyyyMMdd):
    """delete all files from root folder that start with 20230705"""
    for root, dirs, files in os.walk(file_root):
        for fn in files:
            if fn.startswith(yyyyMMdd):
                print(f'Removing file {os.path.join(root, fn)}')
                os.remove(os.path.join(root, fn))


def upsample_with_args_lst(args_lst: List[tuple]):
    for args in args_lst:
        upsample_with_args(args)


def upsample_with_args(args):
    sec_type, market, resolution_from, resolution_to, symbol, trade_type, start_date, end_date = args
    if sec_type == SecurityType.option:
        upsample_option_bars(sec_type, market, resolution_from, resolution_to, symbol, trade_type, start_date, end_date)
    elif sec_type == SecurityType.equity:
        upsample_equity_bars(sec_type, market, resolution_from, resolution_to, symbol, start_date, end_date)


def rm_tmp_files():
    for dir_, _, fns in os.walk(r'D:\trade\data'):
        for fn in [f for f in fns if f.endswith('.tmp')]:
            path = os.path.join(dir_, fn)
            print(f'Removing {path}')
            os.remove(path)


def rm_iv_quote_trade_tmp_files():
    for dir_, _, fns in os.walk(r'D:\trade\data'):
        for fn in fns:
            if 'iv_quote' in fn or 'iv_trade' in fn and 'nke' not in fn:
                path = os.path.join(dir_, fn)
                print(f'Removing {path}')
                # os.remove(path)


def rm_duplicate_zip_entry_names(directory):
    """
    Recursively iterates through the given directory, targeting .zip files.
    For each .zip file, searches for duplicate entry names inside the archive.
    Logs the respective sizes of duplicates and removes the smaller entry from the archive.
    """
    import tempfile
    import shutil

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                try:
                    with zipfile.ZipFile(zip_path, 'r') as archive:
                        # Group entries by name
                        name_to_entries = {}
                        for entry in archive.filelist:
                            name = entry.filename
                            if name not in name_to_entries:
                                name_to_entries[name] = []
                            name_to_entries[name].append(entry)

                        # Find duplicates
                        duplicates = {name: entries for name, entries in name_to_entries.items() if len(entries) > 1}

                        if duplicates:
                            print(f"Processing {zip_path} with duplicates: {list(duplicates.keys())}")

                            # For each duplicate group, log sizes and identify smaller ones to remove
                            to_remove = []
                            for name, entries in duplicates.items():
                                sizes = [(entry, entry.file_size) for entry in entries]
                                print(f"  Duplicate '{name}' sizes: {[(entry.filename, size) for entry, size in sizes]}")

                                # Sort by size, keep the largest, remove smaller
                                sizes.sort(key=lambda x: x[1], reverse=True)
                                for i in range(1, len(sizes)):  # Skip the largest
                                    to_remove.append(sizes[i][0].filename)

                            if to_remove:
                                # Create a new zip without the smaller duplicates
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
                                    temp_path = temp_zip.name

                                with zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED) as new_archive:
                                    for entry in archive.filelist:
                                        if entry.filename not in to_remove:
                                            with archive.open(entry.filename) as file_data:
                                                new_archive.writestr(entry.filename, file_data.read())

                                # Replace original with the new one
                                shutil.move(temp_path, zip_path)
                                print(f"  Removed smaller duplicates from {zip_path}: {to_remove}")

                except (zipfile.BadZipFile, OSError) as e:
                    print(f"Error processing {zip_path}: {e}")


def deleting_tmp_file(directory: str | Path):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tmp'):
                print(f"Deleting tmp file: {os.path.join(root, file)}")
                os.remove(os.path.join(root, file))


def _scan_one(file_path: Path):
    bad_members = is_bad_zip(file_path)
    return file_path, bad_members


def scan_root_for_bad_zip_members(directory: str | Path, start_from=None, remove=False):
    directory = Path(directory)

    zip_files = [
        Path(root) / file
        for root, _, files in os.walk(directory)
        for file in files
        if file.endswith('.zip')
    ]
    if start_from:
        try:
            zip_files = zip_files[zip_files.index(start_from):]
        except IndexError:
            return
    total_files = len(zip_files)

    if total_files == 0:
        print(f"No ZIP files found under {directory}\n", flush=True)
        return

    scanned = 0
    with multiprocessing.Pool(processes=20) as pool:
        for file_path, is_bad in pool.imap_unordered(_scan_one, zip_files, chunksize=1):
            scanned += 1
            pct = (scanned / total_files) * 100
            try:
                print(f"\rScanning {scanned}/{total_files} ({pct:.1f}%) : {file_path} ", end='', flush=True)
            except Exception as e:
                info(str(e), RuntimeWarning)

            if is_bad:
                print(f"\nDeleting zip file with bad member: {file_path}\n")
                if remove:
                    os.remove(file_path)

    print("\n", flush=True)


def is_bad_zip(zip_path: str | Path):
    try:
        zip_path = Path(zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            first_bad = zf.testzip()
            if first_bad:
                return True

            for info in zf.infolist():
                with zf.open(info, "r") as f:
                    while f.read(1024 * 1024):
                        pass
    except Exception as e:
        return True
    return False


def process_ticks_to_bars(configs: List[RawDataConfig], skip_zip=True, n_processes=16, market='usa', security_types=(SecurityType.equity, SecurityType.option)):
    arg_list = []
    for cfg in configs:
        for dt in pd.date_range(cfg.start, cfg.end):
            for security_type in security_types:
                for tick_type in [TickType.quote, TickType.trade]:
                    for resolution in [Resolution.second, Resolution.minute]:
                        for symbol in cfg.tickers.split(','):
                            arg_list.append((security_type, market, resolution, symbol.lower(), tick_type, dt.strftime(dt_fmt_ymd), dt.strftime(dt_fmt_ymd), True, skip_zip))
    processed = 0

    def update_progress(result):
        nonlocal processed
        processed += 1
        pct = (processed / len(arg_list)) * 100
        progress = f"\rprocess_ticks_to_bars(): {pct:.1f}%"
        print(progress, end='', flush=True)

    with multiprocessing.Pool(min(n_processes, len(arg_list))) as pool:
        results = [pool.apply_async(upsample_ticks_with_args, args, callback=update_progress) for args in arg_list]
        for res in results:
            res.get()


def process_hour_daily_upsample(configs: List[RawDataConfig], n_processes=16, security_types=(SecurityType.equity, SecurityType.option)):
    arg_list = []
    for cfg in configs:
        for symbol in cfg.tickers.split(','):
            for res in [Resolution.hour, Resolution.daily]:
                for security_type in security_types:
                    arg_list.append((security_type, 'usa', Resolution.minute, res, symbol.lower(), TradeType.trade, cfg.start.strftime(dt_fmt_ymd), cfg.end.strftime(dt_fmt_ymd)))
                    arg_list.append((security_type, 'usa', Resolution.minute, res, symbol.lower(), TradeType.quote, cfg.start.strftime(dt_fmt_ymd), cfg.end.strftime(dt_fmt_ymd)))
    grouped_args = [list(el[1]) for el in groupby(arg_list, lambda x: x[4])]
    processed = 0

    def update_progress(result):
        nonlocal processed
        processed += 1
        pct = (processed / len(grouped_args)) * 100
        progress = f"\rprocess_hour_daily_upsample(): {pct:.1f}%"
        print(progress, end='', flush=True)

    with multiprocessing.Pool(n_processes) as pool:
        results = [pool.apply_async(upsample_with_args_lst, (args,), callback=update_progress) for args in grouped_args]
        for res in results:
            res.get()


def transform(configs: List[RawDataConfig], skip_zip=False):
    process_ticks_to_bars(configs, skip_zip)
    process_hour_daily_upsample(configs)
    process_open_interest(configs)

def process_open_interest(configs: List[RawDataConfig]):
    # open interest files
    total_configs = len(configs)
    processed = 0
    for cfg in configs:
        for resolution in [Resolution.tick, Resolution.minute, Resolution.second, Resolution.daily]:
            for sym in cfg.tickers.split(','):
                gen_openinterest_files(sec_type=SecurityType.option, market='usa', resolution=resolution, symbol=sym, start_date=cfg.start.strftime(dt_fmt_ymd))
        processed += 1
        progress = f"\r{(processed / total_configs) * 100:.1f}%"
        print(progress, end='', flush=True)


def move_live_data(dt_in: datetime.date, rev=False):
    dt = dt_in.strftime(dt_fmt_ymd)
    for directory, subdirectories, files in os.walk(file_root.replace('data', 'dataLive') if rev else file_root):
        for fn in files:
            if fn.startswith(dt):
                print(f'Moving {os.path.join(directory, fn)} to live data folder')
                target_folder = directory.replace('dataLive', 'data') if rev else directory.replace('data', 'dataLive')
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                if not os.path.exists(os.path.join(target_folder, fn)):
                    os.rename(os.path.join(directory, fn), os.path.join(target_folder, fn))
                else:
                    os.remove(os.path.join(directory, fn))


def upsample_iv(configs: List[RawDataConfig]):
    for cfg in configs:
        for symbol in cfg.tickers.split(','):
            for res in [Resolution.daily]:
                upsample_with_args(
                    (SecurityType.option, 'usa', Resolution.second, res, symbol.lower(), TradeType.iv_quote, cfg.start.strftime(dt_fmt_ymd), cfg.end.strftime(dt_fmt_ymd)))
                upsample_with_args(
                    (SecurityType.option, 'usa', Resolution.second, res, symbol.lower(), TradeType.iv_trade, cfg.start.strftime(dt_fmt_ymd), cfg.end.strftime(dt_fmt_ymd)))


def cp2local(sym):
    src_dir = file_root
    fn2copy = lambda fn_, dir_: fn in ('market-hours-database.json', 'interest-rate.csv') or sym in fn_ or dir_ in (sym, 'symbol-properties')
    for directory, subdirectories, files in os.walk(src_dir):
        dir_name = Path(directory).name
        for fn in files:
            if fn2copy(fn, dir_name):
                tgt = os.path.join(directory, fn)
                dst = os.path.join(directory.replace(src_dir, Paths.path_data), fn)
                tgt_dir = Path(dst).parent
                if not os.path.exists(tgt_dir):
                    os.makedirs(tgt_dir)
                if ('minute' in tgt or Resolution.second in tgt or 'tick' in tgt) and os.path.exists(dst):
                    continue
                shutil.copy(tgt, dst)
                print(f'Copy {tgt} -> {dst}')


def scan_for_missing_tick_csv_entries():
    pass


def scan_for_missing_bar_csv_entries(sym: str, sec_type: str, start: datetime.date, end: datetime.date):
    """
    Given an underlying and security type and date range
    get the file entries for all ticks files
    removal of the date in the entry name yields a symbol key
    min max available dates for each symbol key is updated
    if any file with date x in between min max range does not include a file entry for symbol key
    then flag it up
    """
    pass


def get_v_empty_option_fp(ticker: str = None) -> List[Path]:
    """Occassionally, entire days contain option .csv files without any data. Likely a downloading error.
    Find, delete and redownload these days

    Refactor C# process to request for every single file that has zero quotes, trades ...
    """
    empty = []
    root_ticks = Paths.path_data.joinpath('option', 'usa', 'tick')
    for dir_, folders, fns in os.walk(root_ticks):
        if ticker and not dir_.endswith((ticker or '').lower()):
            continue
        if folders:
            continue

        dates_of_interest = set()
        sym = dir_.split('\\')[-1].upper()
        for take in range(-30, 0):
            try:
                release_date = EarningsPreSessionDates(sym)[take]
            except IndexError:
                continue
            for d_days in range(-20, 2, 1):
                dates_of_interest.add(add_trade_days(release_date, d_days))

        print(f'get_v_empty_option_fp(): Checking {sym} # dates={len(dates_of_interest)} in path={dir_}')

        for fn in filter(lambda x: x.endswith('.zip'), fns):
            fp = os.path.join(dir_, fn)
            dt = datetime.datetime.strptime(fn[:8], dt_fmt_ymd)
            # print(f'Checking {dt.date()} at {path}')
            if dt.date() in dates_of_interest:
                try:
                    with zipfile.ZipFile(fp, 'r') as archive:
                        if all((f.file_size <= 0 for f in archive.infolist())):
                            info(f'get_v_empty_option_fp(): All .csv empty in {fp}')
                            empty.append(fp)
                except Exception as e:
                    print(f'Error: {fp} = {e}')
                    empty.append(fp)

    return [Path(x) for x in empty]


def get_configs(v_ticker: List[str], n_days_lookback=-1, n_days_lookahead=2, min_release_date=None, takes=()) -> List[RawDataConfig]:
    """Sorted by release date. Easy to continue where left off"""
    ea_configs = []
    for take in takes or range(-30, 0):
        for ticker in v_ticker:
            try:
                ea_date = EarningsPreSessionDates(ticker)[take]
            except IndexError:
                continue

            if ea_date.year < 2024:
                continue

            if min_release_date and min_release_date > ea_date:
                continue

            start = add_trade_days(ea_date, n_days_lookback)
            end = add_trade_days(ea_date, n_days_lookahead)
            ea_configs.append((ea_date, RawDataConfig(start, end, ticker)))
    return [c[1] for c in sorted(ea_configs, key=lambda x: x[0])]

# def transform(configs: List[RawDataConfig], skip_zip=True):
#     """refactor mp part and loop into where f was imported from """
#     arg_list = []
#     for cfg in configs:
#         # Ticks to Bar
#         for dt in pd.date_range(cfg.start, cfg.end):
#             # dt_end = dt + pd.Timedelta(days=1)
#             # if dt_end.strftime(dt_fmt_ymd) > end.strftime(dt_fmt_ymd):
#             #     break
#             for security_type in [SecurityType.equity, SecurityType.option]:
#                 for tick_type in [TickType.quote, TickType.trade]:
#                     for resolution in [Resolution.second, Resolution.minute]:
#                         for symbol in cfg.tickers.split(','):
#                             arg_list.append((security_type, market, resolution, symbol.lower(), tick_type, dt.strftime(dt_fmt_ymd), dt.strftime(dt_fmt_ymd), True, skip_zip))
#     with multiprocessing.Pool(min(n_processes, len(arg_list))) as pool:
#         pool.starmap(upsample_ticks_with_args, arg_list)
#
#     # Upsample HOUR/DAILY - MP grouped by symbol
#     # Tuples to be refactored to types.
#     arg_list = []
#     for cfg in configs:
#         for symbol in cfg.tickers.split(','):
#             for res in [Resolution.hour, Resolution.daily]:
#                 arg_list.append((SecurityType.option, 'usa', Resolution.minute, res, symbol.lower(), TradeType.trade, cfg.start.strftime(dt_fmt_ymd), cfg.end.strftime(dt_fmt_ymd)))
#                 arg_list.append((SecurityType.option, 'usa', Resolution.minute, res, symbol.lower(), TradeType.quote, cfg.start.strftime(dt_fmt_ymd), cfg.end.strftime(dt_fmt_ymd)))
#                 arg_list.append((SecurityType.equity, 'usa', Resolution.minute, res, symbol.lower(), TradeType.trade, cfg.start.strftime(dt_fmt_ymd), cfg.end.strftime(dt_fmt_ymd)))
#                 arg_list.append((SecurityType.equity, 'usa', Resolution.minute, res, symbol.lower(), TradeType.quote, cfg.start.strftime(dt_fmt_ymd), cfg.end.strftime(dt_fmt_ymd)))
#     # for args in arg_list:  # no mp because target is a single file. conflicts
#     #     upsample_with_args(args)
#     with multiprocessing.Pool(n_processes) as pool:
#         pool.map(upsample_with_args_lst, (list(el[1]) for el in groupby(arg_list, lambda x: x[4])))
#
#     # open interest files
#     for cfg in configs:
#         for resolution in [Resolution.tick, Resolution.minute, Resolution.second, Resolution.daily]:
#             for sym in cfg.tickers.split(','):
#                 gen_openinterest_files(sec_type=SecurityType.option, market='usa', resolution=resolution, symbol=sym, start_date=cfg.start.strftime(dt_fmt_ymd))


def try_open_file(file_path: str) -> bool:
    """
    Try to open and read a file to check if it's healthy.
    Returns True if file can be opened/read, False if broken.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()  # Try to read the entire file
        return True
    except Exception:
        return False

import os
import zipfile
from pathlib import Path

def fix_entries_to_zip(fix_entries: list) -> list:
    """
    Process fix_entries, try to open each file, and save healthy ones as .zip
    """
    saved_files = []
    broken_files = []

    for fix_path in fix_entries:
        fix_path = Path(fix_path)

        if not fix_path.exists():
            print(f"File not found: {fix_path}")
            continue

        if fix_path.suffix.lower() != '.fixme':
            print(f"Skipping non-.fixme file: {fix_path}")
            continue

        # Try to open the file
        if not try_open_file(str(fix_path)):
            print(f"BROKEN: {fix_path.name}")
            broken_files.append(fix_path)
            continue

        # File is healthy - create .zip
        zip_path = fix_path.with_suffix('.zip')

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Read the CSV content and add to zip
            with open(fix_path, 'r', encoding='utf-8') as f:
                csv_content = f.read()

            # Use the CSV filename inside the zip (without fixxme)
            csv_filename = fix_path.stem + '.csv'
            zf.writestr(csv_filename, csv_content)

        print(f"SAVED: {fix_path.name} -> {zip_path.name}")
        saved_files.append(str(zip_path))

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Healthy (saved): {len(saved_files)}")
    print(f"Broken (excluded): {len(broken_files)}")

    return saved_files


if __name__ == '__main__':
    scan_root_for_bad_zip_members(r'D:\trade\data\option\usa\tick\fdx', remove=True)
    scan_root_for_bad_zip_members(r'D:\trade\data\equity\usa\tick\fdx', remove=True)
    # fix_entries = [
    #     r'D:\trade\data\option\usa\tick\fdx\20241219_quote_american.zip',
    # ]
    # fix_entries_to_zip(fix_entries)

    # v_fp = get_v_empty_option_fp()
    # for p in v_fp:
    #     print(f'Deleted {p}')
    #     os.remove(p)
    # print(len(v_fp))