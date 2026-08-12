import datetime

from connector.raw_data_processors import process_ticks_to_bars, process_hour_daily_upsample, get_configs, upsample_iv, deleting_tmp_file, scan_root_for_bad_zip_members, \
    rm_duplicate_zip_entry_names, RawDataConfig, process_open_interest
from options.types.enums import SecurityType
from shared.modules.logger import info

n_processes = 4
market = 'usa'


if __name__ == '__main__':
    # rm_iv_quote_trade_tmp_files()
    # find_empty_option_data_days(None)
    # yyyyMMdd = '20241009'
    # clean_up(yyyyMMdd)
    # deleting_tmp_file(r'D:\trade\data\option\usa\tick')
    # deleting_tmp_file(r'D:\trade\data\equity\usa\tick')
    # scan_root_for_bad_zip_members(r'D:\trade\data\equity\usa\tick', remove=True)
    # scan_root_for_bad_zip_members(r'D:\trade\data\option\usa\tick', remove=True)
    # rm_duplicate_zip_entry_names(r'D:\trade\data\equity\usa\minute')
    # rm_duplicate_zip_entry_names(r'D:\trade\data\equity\usa\second')
    # for sec in ('option', 'equity'):
    #     scan_root_for_bad_zip_members(rf'D:\trade\data\{sec}\usa\daily', start_from='', remove=True)
    #     scan_root_for_bad_zip_members(rf'D:\trade\data\{sec}\usa\hour', start_from='', remove=True)
    #     scan_root_for_bad_zip_members(rf'D:\trade\data\{sec}\usa\minute', start_from='', remove=True)
    #     scan_root_for_bad_zip_members(rf'D:\trade\data\{sec}\usa\second', start_from='', remove=True)

    # v_config = [RawDataConfig(datetime.date(2025, 12, 16), datetime.date(2025, 12, 22), 'FDX')]

    v_ticker = ["AMAT"]

    # v_config_eq = []
    # v_config_op = []
    # for sym in v_ticker:
    #     dt = datetime.date(2026, 7, 10)
    #     v_config_eq += [RawDataConfig(start=dt, end=dt, tickers=sym)]
    #     v_config_op += [RawDataConfig(start=dt, end=dt, tickers=sym)]

    v_config_eq = get_configs(v_ticker, n_days_lookback=-20)#, takes=[-1])
    v_config_op = get_configs(v_ticker, n_days_lookback=-1)#, takes=[-1])

    process_ticks_to_bars(v_config_eq, skip_zip=False, security_types=(SecurityType.equity,), n_processes=n_processes)
    process_hour_daily_upsample(v_config_eq, security_types=(SecurityType.equity,), n_processes=n_processes)

    process_ticks_to_bars(v_config_op, skip_zip=False, security_types=(SecurityType.option,), n_processes=n_processes)
    process_hour_daily_upsample(v_config_op, security_types=(SecurityType.option,), n_processes=n_processes)

    process_open_interest(v_config_op)  # replace with polygon downloader

    info(f'Done.')
