import multiprocessing

from market_data.qc.raw_data_processors import process_ticks_to_bars, process_hour_daily_upsample, get_configs, \
    process_open_interest, RawDataConfig
from options.types.enums import SecurityType
from shared.modules.logger import info
from datetime import date

n_processes = multiprocessing.cpu_count() // 2

def transform_earnings(v_ticker, takes=None):
    v_config_eq = get_configs(v_ticker, n_days_lookback=-20, takes=takes)
    v_config_op = get_configs(v_ticker, n_days_lookback=-1, takes=takes)
    transform(v_config_eq, v_config_op)


def transform_dates(v_ticker, dt):
    v_config_eq = []
    v_config_op = []
    for sym in v_ticker:
        v_config_eq += [RawDataConfig(start=dt, end=dt, tickers=sym)]
        v_config_op += [RawDataConfig(start=dt, end=dt, tickers=sym)]
    transform(v_config_eq, v_config_op)

def transform(v_config_eq=None, v_config_op=None):
    if v_config_eq:
        process_ticks_to_bars(v_config_eq, skip_zip=True, security_types=(SecurityType.equity,), n_processes=n_processes)
        process_hour_daily_upsample(v_config_eq, security_types=(SecurityType.equity,), n_processes=n_processes)

    if v_config_op:
        process_ticks_to_bars(v_config_op, skip_zip=True, security_types=(SecurityType.option,), n_processes=n_processes)
        process_hour_daily_upsample(v_config_op, security_types=(SecurityType.option,), n_processes=n_processes)

    process_open_interest(v_config_op)  # replace with polygon downloader


if __name__ == '__main__':
    v_ticker = ["DKS"]
    transform_earnings(v_ticker)

    # transform_dates(v_ticker, date(2099, 1, 1))

    # transform(
    #     [RawDataConfig(date(2025, 12, 16), date(2025, 12, 22), 'FDX')],
    #     [RawDataConfig(date(2025, 12, 16), date(2025, 12, 22), 'FDX')],
    # )

    info(f'Done.')
