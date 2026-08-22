from market_data.qc.raw_data_processors import process_ticks_to_bars, process_hour_daily_upsample, get_configs
from options.types.enums import SecurityType
from shared.modules.logger import info

n_processes = 24
market = 'usa'


if __name__ == '__main__':
    v_ticker = [
        # 'NVDA',
        'PEP', 'PGR', 'FAST', 'DAL', 'CAG', 'JPM', 'UNH', 'WFC', 'C', 'BLK', 'STT', 'SCHW',
        'PLD', 'BK', 'ASML', 'NFLX', 'GS', 'ELV', 'TSM', 'ABT', 'MMC', 'AXP', 'CDNS', 'TMUS',
        'HSY', 'CBRE', 'PG', 'ON', 'MRK', 'PFE', 'PSA', 'O', 'VRSK', 'AMGN', 'MRNA',
        'LNG', 'XPO', 'PLTR', 'TSN', 'ONON', 'TJX', 'TGT', 'JD', 'AMAT', 'ROST',
        'BIDU', 'DKS', 'ADI', 'SNOW', 'RY', 'WDAY', 'MRVL', 'DLTR', 'ULTA', 'PDD',
        'HPE', 'CRM', 'CRWD', 'AVGO', 'DELL', 'DG', 'MDB', 'PATH', 'DOCU', 'ORCL',
        'FDX', 'SNX', 'MU', 'KMX', 'JBL', 'NKE', 'CCL', 'EXC', 'PANW', 'CSCO', 'FANG', 'ADBE'
    ]
    v_ticker = ['FDX']
    v_config_eq = get_configs(v_ticker, n_days_lookback=-20, takes=(-2,))
    v_config_op = get_configs(v_ticker, n_days_lookback=-1, takes=(-2,))
    n_processes = 4
    process_ticks_to_bars(v_config_eq, skip_zip=False, security_types=(SecurityType.equity,), n_processes=n_processes)
    process_hour_daily_upsample(v_config_eq, security_types=(SecurityType.equity,), n_processes=n_processes)

    process_ticks_to_bars(v_config_op, skip_zip=False, security_types=(SecurityType.option,), n_processes=n_processes)
    process_hour_daily_upsample(v_config_op, security_types=(SecurityType.option,), n_processes=n_processes)

    info(f'Done.')
