import datetime

from connector.ib.enums import WhatToShow, IBSecType, Resolution
from connector.ib.ib_client import fetch
from connector.raw_data_processors import RawDataConfig, transform
from options.helper import add_trade_days
from shared.constants import EarningsPreSessionDates

def ps_dl_data(cfg: RawDataConfig):
    # Download equity
    fetch(symbols=[cfg.tickers], start=cfg.start.isoformat(), end=cfg.end.isoformat(), what_to_show=WhatToShow.TRADES, sec_type=IBSecType.STK, resolution=Resolution.tick)
    fetch(symbols=[cfg.tickers], start=cfg.start.isoformat(), end=cfg.end.isoformat(), what_to_show=WhatToShow.BID_ASK, sec_type=IBSecType.STK, resolution=Resolution.tick)


def download():
    configs = []

    ticker_lst = [
        # 'NVDA',
        'PEP', 'PGR', 'FAST', 'DAL', 'CAG', 'JPM', 'UNH', 'WFC', 'C', 'BLK', 'STT', 'SCHW',
        'PLD', 'BK', 'ASML', 'NFLX', 'GS', 'ELV', 'TSM', 'ABT', 'MMC', 'AXP', 'CDNS', 'TMUS',
        'HSY', 'CBRE', 'PG', 'ON', 'MRK', 'PFE', 'PSA', 'O', 'VRSK', 'AMGN', 'MRNA',
        'LNG', 'XPO', 'PLTR', 'TSN', 'ONON', 'TJX', 'TGT', 'JD', 'AMAT', 'ROST',
        'BIDU', 'DKS', 'ADI', 'SNOW', 'RY', 'WDAY', 'MRVL', 'DLTR', 'ULTA', 'PDD',
        'HPE', 'CRM', 'CRWD', 'AVGO', 'DELL', 'DG', 'MDB', 'PATH', 'DOCU', 'ORCL',
        'FDX', 'SNX', 'MU', 'KMX', 'JBL', 'NKE', 'CCL', 'EXC', 'PANW', 'CSCO', 'FANG', 'ADBE'
    ]
    # ticker_lst = ['DAL']
    for take in [-1]:
        for ticker in ticker_lst:
            # start = datetime.date(2024, 10, 14)
            # end = datetime.date(2024, 10, 14)
            # configs.append(RawDataConfig(start, end, ticker))

            for d_days in [-20, -15, -10, -5, -3, -2, -1, 0, 1, 2]:
                release_date = EarningsPreSessionDates(ticker)[take]
                start = end = add_trade_days(release_date, d_days)
                if start < datetime.date.today():
                    configs.append(RawDataConfig(start, end, ticker))

    # IB
    for cfg in configs:
        ps_dl_data(cfg)  # move this to docker too. Toolbox image. Every ticker and date to its own container, max ~16 containers

    transform(configs)

    print(f'Done.')


if __name__ == '__main__':
    download()
