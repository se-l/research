import os
import re
import shutil
from dataclasses import dataclass
from typing import List

import pandas as pd

root_analytics = r'C:\repos\quantconnect\Lean\Launcher\bin\Analytics'


@dataclass
class Holding:
    symbol: str
    quantity: float
    price: float


if __name__ == '__main__':
    """
    2023-11-08T10:03:40.3371984Z TRACE:: Log: Initialized Position DELL with Holding: DELL: -52 @ 71.4115904
    2023-11-08T10:03:40.3372265Z TRACE:: Log: Initialized Position DELL  231124P00068000 with Holding: DELL  231124P00068000: 1 @ 2.223893
    """
    folder = r'live-interactive-iqfeed-U7293918\231204145306-AMarketMakeOptionsAlgorithm-trader.prod.1'
    fn = 'log.txt'
    fp_log = os.path.join(root_analytics, folder, fn)
    with open(fp_log, 'r') as f:
        log = f.read()

    # Parse log file.
    holdings: List[Holding] = []
    lns = re.findall(r'BrokerageSetupHandler.Setup\(\): Has existing holding: .*', log)
    for ln in lns:
        holdings.append(Holding(
            re.search(r'BrokerageSetupHandler.Setup\(\): Has existing holding: (.*@)', ln).group(1).split(':')[0],
            float(re.search(r'(.\d+) @', ln).group(1)),
            float(re.search(r'@ \$(\d+[\.]\d+).*', ln).group(1)),
        ))

    date = '231204'
    p_out = rf'C:\repos\quantconnect\Lean\Launcher\bin\Debug\BacktestingHoldings-{date}.csv'
    pd.DataFrame(holdings).rename(columns={'symbol': 'Symbol', 'quantity': 'Quantity', 'price': 'FillPrice'}).to_csv(p_out, index=False)
    shutil.copy(p_out, rf'C:\repos\quantconnect\Lean\Launcher\bin\BacktestingHoldings-{date}.csv')
