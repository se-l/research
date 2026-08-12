import datetime

from collections import defaultdict

from derivatives import Resolution
from derivatives import Equity
from derivatives import OptionContract
from derivatives import Option


if __name__ == '__main__':
    """
    Todo: Check how much the extrinsic value decreases towards release date. Want to short/long more than 1 D in advance?? IV goes up. Leaving money on the table?
    """
    sym = 'PANW'
    equity = Equity(sym)
    resolution = Resolution.second
    seq_ret_threshold = 0.005

    s0 = 283.90
    # s0 = 370
    # s0 = 320
    pf = defaultdict(int, {
    # 'PANW':	343,
    'PANW  240223P00365000': -2,
    'PANW  240223P00367500': -2,
    'PANW  240223C00365000': -2,
    'PANW  240223C00367500': -2,
    'PANW  241220P00260000': 2,
    'PANW  241220P00290000': 2,
    'PANW  241220P00300000': 2,
    'PANW  241220P00310000': 2,
    'PANW  241220P00320000': 2,
    'PANW  241220P00330000': 2,
    'PANW  241220P00370000': 2,
    'PANW  241220C00490000': 2,
    'PANW  241220C00500000': 2,
    'PANW  241220C00520000': 2,
    'PANW  241220C00540000': 2,
    'PANW  250117C00520000': 2,
    'PANW  250117C00540000': 2,
    })
    calculationDate = datetime.date(2024, 2, 21)
    price_underlying = s0
    volatility = 0.38
    pf = {Option(OptionContract.from_ib_symbol(k), calculationDate, price_underlying, volatility): v for k, v in pf.items() if v != 0}
    deltaTotalPf0Total = sum([o.delta(volatility, s0, calculationDate) * q * 100 for o, q in pf.items()])
    for o, q in pf.items():
        print(f'{o}: Strike:{o.strike}, Delta: {o.delta(volatility, s0, calculationDate):2f}, Quantity: {q}, DeltaTotal: {o.delta(volatility, s0, calculationDate) * q * 100:0f}')
    deltaTotalPf0Total += 343
    print(f'Portfolio Equity Position: {deltaTotalPf0Total}')

    print('Done')
