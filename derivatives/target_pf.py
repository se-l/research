from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Tuple

import json5
backtest_holdings_23Q4 = """
"""
# iteration_15.html
# DG    160	4	640
# ADBE	600	1	600
# ULTA	550	1	550
# JBL 	150	4	600

backtest_holdings_24Q1 = """
ORCL  240315C00113000,-4,1.0736
ORCL  240315P00113000,-4,1.0697
ORCL  240315C00114000,-4,1.0679
ORCL  240315P00114000,-4,1.0747

ORCL  250117P00120000,1,0.2991
ORCL  250117C00140000,1,0.2972
ORCL  250117P00125000,1,0.2931
ORCL  250117C00160000,1,0.2966
ORCL  250117P00115000,1,0.3008
ORCL  250117C00170000,1,0.2967
ORCL  250117C00145000,2,0.2963
ORCL  260116P00155000,1,0.3491
ORCL  250117C00185000,1,0.3015
ORCL  250620P00095000,1,0.3141
multiplier: 4, more if volas same...


ONON  240315C00034500,-5,1.7053
ONON  240315P00034500,-5,1.7308
ONON  240315C00034000,-5,1.6781
ONON  240315P00034000,-5,1.7283
ONON  260116C00050000,10,0.4979
ONON  260116P00015000,5,0.5243
ONON  250117C00045000,5,0.5603
ONON  250117P00042500,25,0.5139
ONON  250117C00042500,10,0.5710
ONON  250117P00035000,5,0.5398
ONON  250117P00030000,5,0.5479
ONON  250117P00015000,4.0,0.6882
multiplier: 5


DLTR  240315C00150000,-2,1.1289
DLTR  240315P00150000,-2,1.1038
DLTR  240315C00149000,-2,1.1215
DLTR  240315P00149000,-2,1.1100

DLTR  250117P00165000,6,0.3266
DLTR  250117C00190000,3.0,0.3087
DLTR  250117C00200000,3.0,0.3149
DLTR  250117P00155000,2,0.3279
DLTR  250117P00140000,2,0.3353
DLTR  250117C00250000,2,0.3065

DLTR  250117C00055000,2,0.6626
DLTR  250117C00120000,5,0.3578
DLTR  250117P00140000,5,0.3353
DLTR  250117P00120000,1,0.3571
DLTR  250117C00150000,1,0.3379
DLTR  250117C00195000,1,0.3076
DLTR,0,149.86

multiplier: 2


PATH  240315C00025000,-5,2.3694
PATH  240315P00025000,-5,2.3626
PATH  240315C00024500,-5,2.3710
PATH  240315P00024500,-5,2.3527

PATH  250117P00010000,10,0.6979
PATH  250117P00015000,10,0.6686
PATH  250117P00017500,5,0.6269
PATH  250117P00022500,10,0.6116
PATH  250117P00030000,10,0.6063
PATH  250117C00037000,15,0.6088
PATH  251219C00037000,5,0.5777

PATH  250117C00007500,2,0.6836
PATH  250117C00010000,7,0.6452
PATH  250117P00037000,5,0.6732
PATH,-33,24.875
multiplier: 5
"""


@dataclass
class ManualOrderInstruction:
    Symbol: str
    TargetQuantity: float
    Utility: float
    IV: float
    TimeToTrade: List[List[str]] = None

    def __post_init__(self):
        if self.TargetQuantity < 0:
            self.TimeToTrade = [["15:00", "16:00"]]
        else:
            self.TimeToTrade = [["09:50", "16:00"]]

    def __str__(self):
        return f'{self.Symbol.upper()},{self.TargetQuantity},{self.Utility},{self.IV},{self.TimeToTrade}'


def manual_order_instructions(s):
    multiplier = defaultdict(lambda: 1, {
        'CRWD': 5,
        'TGT': 5,
        'ROST': 2,
    })
    instructions = []
    for ln in s.split('\n'):
        if ln:
            symbol, target_quantity, iv = ln.split(',')
            underlying = symbol.split()[0]
            instructions.append(ManualOrderInstruction(symbol.upper(), float(target_quantity)*multiplier[underlying], 20, float(iv)))
    return json5.dumps(list(map(asdict, instructions)))


if __name__ == '__main__':
    # print(manual_order_instructions(backtest_holdings_23Q4))
    print(manual_order_instructions(backtest_holdings_24Q1))
