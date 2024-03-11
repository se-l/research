import os
import datetime
import json5

from shared.paths import Paths

dividend_yield = {
    "_": 0.0,
    "DELL": 0.0175,
    "CSCO": 0.0299,
    "HPE": 0.0327,
    "ORCL": 0.014,
    "PFE": 0.00622,
    "AMAT": 0.0064,
    "HSBC": 0.0505,
    "PANW": 0.0,
    "PSA": 0.0426,
    "O": 0.0588,
    "CSGP": 0.0,
    "FANG": 0.0571,
    "RIO": 0.0448,
    "ADI": 0.0183,
    "VRSK": 0.0056,
    "EXC": 0.0412,
    "NVDA": 0.0023,
    "SPNS": 0.0169,
    "MRNA": 0.0,
    "WDAY": 0.0,
    "SNOW": 0.0,
    "CRM": 0.0,
    "TJX": 0.0134,
    "RY": 0.0417,
    "BIDU": 0.0,
    "MDB": 0.0,
    "CRWD": 0.0,
    "AVGO": 0.015,
    "JD": 0.0269,
    "TGT": 0.0283,
    "ROST": 0.009,
    "MRVL": 0.0031,
    "ONON": 0.0,
    "DLTR": 0.0,
    "PATH": 0.0,
    "ADBE": 0.0,
    "ULTA": 0.0,
    "DG": 0.0148,
    "JBL": 0.0021
  }

path_amm_cfg = r'C:\repos\quantconnect\Lean\Algorithm.CSharp\AMarketMakeOptionsAlgorithmConfig.json'
amm_cfg = None
if os.path.exists(path_amm_cfg):
    with open(path_amm_cfg, 'r') as fh:
        amm_cfg = json5.load(fh)

with open(Paths.path_earnings, 'r') as fh:
    earnings_cfg = json5.load(fh)


file_root = r"D:\trade\data"

DiscountRateMarket = amm_cfg['DiscountRateMarket'] if amm_cfg else 0.0435  # https://www.bloomberg.com/markets/rates-bonds/government-bonds/us
DividendYield = amm_cfg['DividendYield'] if amm_cfg else dividend_yield
EarningsPreSessionDates = lambda sym: sorted([datetime.date.fromisoformat(x['Date']) for x in earnings_cfg if x['Symbol'].upper() == sym.upper()])
