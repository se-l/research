from dataclasses import dataclass
from datetime import date, datetime
from pprint import pprint
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from options.client import read_option_csvs_from_zip, Client, CsvNmSymDt
from options.helper import get_tenor
from options.typess.enums import Resolution, SecurityType, TickType, OptionRight
from options.typess.equity import Equity
from options.typess.option_contract import OptionContract

@dataclass
class Stats:
    tenor: float
    right: OptionRight
    bid_iv: float
    ask_iv: float

@dataclass
class StatsDiff:
    tenor: float
    right: OptionRight
    strike: float
    df: pd.DataFrame

def print_avg_iv_by_tenor_right():
    start = date(2024, 2, 22)
    end = date(2024, 2, 22)
    resolution = Resolution.second
    security_type = SecurityType.option
    client = Client()
    sym = 'FDX'
    equity = Equity(sym)
    # zip_path = r'D:\trade\data\option\usa\second\fdx\20240222_quote_american_iv.zip'
    tick_type = TickType.iv_quote
    dummy_contract = OptionContract(str(equity), str(equity), None, None, None)
    zip_paths = client.get_zip_paths([dummy_contract], tick_type, resolution, security_type, start, end)

    zip_path = zip_paths[0]
    underlying = Equity(zip_path.parent.name.upper())
    dt = datetime.strptime(zip_path.name[:8], "%Y%m%d").date()
    csv_names: List[str] = client.get_csv_names_in_zip(zip_path)
    csvNmSymDts = [CsvNmSymDt(nm, OptionContract.from_filename(nm), dt) for nm in csv_names]

    csv_nm_df = read_option_csvs_from_zip(zip_path, csvNmSymDts, tick_type)

    stats_lst = []
    for csv_nm, df in csv_nm_df.items():
        contract = OptionContract.from_filename(csv_nm)
        calc_date = df.index[0].date()
        tenor = get_tenor(contract.expiry, calc_date)
        stats_lst.append(Stats(tenor, contract.right, df['bid_iv'].mean(), df['ask_iv'].mean()))

    stats_lst2 = [el for el in stats_lst if el.bid_iv > 0 and el.ask_iv > 0]
    df = pd.DataFrame(stats_lst2).groupby(['tenor', 'right']).mean().sort_values(['tenor', 'right'])
    df['mid'] = (df['bid_iv'] + df['ask_iv']) / 2
    pprint(df['mid'])

    # tenor     right
    # 0.002740  call     0.298718
    #           put      0.343586
    # 0.021918  call     0.325294
    #           put      0.245799
    # 0.041096  call     0.255454
    #           put      0.448595
    # 0.060274  call     0.311433
    #           put      0.269419
    # 0.079452  call     0.310942
    #           put      0.320275
    # 0.095890  call     0.287916
    #           put      0.304588
    # 0.117808  call     0.274914
    #           put      0.268128
    # 0.156164  call     0.299703
    #           put      0.298686
    # 0.232877  call     0.259312
    #           put      0.282080
    # 0.328767  call     0.242665
    #           put      0.273199
    # 0.405479  call     0.246546
    #           put      0.295894
    # 0.578082  call     0.270992
    #           put      0.286269
    # 0.654795  call     0.267195
    #           put      0.289348
    # 0.827397  call     0.281896
    #           put      0.297047
    # 0.904110  call     0.285541
    #           put      0.328486
    # 1.326027  call     0.304232
    #           put      0.294618
    # 1.575342  call     0.293124
    #           put      0.293476
    # 1.824658  call     0.292189
    #           put      0.286762
    # 1.901370  call     0.292583
    #           put      0.282257

def magic(x):
    if len(x) < 2: return np.nan
    df_c = [i['df'] for ix, i in x.iterrows() if i['right'] == 'call'][0][['bid_iv', 'ask_iv']]
    df_p = [i['df'] for ix, i in x.iterrows() if i['right'] == 'put'][0][['bid_iv', 'ask_iv']]
    df = pd.concat([df_c, df_p], axis=1).sort_index()
    for i, c in enumerate(df.columns):
        df.iloc[:, i] = df.iloc[:, i].bfill().ffill()
    df = df[~((df.isna() | df ==0).any(axis=1))]
    if df.shape[0] == 0:
        return np.nan
    return ((df.iloc[:, 1] - df.iloc[:, 0]) - (df.iloc[:, 3] - df.iloc[:, 2])).median()

def print_diff_iv_by_tenor_right_ts_bucket():
    """Essentially fill forward iv along ts -
    remove where either is 0 or nan
     subtract mid ivs from each other.
     """
    start = date(2024, 2, 22)
    end = date(2024, 2, 22)
    resolution = Resolution.second
    security_type = SecurityType.option
    client = Client()
    sym = 'FDX'
    equity = Equity(sym)
    # zip_path = r'D:\trade\data\option\usa\second\fdx\20240222_quote_american_iv.zip'
    tick_type = TickType.iv_quote
    dummy_contract = OptionContract(str(equity), str(equity), None, None, None)
    zip_paths = client.get_zip_paths([dummy_contract], tick_type, resolution, security_type, start, end)

    zip_path = zip_paths[0]
    dt = datetime.strptime(zip_path.name[:8], "%Y%m%d").date()
    csv_names: List[str] = client.get_csv_names_in_zip(zip_path)
    csvNmSymDts = [CsvNmSymDt(nm, OptionContract.from_filename(nm), dt) for nm in csv_names]

    csv_nm_df = read_option_csvs_from_zip(zip_path, csvNmSymDts, tick_type)

    stats_lst = []
    for csv_nm, df in csv_nm_df.items():
        contract = OptionContract.from_filename(csv_nm)
        calc_date = df.index[0].date()
        tenor = get_tenor(contract.expiry, calc_date)
        # df = df[(df['ask_iv'] > 0) & (df['bid_iv'] > 0)]

        stats_lst.append(StatsDiff(tenor, contract.right, contract.strike, df))

    # stats_lst2 = [el for el in stats_lst if el.bid_iv > 0 and el.ask_iv > 0]
    df = pd.DataFrame(stats_lst).groupby(['tenor', 'strike']).apply(magic)#.sort_values(['tenor', 'strike'])
    heatmap_from_multiindex_series(df)

def heatmap_from_multiindex_series(s: pd.Series, title: str = "Heatmap"):
    # Ensure the index levels are named / accessible
    if s.index.nlevels != 2:
        raise ValueError("Expected a 2-level MultiIndex: (tenor, strike).")

    z = (
        s.rename("value")
         .reset_index()
         .pivot(index=s.index.names[0], columns=s.index.names[1], values="value")
         .sort_index(axis=0)
         .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(z.to_numpy(), origin="lower", aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(z.columns.name or "strike")
    ax.set_ylabel(z.index.name or "tenor")

    # Tick labels (downsample if huge)
    ax.set_xticks(range(len(z.columns)))
    ax.set_xticklabels(z.columns, rotation=90)
    ax.set_yticks(range(len(z.index)))
    ax.set_yticklabels(z.index)

    fig.colorbar(im, ax=ax, label="value")
    fig.tight_layout()
    plt.show()

    return z  # return the pivoted grid in case you want it

if __name__ == '__main__':
    # print_avg_iv_by_tenor_right()
    print_diff_iv_by_tenor_right_ts_bucket()