import datetime
import pandas as pd
import numpy as np
import multiprocessing as mp

from dataclasses import dataclass
from typing import List
from functools import partial
from options.helper import historical_volatility
from options.types.enums import Resolution
from options.types.equity import Equity
from options.types.option_frame import OptionFrame


@dataclass
class Hedge:
    ts: datetime.datetime
    quantity: float
    price_fill: float
    fee: float = 1


@dataclass
class PnLSnap:
    ts: datetime.datetime
    spot: float
    pnl: float
    delta: float


@dataclass
class EpochResult:
    pnl: float
    fee: float
    total: float
    gamma: float
    delta_threshold: float


def find_optimal_gamma_scalping_trailing_stop(
        df_equity: pd.DataFrame,
        trailing_stop: float,
        gamma: float = 200,
        delta_threshold: float = 5,
        transaction_cost_fixed=1,
        transaction_cost_proportional=0,
):
    """
    Per epoch, assume a constant gamma. Loop through delta return frame. Hedge every time x delta threshold is crossed.
    So grid search of gamma, delta threshold.

    :return:
    """
    print(f'find_optimal_equity_hedging_frequency: rows={len(df_equity)}, trailing_stop={trailing_stop}, gamma={gamma}, delta_threshold={delta_threshold}')
    df_equity.loc[:, 'dDelta'] = df_equity['dS'] * gamma

    hedges: List[Hedge] = []
    delta_pf = 0
    pnl_snaps: List[PnLSnap] = []
    scalping = False
    trailing_stop_price = 0

    for tup in df_equity.itertuples():
        pnl_delta0 = tup.dS * delta_pf
        pnl_delta1 = tup.dS * tup.dDelta / 2
        pnl_snaps.append(PnLSnap(tup.Index, tup.spot, pnl_delta0 + pnl_delta1, delta_pf))
        delta_pf += tup.dDelta

        if scalping:
            if (
                (delta_pf >  delta_threshold and tup.spot < trailing_stop_price) or
                (delta_pf < -delta_threshold and tup.spot > trailing_stop_price)
            ):
                quantity = -int(delta_pf)
                hedges.append(Hedge(tup.Index, quantity, tup.spot, -transaction_cost_fixed - transaction_cost_proportional * abs(quantity)))
                delta_pf += quantity
                scalping = False
            elif delta_pf > delta_threshold:
                trailing_stop_price = max(trailing_stop_price, tup.spot * (1 - trailing_stop))
            elif delta_pf < -delta_threshold:
                trailing_stop_price = min(trailing_stop_price, tup.spot * (1 + trailing_stop))
            else:
                trailing_stop_price = 0
                scalping = False

        else:
            if delta_pf >= delta_threshold:
                trailing_stop_price = tup.spot * (1 - trailing_stop)
                scalping = True
            elif delta_pf <= -delta_threshold:
                trailing_stop_price = tup.spot * (1 + trailing_stop)
                scalping = True

    return pnl_snaps, hedges


def score(pnl_snaps, hedges):
    fee = pd.DataFrame(hedges)['fee'].sum() if hedges else 0
    pnl = pd.DataFrame(pnl_snaps, columns=['ts', 'pnl'])['pnl'].sum() if pnl_snaps else 0
    return pnl, fee, pnl + fee


def setup_df_equity_frame(df_equity: pd.DataFrame):
    df_equity.sort_index(level='ts', inplace=True)
    df_equity['hv'] = historical_volatility(df_equity['close'])
    df_equity['hv_1d'] = historical_volatility(df_equity['close'], window=pd.Timedelta(days=1))

    df_equity['spot'] = df_equity['close']
    df_equity['dS_pct'] = df_equity['spot'] / df_equity['spot'].shift(1) - 1
    df_equity['dS'] = df_equity['spot'] - df_equity['spot'].shift(1)
    df_equity['dDelta'] = None
    return df_equity.iloc[1:]


def epoch(tup, df_equity) -> EpochResult:
    gamma, delta_threshold = tup
    pnl_snaps, hedges = find_optimal_gamma_scalping_trailing_stop(df_equity, gamma, delta_threshold)
    pnl, fee, total = score(pnl_snaps, hedges)
    return EpochResult(pnl, fee, total, gamma, delta_threshold)


def grid_search_mp(v_gamma, v_delta_threshold, df_equity: pd.DataFrame) -> List[EpochResult]:
    print(f'Simulating {len(v_gamma) * len(v_delta_threshold)} combinations.')
    f = partial(epoch, df_equity=df_equity)
    combinations = ((gamma, delta_threshold) for gamma in v_gamma for delta_threshold in v_delta_threshold)
    with mp.Pool(8) as pool:
        z = pool.map(f, combinations)
    return z


if __name__ == '__main__':
    start = datetime.date(2023, 9, 1)
    end = datetime.date(2024, 2, 1)
    sym = 'csco'
    equity = Equity(sym)
    resolution = Resolution.second
    seq_ret_threshold = 0.001

    option_frame = OptionFrame.load_frame(equity, resolution, seq_ret_threshold, '1')

    hv_cols = ['hv', 'hv_1d']
    df_equity = option_frame.df_equity
    df_equity = setup_df_equity_frame(df_equity)
    # equity: Equity, resolution: Resolution, seq_ret_threshold: float

    # # Single run
    # for trailing_stop in np.linspace(0.01, 0.1, 10):
    gamma = 100
    delta_threshold = 5
    plotted_spot = False
    for trailing_stop in np.arange(0.002, 0.007, 0.0005):
        pnl_snaps, hedges = find_optimal_gamma_scalping_trailing_stop(df_equity, trailing_stop, gamma, delta_threshold)
        pnl, fee, total = score(pnl_snaps, hedges)
        print(f'trailing_stop={trailing_stop}, pnl={pnl}, fee={fee}, score={total}, # hedges={len(hedges)}')
        df_pnl = pd.DataFrame(pnl_snaps).set_index('ts')
        if not hedges:
            continue
        df_hedge = pd.DataFrame(hedges).set_index('ts')
        df = df_pnl.merge(df_hedge, left_index=True, right_index=True, how='outer').fillna(0)
        df['total'] = df['pnl'] + df['fee']

        if not plotted_spot:
            ax1 = df[['spot']].plot()
            ax1.set_title(f'Spot; trailing_stop={trailing_stop}')
            plotted_spot = True

        ax = df[['pnl', 'fee', 'total']].cumsum().plot()
        ax.set_title(f'trailing_stop={trailing_stop}')
    pass
