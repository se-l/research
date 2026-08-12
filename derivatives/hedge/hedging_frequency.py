import datetime
import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import List
from matplotlib import pyplot as plt
from functools import partial
from derivatives.helper import historical_volatility
from derivatives.typess.enums import Resolution
from derivatives.typess.equity import Equity
from derivatives.typess.option_frame import OptionFrame


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
    non_gap_total: float
    score: float
    gamma: float
    delta_threshold: float


def find_optimal_equity_hedging_frequency(
        df_equity: pd.DataFrame,
        gamma: float = 200,
        delta_threshold: float = 25,
        transaction_cost_fixed=1,
        transaction_cost_proportional=0,
):
    """
    Per epoch, assume a constant gamma. Loop through delta return frame. Hedge every time x delta threshold is crossed.
    So grid search of gamma, delta threshold.

    :return:
    """
    print(f'find_optimal_equity_hedging_frequency: rows={len(df_equity)}, gamma={gamma}, delta_threshold={delta_threshold}')
    df_equity.loc[:, 'dDelta'] = df_equity['dS'] * gamma

    hedges: List[Hedge] = []
    delta_pf = 0
    pnl_snaps: List[PnLSnap] = []

    eod_ts = list({dt: v[-1] for dt, v in df_equity.index.groupby(df_equity.index.date).items()}.values())

    for tup in df_equity.itertuples():
        pnl_delta0 = tup.dS * delta_pf
        pnl_delta1 = tup.dS * tup.dDelta / 2
        pnl_snaps.append(PnLSnap(tup.Index, tup.spot, pnl_delta0 + pnl_delta1, delta_pf))
        delta_pf += tup.dDelta

        if abs(delta_pf) >= delta_threshold or tup.Index in eod_ts:
            quantity = -int(delta_pf)
            hedges.append(Hedge(tup.Index, quantity, tup.spot, -transaction_cost_fixed + -transaction_cost_proportional * abs(quantity)))
            delta_pf += quantity
    return pnl_snaps, hedges


def score(pnl_snaps, hedges):
    df_hedge = pd.DataFrame(hedges).set_index('ts') if hedges else np.array([0])
    df_pnl = pd.DataFrame(pnl_snaps, columns=['ts', 'pnl']).set_index('ts') if pnl_snaps else np.array([0])

    df = df_pnl.merge(df_hedge, left_index=True, right_index=True, how='left')
    df['fee'] = df['fee'].fillna(0)
    df['total'] = df['pnl'] + df['fee']

    ps_total = df['total'][~np.isinf(df['total'])]
    # Only consider intra-day returns. Hedgeable returns.
    ps_total = ps_total[ps_total.index.time > datetime.time(9, 32)]
    _score = ps_total.sum() * ps_total.std()

    return ps_total.sum(), _score


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
    pnl_snaps, hedges = find_optimal_equity_hedging_frequency(df_equity, gamma, delta_threshold)

    fee = pd.DataFrame(hedges)['fee'].sum() if hedges else 0
    pnl = pd.DataFrame(pnl_snaps, columns=['ts', 'pnl'])['pnl'] if pnl_snaps else 0

    non_gap_total, _score = score(pnl_snaps, hedges)
    return EpochResult(pnl, fee, non_gap_total, _score, gamma, delta_threshold)


def grid_search_mp(v_gamma, v_delta_threshold, df_equity: pd.DataFrame, mp=False) -> List[EpochResult]:
    print(f'Simulating {len(v_gamma) * len(v_delta_threshold)} combinations.')
    f = partial(epoch, df_equity=df_equity)
    combinations = ((gamma, delta_threshold) for gamma in v_gamma for delta_threshold in v_delta_threshold)
    if mp:
        with mp.Pool(8) as pool:
            z = pool.map(f, combinations)
    else:
        z = list(map(f, combinations))
    return z


if __name__ == '__main__':
    start = datetime.date(2023, 9, 1)
    end = datetime.date(2024, 2, 1)
    sym = 'orcl'
    equity = Equity(sym)
    resolution = Resolution.second
    seq_ret_threshold = 0.001

    option_frame = OptionFrame.load_frame(equity, resolution, seq_ret_threshold, '1')

    hv_cols = ['hv', 'hv_1d']
    df_equity = option_frame.df_equity
    df_equity = setup_df_equity_frame(df_equity)
    # equity: Equity, resolution: Resolution, seq_ret_threshold: float

    # # Single run
    if False:
        results = []
        plotted_spot = True
        gamma = -100
        # gamma = -5
        for delta_threshold in np.arange(15, 100, 5):
        # for delta_threshold in [50]:
            pnl_snaps, hedges = find_optimal_equity_hedging_frequency(df_equity, gamma, delta_threshold, transaction_cost_fixed=1)
            non_gap_total, _score = score(pnl_snaps, hedges)

            df_pnl = pd.DataFrame(pnl_snaps).set_index('ts')
            df_hedge = pd.DataFrame(hedges).set_index('ts')
            pnl = df_pnl['pnl'].sum()
            fee = df_hedge['fee'].sum()

            df = df_pnl.merge(df_hedge, left_index=True, right_index=True, how='left')
            df['fee'] = df['fee'].fillna(0)
            df['total'] = df['pnl'] + df['fee']

            # rolling score
            df['score'] = None

            # dont do std on returns. too noisy given tiny cent changes.
            # ps_return = (df['total'] / df['total'].shift(1) - 1).dropna()
            ps_total = df['total'][~np.isinf(df['total'])]

            # Only consider intra-day returns. Hedgeable returns.
            ps_total = ps_total[ps_total.index.time > datetime.time(9, 32)]

            _score = ps_total.std()

            # rolling
            rolling_std = ps_total.rolling(len(ps_total), min_periods=0).std().dropna()
            df.loc[rolling_std.index, 'score'] = rolling_std

            print(f'gamma={gamma}, delta_threshold={delta_threshold}, total={pnl + fee}, pnl={pnl}, fee={fee}, score={_score}, mean_ret_std={df["score"].mean()}'
                  f' # hedges={len(hedges)}, non-gap total={ps_total.sum()} total*std={ps_total.sum() * _score}')

            if not plotted_spot:
                ax1 = df[['spot']].plot()
                ax1.set_title(f'Spot; gamma={gamma}, delta_threshold={delta_threshold}')
                plotted_spot = True

            ax2 = df[['score']].plot()
            ax2.set_title(f'Score; gamma={gamma}, delta_threshold={delta_threshold}')

            # ax = df[['pnl', 'fee', 'total']].cumsum().plot()
            ax = ps_total.cumsum().plot()
            ax.set_title(f'gamma={gamma}, delta_threshold={delta_threshold}')

    # Zero transaction cost and minimal delta threshold do not yield the best score, because hedges get whipsawed.
    # Therefore, is the ideal hedge frequency dependent on the underlying's volatility?

    # Breakeven gamma / theta ratio? Whenever IV > fwd realized vol, theta should earn more... provided no drift?! If drift, more hedging effort, trickier...

    # Gap trades / jumps where I cannot hedge add most risk. Best safety is being zero delta, gamma, speed at EOD. Insure against those with wing derivatives...

    # Grid search
    if True:
        v_gamma = np.linspace(-500, -25, 20)
        v_delta_threshold = np.arange(1, 100, 5)
        results = grid_search_mp(v_gamma, v_delta_threshold, df_equity, False)

        Z = pd.DataFrame(results).groupby(['gamma', 'delta_threshold'])['score'].aggregate('first').unstack().sort_index()
        # Zfee = pd.DataFrame(results).groupby(['gamma', 'delta_threshold'])['fee'].aggregate('first').apply(lambda ps: ps.sum()).unstack().sort_index()
        Zpnl = pd.DataFrame(results).groupby(['gamma', 'delta_threshold'])['pnl'].aggregate('first').apply(lambda ps: ps.sum()).unstack().sort_index()
        Znon_gap_total = pd.DataFrame(results).groupby(['gamma', 'delta_threshold'])['non_gap_total'].aggregate('first').unstack().sort_index()
        X, Y = np.meshgrid(v_delta_threshold, v_gamma)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z.values, rstride=1, cstride=1, linewidth=0.1)
        ax.set_title('Total Score')
        ax.set_xlabel('Delta Threshold')
        ax.set_ylabel('Gamma')
        # fig.colorbar(surf, shrink=0.5, aspect=5)

        fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"})
        surf3 = ax3.plot_surface(X, Y, Znon_gap_total.values, rstride=1, cstride=1, linewidth=0.1)
        ax3.set_title('Total Fee')
        ax3.set_xlabel('Delta Threshold')
        ax3.set_ylabel('Gamma')

        figPnl, axPnl = plt.subplots(subplot_kw={"projection": "3d"})
        surfPnl = axPnl.plot_surface(X, Y, Zpnl.values, rstride=1, cstride=1, linewidth=0.1)
        axPnl.set_title('PnL')
        axPnl.set_xlabel('Delta Threshold')
        axPnl.set_ylabel('Gamma')

        # fig4, ax4 = plt.subplots(subplot_kw={"projection": "3d"})
        # surf4 = ax4.plot_surface(X, Y, Zpnl.values, rstride=1, cstride=1, linewidth=0.1)
        # ax4.set_title('Total PnL')
        # ax4.set_xlabel('Delta Threshold')
        # ax4.set_ylabel('Gamma')

        best_score_gamma_threshold = {}
        for gamma in v_gamma:
            best_score_gamma_threshold[gamma] = Z.columns[Z.loc[gamma].argmax()]
        fig4, ax4 = plt.subplots()
        ps_gamma_thresh = pd.Series(best_score_gamma_threshold)
        ps_gamma_thresh.plot(ax=ax4)

        from sklearn.linear_model import LinearRegression
        x = ps_gamma_thresh.index.to_numpy().reshape(-1, 1)
        y = ps_gamma_thresh.values#.reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        r2 = reg.score(x, y)
        print(f'''R2: {r2}, Intercept: {reg.intercept_}, Coef: {reg.coef_}''')

        threshold_predict = pd.Series(reg.predict(x), index=v_gamma)
        threshold_predict.plot(ax=ax4)
    pass