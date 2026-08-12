from itertools import chain
import numpy as np
import pandas as pd

from decimal import Decimal
from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Tuple
from scipy.stats import chi2, norm
from pykalman import KalmanFilter
from collections import deque

from shared.constants import EarningsPreSessionDates, DiscountRateMarket
from options.helper import get_dividend_yield, get_moneyness_fwd, timer
from options.types.enums import Resolution
from options.types.equity import Equity
from shared.plotting import show
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from plotly import graph_objects as go
from pprint import pprint

y_col_nm = 'fill_iv'
kalman_mean_col = 'fill_iv_kalman_mean'
kalman_up_col = 'fill_iv_kalman_up'
kalman_low_col = 'fill_iv_kalman_low'
kalman_my_ask_col = 'fill_iv_kalman_my_ask'
kalman_my_bid_col = 'fill_iv_kalman_my_bid'
# kalman_my_ask_price_col = 'fill_iv_kalman_my_ask_price'
# kalman_my_bid_price_col = 'fill_iv_kalman_my_bid_price'


def a_bx_cx2_dx3(x, a, b, c, d):
    return a + b * x + c * x ** 2 + d * x ** 3


@dataclass
class FitFillIVResult:
    expiry: date
    right: str
    popt: tuple
    pcov: np.ndarray


@dataclass
class TrainKalmanResult:
    expiry: date
    right: str
    popt: tuple
    pcov: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    strikes_train: List[Decimal]


def fit_fill_iv(df: pd.DataFrame,
                x_col_nm='moneyness_fwd_ln',
                y_col_nm='fill_iv',
                scoped_moneyness=(0.6, 1.4),
                bounds=((0, 0, 0, -np.inf), (np.inf, np.inf, np.inf, np.inf)),
                plot=False
                ) -> List[FitFillIVResult]:
    """
    Regressing the central moment of the IV surface. This is the skew of the IV surface.
    Applying more weight to ATM IV, IV more uncertain the further OTM/ITM it is. That'll be a sigma array, where sigma~abs(1-moneyness) or sigma~abs(ln(moneyness))
    Better weight: vega!
    Enforcing a smirk/smile like shape by setting bounds on the regression coefficients.
    Saving each regression coefficient to the frame.
    """
    v_strike = df.index.get_level_values('strike').astype(float).values
    df['moneyness'] = v_strike / df['spot']
    if 'moneyness_fwd' not in df.columns:
        df['moneyness_fwd'] = get_moneyness_fwd(option_frame.equity, v_strike, df['spot'].astype(float).values, df['tenor'].values, t0=df['date'].values)

    df_scoped = df[
        (df['moneyness_fwd'] > scoped_moneyness[0]) & (df['moneyness_fwd'] < scoped_moneyness[1])
        # & (df.index.get_level_values('ts').date == date(2024, 2, 5))
    ]

    expiries = df_scoped.index.get_level_values('expiry').unique()

    if plot:
        fig = make_subplots(rows=len(expiries)*2+2, cols=2)
        r2s = []

    res = []
    for k, (expiry, s_df) in enumerate(df_scoped.groupby(level='expiry')):
        for l, (right, ss_df) in enumerate(s_df.groupby(level='right')):
            if (~ss_df[y_col_nm].isna()).sum() == 0:
                continue

            ix_compatible = ss_df[y_col_nm].reset_index().index[ss_df[y_col_nm].notna()]

            # More weight to ATM IV, IV more uncertain the further OTM/ITM it is. That'll be a sigma array, where sigma~abs(1-moneyness_fwd) or sigma~abs(ln(moneyness_fwd))
            sigma = np.exp(((np.abs(ss_df['moneyness_fwd'] - 1) + 1) ** 2).astype(float).values[ix_compatible])
            popt, pcov = curve_fit(a_bx_cx2_dx3, ss_df[x_col_nm].values[ix_compatible], ss_df[y_col_nm].values[ix_compatible], sigma=sigma, bounds=bounds)
            res.append(FitFillIVResult(expiry, right, popt, pcov))

            if plot:
                ix_scope = ss_df[(ss_df['moneyness_fwd'] > scoped_moneyness[0]) & (ss_df['moneyness_fwd'] < scoped_moneyness[1])].index
                y_pred = a_bx_cx2_dx3(ss_df.loc[ix_scope, x_col_nm], *popt)
                r2 = r2_score(ss_df.loc[ix_scope, y_col_nm], y_pred)
                r2s.append(r2)
                print(f'R2 Quadratic Regr. {right} {expiry}: {r2}')
                fig.add_trace(go.Scatter(x=ss_df['moneyness'].values, y=ss_df[y_col_nm].values, mode='markers', name=f'{y_col_nm} {expiry} {right}', marker=dict(size=2)), row=k*2 + 1, col=l+1)
                yv = y_pred.astype(float).values
                fig.add_trace(go.Scatter(x=ss_df['moneyness'].values, y=yv, mode='markers', name=f'Regressed {y_col_nm} {expiry} {right}', marker=dict(size=3)), row=k*2+2, col=l+1)

    if plot:
        print(f'Mean R2: {np.mean(r2s)}')
        fig.update_layout(title_text=f'Regressed {y_col_nm}')
        show(fig, 'fill_iv_regression.html')
        print(pd.DataFrame(res))

    return res


def strike2moneyness_fwd_ln(strikes: List[float], s, net_yield: float, tenor) -> np.ndarray:
    _s = s.values if isinstance(s, pd.Series) else s
    _tenor = tenor.values if isinstance(tenor, pd.Series) else tenor

    fwd = _s * np.exp(net_yield * _tenor)
    return np.log(strikes / fwd)


def get_confidence_bounds(y_true: np.ndarray, var: np.ndarray, alpha=0.95, one_sided=True):
    if False:
        # Resulting bounds are way way too large by magnitudes.
        # converting to scale might be off. Just taking square root is taken from ellipsis example. may not be applicable to higher dimensions.
        # https://www.kalmanfilter.net/background2.html
        # chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://users.cs.utah.edu/~tch/CS4640/resources/A%20geometric%20interpretation%20of%20the%20covariance%20matrix.pdf

        print(smoothed_state_means[-1])
        eigVal, eigVec = np.linalg.eig(smoothed_state_covariances[-1])  # find eigenvalues and eigenvectors
        # theta = np.arctan(eigVec[1, 0] / eigVec[0, 0])  # for ellipse; k=2

        df = degress_freedom
        alpha = 0.65
        scale = np.sqrt(chi2.ppf(alpha, df))

        state_upper = smoothed_state_means[-1] + scale * np.sqrt(eigVal)
        state_lower = smoothed_state_means[-1] - scale * np.sqrt(eigVal)

        # # ellipsis
        # df = 2
        # alpha = chi2.cdf(1, df)  # 0.3935
        # alpha = 0.95
        # scale = np.sqrt(chi2.ppf(alpha, df))
        # print(scale)  # agrees with https://www.kalmanfilter.net/background2.html

        # Compute upper and lower confidence intervals
        ivs_upper = np.vstack([a_bx_cx2_dx3(v_m, *state_upper) for i, v_m in enumerate(v_v_m)])
        ivs_lower = np.vstack([a_bx_cx2_dx3(v_m, *state_lower) for i, v_m in enumerate(v_v_m)])

        # ivs_upper = np.vstack([a_bx_cx2_dx3(v_m, *state_upper) for i, v_m in enumerate(v_v_m)])
        # ivs_lower = np.vstack([a_bx_cx2_dx3(v_m, *state_lower) for i, v_m in enumerate(v_v_m)])
    else:
        if one_sided:
            lb, ub = -norm.ppf(alpha), norm.ppf(alpha)
        else:
            lb, ub = norm.interval(alpha)

        return y_true + lb * np.sqrt(var), y_true + ub * np.sqrt(var)


def init_measurement(df, strikes_trained: List[Decimal], X, net_yield):
    z = np.empty(len(strikes_trained))
    for i, strike in enumerate(strikes_trained):
        try:
            z[i] = df.loc[(slice(None), strike), y_col_nm].values[-1]
        except Exception as e:
            z[i] = a_bx_cx2_dx3(strike2moneyness_fwd_ln([float(strike)], df['spot'], net_yield, df['tenor'])[-1], *X)
        if np.isnan(z[i]):
            z[i] = a_bx_cx2_dx3(strike2moneyness_fwd_ln([float(strike)], df['spot'], net_yield, df['tenor'])[-1], *X)
    return z


def update_measurement(measurement: np.ndarray, sss_df: pd.DataFrame, strikes_measurement: List[Decimal], y_col_nm: str, v_m: np.ndarray, smoothed_state_means: np.ndarray):
    strikes_new = [s for s in sss_df.index.get_level_values('strike') if s in strikes_measurement]
    values_new = sss_df.loc[(slice(None), slice(None), strikes_new, slice(None)), y_col_nm].values
    # New Fill IVs are used as it. Genuine new event, measurement.
    ix_new = [strikes_measurement.index(s) for s in strikes_new]
    measurement[ix_new] = values_new

    # Remaining are not simply fill forwarded, but regressed with the existing model.
    ix_regressed = sorted(set(range(len(measurement))) - set(ix_new))

    measurement[ix_regressed] = a_bx_cx2_dx3(v_m, *smoothed_state_means)[ix_regressed]
    return measurement


@dataclass
class KalmanState:
    expiry: date
    right: str
    ts: pd.Timestamp
    z: np.ndarray  # observation / measurement
    X: np.ndarray  # state_means
    P: np.ndarray  # state covariance
    H: np.ndarray  # observation matrix
    # residual = v_z[i] - np.dot(observation_matrices[i], a)
    # print(f'i={i}, residual={residual.sum()}, residual_iv={sum(v_z[i] - a_bx_cx2_dx3(v_v_m[i], *a))}, state={a}, iv_calc={a_bx_cx2_dx3(v_v_m[i], *a)}, y_i={v_z[i]}')


def train_kalman_initial_state(df: pd.DataFrame, net_yield, fit_moneyness=(0.8, 1.2), n_iter=100, plot=False) -> List[TrainKalmanResult]:
    res = []
    # Fitting initial state
    fitTradeIVResult = fit_fill_iv(df, scoped_moneyness=fit_moneyness, plot=plot)
    print(f'Trained init state on with {len(fitTradeIVResult)} results')

    # train R, Q on release_date_m1. For each expiry, right.
    n_dim_state = 4
    s0 = df['spot'].iloc[0]
    strikes_train = [s for s in df.index.get_level_values('strike').unique() if fit_moneyness[0] - 1 < abs(float(s) / s0 - 1) < fit_moneyness[1] - 1]
    strikes_train_flt = [float(s) for s in strikes_train]
    df = df.loc[(slice(None), slice(None), strikes_train, slice(None))]

    expiries = list(sorted(df.index.get_level_values('expiry').unique()))

    for i, (expiry, s_df) in enumerate(df.groupby('expiry')):
        if expiry not in expiries:
            continue

        for j, (right, ss_df) in enumerate(s_df.groupby('right')):
            init_state = next(iter([r for r in fitTradeIVResult if r.expiry == expiry and r.right == right]), None)
            if not init_state:
                print(f'No initial state for {expiry} {right}')
                continue

            X = init_state.popt
            P = init_state.pcov

            strike_intersect = sorted(df.loc[(slice(None), expiry, slice(None), right)].index.get_level_values('strike').intersection(strikes_train))
            if len(strike_intersect) == 0:
                print(f'No intersecting strikes for {expiry} {right}')
                continue
            z = init_measurement(df.loc[(slice(None), expiry, strike_intersect, right)], strikes_train, X, net_yield)

            v_z = []
            observation_matrices = []
            for k, (ts, sss_df) in enumerate(ss_df.groupby('ts')):
                v_m = strike2moneyness_fwd_ln(strikes_train_flt, sss_df['spot'].iloc[0], net_yield, sss_df['tenor'].iloc[0])
                z = update_measurement(z, sss_df, strikes_train, y_col_nm, v_m, X)
                v_z.append(z)
                observation_matrices.append(np.array([[1, m, m ** 2, m ** 3] for m in v_m]))

            kf = KalmanFilter(
                n_dim_state=n_dim_state, n_dim_obs=len(z),
                transition_matrices=np.eye(n_dim_state),  # F
                observation_matrices=observation_matrices,
                initial_state_mean=X,
                initial_state_covariance=P,
                em_vars=['transition_covariance', 'observation_covariance'],
            )
            try:
                kf.em(v_z, n_iter=n_iter, em_vars=['transition_covariance', 'observation_covariance'])
            except Exception as e:
                raise e

            res.append(TrainKalmanResult(expiry, right, X, P, kf.transition_covariance, kf.observation_covariance, strikes_train))

    return res


@timer
def fit_kalman_states(df_trades: pd.DataFrame, release_date: date, net_yield, fit_moneyness=(0.8, 1.2), plot=False, scoped_expiries=None) -> Dict[Tuple[date, str], List[KalmanState]]:
    release_date_m1 = max([el for el in set(df_trades.index.get_level_values('ts').date) if el < release_date])
    df_train = df_trades[(df_trades.index.get_level_values('ts').date == release_date_m1)]
    train_kalman_results = train_kalman_initial_state(df_train, net_yield, fit_moneyness, plot)

    # Running Kalman Filter on release date
    df_fit = df_trades[(df_trades.index.get_level_values('ts').date > release_date_m1)]

    expiries = list(sorted(df_fit.index.get_level_values('expiry').unique()))
    scoped_expiries = scoped_expiries or expiries

    kalman_states = {}

    # i, (expiry, s_df) = next(iter(enumerate(df_fit.groupby('expiry'))))
    # j, (right, ss_df) = next(iter(enumerate(s_df.groupby('right'))))
    # k, (ts, sss_df) = next(iter(enumerate(ss_df.groupby('ts'))))
    for i, (expiry, s_df) in enumerate(df_fit.groupby('expiry')):
        if expiry not in scoped_expiries:
            continue

        for j, (right, ss_df) in enumerate(s_df.groupby('right')):
            # May need a rolling observations vector. Drop strikes if no trade/measurement in over an hour. Construct in a rolling fashion and
            # update kalman objects from previous instance (cannot, different dimensions). Could impute with bid/ask potentially.
            # Or drop the respective row from existing kalman matrices before inserting in new kf. Last best, but convoluted.

            init_state = next(iter([r for r in train_kalman_results if r.expiry == expiry and r.right == right]), None)
            if not init_state:
                print(f'No initial state for {expiry} {right}')
                continue

            v_kalman_states = []

            n_dim_state = 4
            F = np.eye(n_dim_state)
            X = init_state.popt
            P = init_state.pcov
            P = P if not np.any(np.isinf(P)) and not np.any(np.isnan(P)) else np.full(P.shape, 0.0004)
            Q = init_state.Q
            # Q = np.eye(n_dim_state),
            R = init_state.R
            # R = None
            strikes_trained = init_state.strikes_train
            strikes_trained_flt = [float(s) for s in strikes_trained]
            # Need to kick the i, j from R whose strikes are not in this dataframe here...
            # Further, bad reliance on strikes. We care for moneyness...

            strike_intersect = sorted(df_train.loc[(slice(None), expiry, slice(None), right)].index.get_level_values('strike').intersection(strikes_trained))
            if len(strike_intersect) == 0:
                print(f'No intersecting strikes for {expiry} {right}')
                continue
            z = init_measurement(df_train.loc[(slice(None), expiry, strike_intersect, right)], strikes_trained, X, net_yield)
            kf = KalmanFilter(
                n_dim_state=n_dim_state, n_dim_obs=len(z),
                transition_matrices=F,
                transition_covariance=Q,
                observation_covariance=R,
                initial_state_mean=X,
                initial_state_covariance=P,
                em_vars=[],
            )
            # print(Q, R)

            for k, (ts, sss_df) in enumerate(ss_df.groupby('ts')):
                v_m = strike2moneyness_fwd_ln(strikes_trained_flt, sss_df['spot'].iloc[0], net_yield, sss_df['tenor'].iloc[0])
                z = update_measurement(z, sss_df, strikes_trained, y_col_nm, v_m, X)

                H = np.array([[1, m, m ** 2, m ** 3] for m in v_m])
                X, P = kf.filter_update(X, P, observation=z, observation_matrix=H)
                # X2, P2 = kf.filter_update(X, P, observation=z, observation_matrix=H)
                # x_mine, p_mine = kf_update(P, H, z, X, Q, R)
                # if False:
                #     assert np.allclose(x_mine, X2)
                #     assert np.allclose(p_mine, P2)
                # X = X2
                # P = P2

                v_kalman_states.append(KalmanState(expiry, right, ts, z, X, P, H))

            kalman_states[(expiry, right)] = v_kalman_states

    return kalman_states


def kf_update(P, H, z, X, Q, R):
    # Prediction
    X1 = X
    P1 = P + Q
    # Update
    S = np.linalg.inv((H @ P1 @ H.T) + R)
    K = P1 @ H.T @ S

    residual = z - H@X

    X2 = X1 + K @ residual
    # I = np.eye(P.shape[0])
    # P_mine = (I - K@H) @ P  # @ (I - K@H)
    P2 = P1 - K @ H @ P1  # @ (I - K@H)
    return X2, P2


def enrich_kalman(
        df_trades: pd.DataFrame,
        kalman_states: Dict[Tuple[date, str], List[KalmanState]],
        net_yield,
        alpha_one_sided=0.9,  # So I'd get filled on remaining 10% of the trades
        ):
    """slow enrichment and handling of the error to calc variance not great. Is window logic now."""
    for i, (expiry, s_df) in enumerate(df_trades.groupby('expiry')):
        for j, (right, ss_df) in enumerate(s_df.groupby('right')):
            v_kalman_states = kalman_states.get((expiry, right))
            if not v_kalman_states:
                continue
            ts2kalman_states = {ks.ts: ks for ks in v_kalman_states}

            v_strike = sorted(ss_df.index.get_level_values('strike').unique())
            dct_strike_err_sq = {s: deque(maxlen=30) for s in v_strike}

            # Super low filling
            for i, (ts, sss_df) in enumerate(ss_df.groupby('ts')):
                kalman_state = ts2kalman_states.get(ts)
                if not kalman_state:
                    continue

                v_strike_sss = sss_df.index.get_level_values('strike')
                v_m = strike2moneyness_fwd_ln(list(sss_df['strike_flt']), sss_df['spot'], net_yield, sss_df['tenor'])
                z_smoothed = np.array(a_bx_cx2_dx3(v_m, *kalman_state.X))
                err = sss_df[y_col_nm].values - z_smoothed
                for i, strike in enumerate(sss_df.index.get_level_values('strike')):
                    dct_strike_err_sq[strike].append(err[i] ** 2)

                var = np.array([sum(dct_strike_err_sq[s]) / len(dct_strike_err_sq[s]) if len(dct_strike_err_sq[s]) > 0 else 0 for s in v_strike_sss])
                v_lower_ivs, v_upper_ivs = get_confidence_bounds(z_smoothed, var, 0.95, one_sided=False)
                v_my_bid_ivs, v_my_ask_ivs = get_confidence_bounds(z_smoothed, var, alpha_one_sided, one_sided=True)

                df_trades.loc[(ts, expiry, slice(None), right), kalman_mean_col] = z_smoothed

                df_trades.loc[(ts, expiry, slice(None), right), kalman_low_col] = v_lower_ivs
                df_trades.loc[(ts, expiry, slice(None), right), kalman_up_col] = v_upper_ivs

                df_trades.loc[(ts, expiry, slice(None), right), kalman_my_bid_col] = v_my_bid_ivs
                df_trades.loc[(ts, expiry, slice(None), right), kalman_my_ask_col] = v_my_ask_ivs

    return df_trades


def kalman_error_report(df_trades: pd.DataFrame):
    """
    Measure total and rolling MAE, RMSE between trade IV fill and mean Kalman IV fill.
    Check where error is the largest.

    Most important for trading is where I give the greatest discount, ie, where bid / ask are furthest away from their respective NBBO.
    The respective boundaries are given by mean and std. So minimizing mean error is the first step!
    """
    df = df_trades[df_trades[kalman_mean_col].notna()].copy()
    mae_total = mean_absolute_error(df[y_col_nm], df[kalman_mean_col])
    rmse_total = mean_squared_error(df[y_col_nm], df[kalman_mean_col], squared=False)
    print(f'Total MAE: {mae_total} RMSE: {rmse_total}')

    for i, (expiry, s_df) in enumerate(df.groupby('expiry')):
        for j, (right, ss_df) in enumerate(s_df.groupby('right')):
            mae = mean_absolute_error(ss_df[y_col_nm], ss_df[kalman_mean_col])
            rmse = mean_squared_error(ss_df[y_col_nm], ss_df[kalman_mean_col], squared=False)
            print(f'{expiry} {right} MAE: {mae} RMSE: {rmse}')

    # Rolling over time. window size: 1h
    # df['ts'] = df.index.get_level_values('ts')
    # mae_rolling = df[[y_col_nm, kalman_mean_col, 'ts']].set_index('ts').rolling('1h').apply(lambda x: mean_absolute_error(x[y_col_nm], x[kalman_mean_col]))
    # rmse_rolling = df[[y_col_nm, kalman_mean_col, 'ts']].set_index('ts').rolling('1h').apply(lambda x: mean_squared_error(x[y_col_nm], x[kalman_mean_col], squared=False))
    # # plot the error in scatter line plot
    # fig = make_subplots(rows=1, cols=1)
    # fig.add_trace(go.Scatter(x=mae_rolling.index, y=mae_rolling, mode='markers', name=f'MAE', marker=dict(size=2)), row=1, col=1)
    # fig.add_trace(go.Scatter(x=rmse_rolling.index, y=rmse_rolling, mode='markers', name=f'RMSE', marker=dict(size=2)), row=1, col=1)
    # show(fig)


def plot_kalman_results(df_trades: pd.DataFrame, df_quote: pd.DataFrame, fit_moneyness=(0.8, 1.2)):
    dt = df_trades.index[0][0].date()
    scoped_ts_quote = [ts for ts in df_quote.index.get_level_values('ts') if ts.date() == dt]

    expiries = df_trades.index.get_level_values('expiry').unique()
    s0 = df_trades['spot'].iloc[0]
    strikes_fit = [s for s in df_trades.index.get_level_values('strike').unique() if fit_moneyness[0] - 1 < abs(float(s) / s0 - 1) < fit_moneyness[1] - 1]

    fig = make_subplots(rows=len(expiries), cols=2, subplot_titles=list(chain(*[[f'{dt.isoformat()}-call', f'{dt.isoformat()}-put'] for dt in expiries])))

    for i, (expiry, s_df) in enumerate(df_trades.groupby('expiry')):
        row = i + 1
        for j, (right, ss_df) in enumerate(s_df.groupby('right')):
            col = j + 1
            strikes_inters = sorted(set(strikes_fit).intersection(ss_df.index.get_level_values('strike')))
            df_sample = ss_df.loc[(slice(None), expiry, strikes_inters, right)]

            v_m = strike2moneyness_fwd_ln(df_sample['strike_flt'], df_sample['spot'], net_yield, df_sample['tenor'])

            fig.add_trace(go.Scatter(x=v_m, y=df_sample[kalman_mean_col], mode='markers', name=f'Kalman {expiry} | {right} | {y_col_nm}', marker=dict(size=2)), row=row, col=col)
            # fig.add_trace(go.Scatter(x=v_v_m[i], y=smoothed_ivs_upper[i], mode='markers', name=f'Kalman U {expiry} | {right} | {y_col_nm}', marker=dict(size=2)),
            # row=row, col=col)
            # fig.add_trace(go.Scatter(x=v_v_m[i], y=smoothed_ivs_lower[i], mode='markers', name=f'Kalman L {expiry} | {right} | {y_col_nm}', marker=dict(size=2)),
            # row=row, col=col)
            fig.add_trace(go.Scatter(x=v_m, y=df_sample[y_col_nm], mode='markers', name=f'Fill IV {expiry} | {right} | {y_col_nm}', marker=dict(size=2)), row=row, col=col)

    title = f'{equity}_iv_quote_trade_by_ts'
    fn = f'{title}.html'
    fig.update_layout(title=title)
    show(fig, fn=fn)

    ############################

    # Plot a couple IVs over time
    scoped_expiry = [expiries[0]]
    scoped_ts = df_trades.index.get_level_values('ts')

    fig_ts = make_subplots(rows=len(strikes_fit), cols=2, subplot_titles=list(chain(*[[f'{s}-call', f'{s}-put'] for s in strikes_fit])))
    for m, (right, s_df) in enumerate(df_trades.loc[(scoped_ts, scoped_expiry, strikes_fit, slice(None))].groupby('right')):
        col = m + 1
        for n, (strike, ss_df) in enumerate(s_df.groupby('strike')):
            # if strike not in strikes_fit:
            #     continue
            row = n + 1
            x = ss_df.index.get_level_values('ts').unique()
            y_iv_fill = ss_df[y_col_nm]
            y_iv_kalman = ss_df[kalman_mean_col]

            fig_ts.add_trace(go.Scatter(x=x, y=y_iv_fill, mode='markers', name=f'Fill IV {strike} {right}', marker=dict(size=2)), row=row, col=col)
            fig_ts.add_trace(go.Scatter(x=x, y=y_iv_kalman, mode='markers', name=f'Kalman IV {strike} {right}', marker=dict(size=2)), row=row, col=col)
            fig_ts.add_trace(go.Scatter(x=x, y=ss_df[kalman_up_col], mode='markers', name=f'U Kalman IV {strike} {right}', marker=dict(size=2)), row=row, col=col)
            fig_ts.add_trace(go.Scatter(x=x, y=ss_df[kalman_low_col], mode='markers', name=f'L Kalman IV {strike} {right}', marker=dict(size=2)), row=row, col=col)

            fig_ts.add_trace(go.Scatter(x=x, y=ss_df[kalman_my_ask_col], mode='markers', name=f'MyAsk Kalman IV {strike} {right}', marker=dict(size=2)), row=row, col=col)
            fig_ts.add_trace(go.Scatter(x=x, y=ss_df[kalman_my_bid_col], mode='markers', name=f'MyBid Kalman IV {strike} {right}', marker=dict(size=2)), row=row, col=col)

            # # Plot the bid/ask IVs
            ss_df_quote = df_quote.loc[(scoped_ts_quote, scoped_expiry, strike, right)]
            x_quote = ss_df_quote.index.get_level_values('ts').unique()
            fig_ts.add_trace(go.Scatter(x=x_quote, y=ss_df_quote['bid_iv'], mode='markers', name=f'Bid IV {strike} {right}', marker=dict(size=4)), row=row, col=col)
            fig_ts.add_trace(go.Scatter(x=x_quote, y=ss_df_quote['ask_iv'], mode='markers', name=f'Ask IV {strike} {right}', marker=dict(size=4)), row=row, col=col)

    title = f'{equity}_iv_kalman_fill_trade_by_strike'
    fn = f'{title}.html'
    fig_ts.update_layout(title=title)
    show(fig_ts, fn=fn)

    # print how many fills are within the confidence intervals grouped by expiry, right, strike
    log_95 = {}
    log_bid = {}
    log_ask = {}
    for expiry, s_df in df_trades.groupby('expiry'):
        for right, ss_df in s_df.groupby('right'):
            for strike, sss_df in ss_df.groupby('strike'):
                if strike in strikes_fit:
                    df_sample = sss_df[(~sss_df[y_col_nm].isna()) & (~sss_df[kalman_low_col].isna()) & (~sss_df[kalman_low_col].isna())]
                    log_95[(expiry, right, strike, len(df_sample))] = ((df_sample[y_col_nm] > df_sample[kalman_low_col]) & (
                            df_sample[y_col_nm] < df_sample[kalman_up_col])).sum() / len(df_sample)
                    log_bid[(expiry, right, strike, len(df_sample))] = ((df_sample[y_col_nm] < df_sample[kalman_my_bid_col])).sum() / len(df_sample)
                    log_ask[(expiry, right, strike, len(df_sample))] = ((df_sample[y_col_nm] > df_sample[kalman_my_ask_col])).sum() / len(df_sample)
    pprint(log_95)
    print('% of bid fills' + '-' * 50)
    pprint(log_bid)
    print('% of ask fills' + '-' * 50)
    pprint(log_ask)
    # On the right track, but occasional 3% where 20 is expected can be worrying as it'll result in 0, unless we're selling/buying right into bid/ask.
    # Here, insufficient quote resolution for that.


if __name__ == '__main__':
    from options.frame_builder import get_option_frame
    from estimators.earnings_iv_drop_ssvi_surface.train_estimator import scoped_dates, ScopePrePost
    sym = 'PEP'
    equity = Equity(sym)
    resolution = Resolution.second
    seq_ret_threshold = 0.001
    take = -2
    release_date = EarningsPreSessionDates(sym)[take]
    rate = DiscountRateMarket
    dividend_yield = get_dividend_yield(sym)
    net_yield = rate - dividend_yield

    marker_size = 2
    dates = scoped_dates(release_date, ScopePrePost.all)
    option_frame = get_option_frame(equity, dates, resolution=resolution, seq_ret_threshold=seq_ret_threshold)
    df_quote = option_frame.df_options

    # T-1 fit initial state
    df_trades = option_frame.df_option_trades.sort_index()
    df_trades['strike_flt'] = df_trades.index.get_level_values('strike').astype(float)
    release_date_m1 = max([el for el in set(df_trades.index.get_level_values('ts').date) if el < release_date])

    kalman_states = fit_kalman_states(df_trades, release_date, net_yield)
    enrich_kalman(df_trades, kalman_states, net_yield=net_yield)

    ix_release_date_m1 = [ts for ts in df_trades.index.get_level_values('ts').unique() if ts.date() == release_date_m1]
    ix_release_date = [ts for ts in df_trades.index.get_level_values('ts').unique() if ts.date() == release_date]

    # kalman_error_report(df_trades.loc[(ix_release_date_m1, slice(None), slice(None), slice(None))])
    kalman_error_report(df_trades.loc[(ix_release_date, slice(None), slice(None), slice(None))])

    # scoped_ts = [ts for ts in df_trades.index.get_level_values('ts').unique() if ts.date() == release_date]
    plot_kalman_results(df_trades.loc[(ix_release_date, slice(None), slice(None), slice(None))], df_quote)

    # scoped_ts = [ts for ts in df_trades.index.get_level_values('ts').unique() if ts.date() == release_date + pd.Timedelta(days=1)]
    plot_kalman_results(df_trades.loc[(ix_release_date, slice(None), slice(None), slice(None))], df_quote)

    print(f'Done {sym}')
