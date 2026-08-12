from datetime import date, datetime, time
from itertools import chain
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from pykalman import KalmanFilter

from connector.api_minlp.common import parse_pb, right_pb2right
from connector.api_minlp.handlers.kalman_init import get_kalman_init
from options.frame_builder import available_sym_dates, get_option_frame
from estimators.earnings_iv_drop_ssvi_surface.kalman_filter import KalmanInitialState, get_params
from estimators.earnings_iv_drop_ssvi_surface.train_estimator import get_v_ivs, get_time_buckets_by_equity_returns
from connector.api_minlp.protos.RequestKalmanInit_pb2 import RequestKalmanInit as RequestKalmanInit_pb
from connector.api_minlp.protos.RequestSSVICalibration_pb2 import SSVIParams as SSVIParams_pb
from options.helper import get_moneyness_fwd_ln, get_dividend_yield
from options.types.enums import OptionRight, Resolution
from options.types.equity import Equity
from options.types.iv_surface_essvi import IVSurface
from options.types.option import price_put
from options.types.option_contract import OptionContract
from shared.constants import dt_fmt_pb, DiscountRateMarket


def ssvi_params2dct(ssvi_params: List[SSVIParams_pb]) -> Dict[Tuple[date, OptionRight | str], Tuple[float, float, float]]:
    return {(date.fromisoformat(p.tenor_dt), right_pb2right(p.right)): (p.model_params.theta, p.model_params.rho, p.model_params.psi) for p in ssvi_params}


def get_ssvi_kalman_filter(sym_date):
    request_kalman_init = RequestKalmanInit_pb(
        underlying=sym_date.symbol,
        ts=sym_date.date.strftime(dt_fmt_pb),
        date_fit_start=sym_date.date.strftime(dt_fmt_pb),
        date_fit_end=sym_date.date.strftime(dt_fmt_pb),
    )
    res = get_kalman_init(request_kalman_init.SerializeToString())
    init_state = [parse_pb(p, SSVIParams_pb) for p in res[0]]
    init_covariance = np.array([res[1][i, :] for i in range(len(res[1]))])
    n_dim_state = len(init_state) * 3
    state_0 = KalmanInitialState(
        equity=sym_date.symbol,
        X=np.array([[p.model_params.theta, p.model_params.rho, p.model_params.psi] for p in init_state]).flatten(),
        P=init_covariance,
        Q=np.eye(n_dim_state),
        R=np.eye(n_dim_state),
        tenors_right_pairs=[(date.fromisoformat(p.tenor_dt), right_pb2right(p.right)) for p in init_state],
        n_dim_state=n_dim_state,
        n_dim_obs=n_dim_state,
    )

    kf = KalmanFilter(
            n_dim_state=state_0.n_dim_state,
            n_dim_obs=state_0.n_dim_obs,
            initial_state_mean=state_0.X,
            initial_state_covariance=state_0.P,
            transition_matrices=state_0.F,
            transition_covariance=state_0.Q,
            observation_matrices=state_0.H,
            observation_covariance=state_0.R,
            em_vars=[],
        )

    ivs = IVSurface(Equity(sym_date.symbol))
    ivs.set_last_calibration_ts(sym_date.date)
    ivs.set_params(ssvi_params2dct(init_state))
    return kf, ivs


class SpreadMA:
    def __init__(self, period: int):
        self.period = period
        self._spread = {}

    def add_spread(self, option: str, spread: float):
        # if len(self._spread) >= self.period:
        #     self._spread.pop(next(iter(self._spread)))
        self._spread[option] = spread

    def get_spread(self, option: str):
        return self._spread.get(option, 0)

def confidence_intervals():
    L = np.linalg.cholesky(P)
    L_star = L.conj().T
    # Calculate A
    A = L @ L_star

    alpha = 0.1
    k = 2 * L / 2
    beta = 2
    W_a = (alpha * k - L) / alpha ** 2 * k
    W_c = W_a + 1 - alpha ** 2 + beta

    X_up = X + np.nan_to_num(np.sqrt(L / (1 - W_c)), 0) @ A
    X_down = X - np.nan_to_num(np.sqrt(L / (1 - W_c)), 0) @ A

    mean = np.asarray(X)
    cov = np.asarray(P)
    c = np.atleast_2d(np.eye(len(mean)))
    k_vars = len(mean)
    nobs = 30000

    values = c.dot(mean)
    quad_form = (c * cov.dot(c.T).T).sum(1)
    df = nobs - 1
    from scipy import stats
    t_critval = stats.t.isf(alpha / 2, df)
    ci_diff = np.sqrt(quad_form / df) * t_critval
    low = values - ci_diff
    upp = values + ci_diff

    # j = 5
    # X[j] + np.nan_to_num(np.sqrt(L / (1 - w)), 0) @ A[:, j]

    # Calculate sell and buy price + half spread
    # mv_n = multivariate_normal(mean=X, cov=P)

    # Compute the Cholesky decomposition of the covariance matrix
    # np.sqrt(P)
    # L = cholesky(P, lower=True)

    z = np.random.normal(size=(10000, 90))  # Independent standard normals
    samples = X + z @ L.T  # Transform to desired distribution
    v = np.percentile(samples, 90, axis=0)

    import numpy as np
    from scipy.stats import chi2
    k = len(X)
    # Critical value for 90% confidence
    alpha = 0.1
    chi2_val = chi2.ppf(1 - alpha, k)
    chi2.ppf(alpha, k)
    chi2.ppf(1 - alpha, 2)
    chi2.ppf(1 - 0.999999, k)
    chi2.ppf(0.999999, k)

    # Degrees of freedom
    k = len(X)  # Number of dimensions

    # Critical value for 90% confidence
    alpha = 0.1
    chi2_val = chi2.ppf(1 - alpha / 2, k)

    # Calculate confidence intervals
    confidence_radius = np.sqrt(chi2_val) * np.sqrt(np.diag(P))

    X_down = X - confidence_radius
    X_up = X + confidence_radius

    confidence_radius = np.sqrt(chi2_val) * np.sqrt(np.diag(P))

    # Calculate confidence intervals
    X_down = X - confidence_radius
    X_up = X + confidence_radius


def mape(x1, x2):
    return np.mean(np.abs((x1 - x2) / x1)) * 100

def get_scoped_options(ivs: IVSurface, df: pd.DataFrame) -> List[OptionContract]:
    contracts = df.apply(lambda ps: OptionContract(symbol='', underlying_symbol=sym_date.symbol, expiry=ps.name[1], strike=ps.name[2], right=ps.name[3]), axis=1).values
    scoped = []
    for c in set(contracts):
        if (c.expiry, c.right) in ivs.params.keys():
            scoped.append(c)
    return scoped


if __name__ == "__main__":
    """
    1. Load some SSVI Kalman fiter from T-1 where T is release date.
    2. Update the fiter every abs(0.2% price return change).
    3. Get a moving average of every options IV spread.
    4. Given the Kalman filter's covariance matrix, calculate a sell and buy price + half spread.

    5a. Get the mean IV of the day's final 30 minutes for each option.    
    5b. Volume: % of time kalman quote is within bid/ask spread.
                Alternatively, # of times SSVI qoute moved into bid/ask spread territory.
    5c: Return: MeanIV1 / QuoteIV0 - 1
    
    The more IV Quote was within bid/ask, the worse the quote for me! So  
    5d: Pick the alpha where quoted 5% of time.
    
    6. Plot alpha, return, volume, moneyness, tenor.
    
    7. Pick an alpha for backtesting! Essentially picked a pricing function!
    """

    tickers_ho = ['ORCL']
    seq_ret_threshold = 0.002
    alpha = 0.8
    spread_factor = 0.5

    trade_events = []

    for sym_date in available_sym_dates(','.join(tickers_ho))[3:4]:
        kf, ssvi = get_ssvi_kalman_filter(sym_date)
        X, P = kf.initial_state_mean, kf.initial_state_covariance

        equity = Equity(sym_date.symbol)
        r = DiscountRateMarket
        q = get_dividend_yield(equity)

        option_frame = get_option_frame(Equity(sym_date.symbol), [sym_date.date], resolution=Resolution.second, seq_ret_threshold=seq_ret_threshold)
        df_equity = option_frame.df_equity
        v_ivs = get_v_ivs(equity, [sym_date.date], Resolution.second, seq_ret_threshold, True)
        spread_ma = SpreadMA(500)
        # taking a shortcut here and just setting the entire day's average spread here for given option
        option_frame.df_options['symbol'] = option_frame.df_options.apply(lambda ps: OptionContract(symbol='',
            underlying_symbol=sym_date.symbol, expiry=ps.name[1], strike=ps.name[2], right=ps.name[3]).ib_symbol(), axis=1)

        # Pick a dataframe of last 30min. Gte average IV
        eod_start = datetime.combine(sym_date.date, time(hour=15, minute=45))
        eod_end = datetime.combine(sym_date.date, time(hour=16, minute=0))
        df_eod = option_frame.df_options.loc[eod_start:eod_end]

        option_frame.df_options['eod_mid_iv'] = np.nan
        for symbol, s_df in option_frame.df_options.groupby('symbol'):
            spread_ma.add_spread(symbol, (s_df['ask_iv'] - s_df['bid_iv']).mean())
            option_frame.df_options.loc[s_df.index, 'eod_mid_iv'] = df_eod[df_eod['symbol'] == str(symbol)]['mid_iv'].mean()
        option_frame.df_options['expiry'] = option_frame.df_options.index.get_level_values('expiry')
        option_frame.df_options['right'] = option_frame.df_options.index.get_level_values('right')
        option_frame.df_options['strike'] = option_frame.df_options.index.get_level_values('strike').astype(float)

        scoped_contracts = get_scoped_options(ssvi, option_frame.df_options)

        for start, end in get_time_buckets_by_equity_returns(df_equity['close'], seq_ret_threshold=seq_ret_threshold):
            # This will break at some point given that each IVS might have different params...
            ivs = next(iter([ivs for ivs in v_ivs if ivs.last_calibration_ts() >= start]))
            calc_date = ivs.last_calibration_ts().date()

            new_params = ssvi.params
            for i, ((tenor, right), new_z) in enumerate(ivs.params.items()):
                if (tenor, right) in new_params:
                    new_params[(tenor, right)] = new_z
                else:
                    print(f'Not found {(tenor, right)}')
            z = np.array(list(chain(*new_params.values())))

            if len(z) != len(X):
                print(f'Kalman filter has {len(X)} states but IVS has {len(z)} states. Skipping {start} - {end}')
                continue

            print(f'Diag / Total Covariance: {np.abs(np.diag(P)).sum() / np.abs(P).sum()}')
            print(f'Mean Variance: {np.mean(np.diag(P))}')
            print(f'MAPE between X and z before update: {mape(X, z)}')

            X, _ = kf.filter_update(X, P, observation=z, observation_matrix=kf.observation_matrices)
            ssvi.set_params(get_params(X, list(ssvi.params.keys())))

            # print relative error of ivs params vs kf updated params
            # visualize how kalman is smoothing temporal fluctuations in IVS params
            print(f'MAPE between X and z after update: {mape(X, z)}')

            # Price surface
            # For every scoped option, get a quote, price + spread*factor.
            df_scoped = option_frame.df_options.loc[start:end]
            for c in scoped_contracts:
                df_c = df_scoped[df_scoped['symbol'] == c.ib_symbol()]

                v_spot = df_c['spot'].values
                v_strike = df_c['strike'].values.astype(float)
                v_tenor = df_c['tenor'].values
                df_c['ivs_iv_mid'] = ssvi.iv(get_moneyness_fwd_ln(equity, v_strike, v_spot, v_tenor), c.expiry, c.right, calc_date)
                df_c['ivs_iv_sell'] = df_c['ivs_iv_mid'] + spread_ma.get_spread(c.ib_symbol()) * spread_factor
                df_c['ivs_iv_buy'] = df_c['ivs_iv_mid'] - spread_ma.get_spread(c.ib_symbol()) * spread_factor

                df_c['quote_sell'] = price_put(v_spot, v_strike, v_tenor, df_c['ivs_iv_sell'].values, r, q)
                df_c['quote_buy'] = price_put(v_spot, v_strike, v_tenor, df_c['ivs_iv_buy'].values, r, q)

                    # Loop through each time step. If a fill was somewhat likely, ie, my quote in between bid/ask, then
                ix_event = (df_c['quote_sell'] < df_c['ask_close']) | (df_c['quote_buy'] > df_c['bid_close'])
                for i, ps in df_c.loc[ix_event].iterrows():
                    trade_events.append(ps.to_dict())

        # Next an PnL Estimate: IV earned, IV * Vega0 earned.
        df = pd.DataFrame(trade_events)
        df['pl_iv'] = np.nan
        df['pl_usd'] = np.nan
        ix_sell = df.index[df['quote_sell'] < df['ask_close']]
        ix_buy = df.index[df['quote_buy'] > df['bid_close']]
        df.loc[ix_sell, 'pl_iv'] = df.loc[ix_sell, 'ivs_iv_sell'] - df.loc[ix_sell, 'mid_iv']
        df.loc[ix_buy, 'pl_iv'] = df.loc[ix_buy, 'mid_iv'] - df.loc[ix_buy, 'ivs_iv_buy']
        df['pl_usd'] = df['pl_iv'] * df['vega_mid_iv'] * 100

        # plot IV earned and USD earned as a surface

        # % of time my quote was within bid/ask spread. approximated by count of snaps  / total snaps.
        df['ratio_sell'] = np.nan
        df['ratio_buy'] = np.nan
        for symbol, s_df in df.groupby('symbol'):
            ix_sell = s_df.index[s_df['quote_sell'] < s_df['ask_close']]
            ix_buy = s_df.index[s_df['quote_buy'] > s_df['bid_close']]
            n_snaps_total = option_frame.df_options[option_frame.df_options['symbol'] == symbol].shape[0]
            df.loc[ix_sell, 'ratio_sell'] = len(ix_sell) / n_snaps_total
            df.loc[ix_buy, 'ratio_buy'] = len(ix_buy) / n_snaps_total
        # for symbol, s_df in df.groupby('symbol'):
            print(f'''{symbol}: ratio sell: {s_df['ratio_sell'].mean():.2f}, r buy: {s_df['ratio_buy'].mean():.2f}, mean pl: {s_df['pl_usd'].mean():.2f} median pl: {s_df['pl_usd'].median():.2f}''')
        pass
        # What portion of trading day, my quote was better than mid? If smoothing the mid, expect equal time call and put. if spread factor is 0.5.
        # Reduce that time! Only need ~25% of time to beat market when conditions are more profitable. That means spread factor of > 0.5.

            # ssvi.plot_surface_as_of(ivs.last_calibration_ts().date(), fn=f'ivs_{sym_date.symbol}_{start.strftime("%Y%M%dT%H%m%S")}-{end.strftime("%Y%M%dT%H%m%S")}.html')
            # ivs.plot_surface_as_of(ivs.last_calibration_ts().date(), fn=f'ivs_{sym_date.symbol}_{start.strftime("%Y%m%dT%H%M%S")}-{end.strftime("%Y%m%dT%H%M%S")}.html')

            # p = stats.norm.ppf(0.92)
            # X_up = X + p * np.sqrt(np.diag(P))
            # X_down = X - p * np.sqrt(np.diag(P))
            #
            # ivs_up = IVSurface(Equity(sym_date.symbol))
            # ivs_up.set_last_calibration_ts(ivs.last_calibration_ts())
            # ivs_up.set_params(get_params(X_up, list(ssvi.params.keys())))
            # ivs_up.plot_surface_as_of(ivs.last_calibration_ts().date(), fn=f'ivs_up_{sym_date.symbol}_{ivs.last_calibration_ts().date()}.html')
            #
            # ivs_down = IVSurface(Equity(sym_date.symbol))
            # ivs_down.set_last_calibration_ts(ivs.last_calibration_ts())
            # ivs_down.set_params(get_params(X_down, list(ssvi.params.keys())))
            # ivs_down.plot_surface_as_of(ivs.last_calibration_ts().date(), fn=f'ivs_down_{sym_date.symbol}_{ivs.last_calibration_ts().date()}.html')


