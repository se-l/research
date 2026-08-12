from functools import reduce

import numpy as np
import pandas as pd

from itertools import chain, groupby
from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Tuple
from pykalman import KalmanFilter

from options.types.iv_surface_essvi import IVSurface
from options.helper import timer
from options.types.equity import Equity
"""
Kalman Filter for the model parameters of an IV surface. Here, theta, rho, psi.
No trends models. Just expected mean and variance of the parameters.

Load IVS surfaces. Each has 3 params * 2 right * tenors vars => One statevector.
"""


@dataclass
class KalmanState:
    ts: pd.Timestamp
    tenors: List[date]  # has same shape as z
    z: np.ndarray
    X: np.ndarray
    P: np.ndarray
    # residual = v_z[i] - np.dot(observation_matrices[i], a)


@dataclass
class KalmanInitialState:
    equity: Equity
    X: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    tenors: List[date]  # has same shape as z
    n_dim_state: int = None
    n_dim_obs: int = None
    F: np.ndarray = None
    H: np.ndarray = None

    def __post_init__(self):
        self.n_dim_state = len(self.X)
        self.n_dim_obs = len(self.X)
        self.F = np.eye(self.n_dim_state)
        self.H = np.eye(self.n_dim_state)


def get_measurement(ivs: IVSurface):
    # make this deterministic sorting by tenor and right
    return np.array(list(chain(*ivs.params.values())))


def get_params(z: np.ndarray, tenors: List[date]):
    return {tenors[i]: (z[i*3], z[i*3 + 1], z[i*3 + 2]) for i in range(len(tenors))}


def train_kalman_initial_state(equity: Equity, v_ivs_in: List[IVSurface], n_iter=100) -> KalmanInitialState:
    """
    Not working yet, having np.nan for missing observations. for kf.em, all must be finite number
    :param equity:
    :param v_ivs_in:
    :param n_iter:
    :return:
    """
    v_z = []
    v_tenors = []
    for i, ivs in enumerate(sorted(v_ivs_in, key=lambda x: x.last_calibration_ts())):
        z = get_measurement(ivs)
        v_z.append(z)
        v_tenors.append(list(ivs.params.keys()))

    unique_tenors = sorted(reduce(lambda a, b: a.union(set(b)), v_tenors, set()))
    tenor_to_ix = {tenor: i for i, tenor in enumerate(unique_tenors)}
    n_dim_state = len(unique_tenors) * 3

    v_z_reshaped = []
    for v_tenor, z in zip(v_tenors, v_z):
        z_reshaped = np.full(len(unique_tenors) * 3, np.nan, dtype=float)

        for k, tenor in enumerate(v_tenor):
            ix = tenor_to_ix[tenor] * 3
            z_reshaped[ix:ix + 3] = z[k:k + 3]
            v_z_reshaped.append(z_reshaped)  # possible wrong. too many observations.
        # v_z_reshaped.append(z_reshaped) # possibly should be

    kf = KalmanFilter(
        n_dim_state=n_dim_state,
        n_dim_obs=n_dim_state,
        transition_matrices=np.eye(n_dim_state),  # F
        observation_matrices=np.eye(n_dim_state),  # H    z=Hx+v
        initial_state_mean=v_z_reshaped[0],
        # initial_state_covariance=P,  # This will be calibrated!
        observation_covariance=np.eye(n_dim_state),  # R
        transition_covariance=np.eye(n_dim_state),  # Q
        em_vars=[
            'initial_state_covariance',  # P
        ],
    )
    try:
        kf.em(v_z_reshaped, n_iter=n_iter, em_vars=['initial_state_covariance'])
    except Exception as e:
        raise e

    return KalmanInitialState(
        equity,
        kf.initial_state_mean,
        kf.initial_state_covariance,
        kf.transition_covariance,
        kf.observation_covariance,
        unique_tenors
    )


def initialize_kalman_filter(equity: Equity, v_ivs: List[IVSurface]) -> Tuple[KalmanFilter, KalmanInitialState]:
    state_0 = train_kalman_initial_state(equity, v_ivs)
    return KalmanFilter(
            n_dim_state=state_0.n_dim_state,
            n_dim_obs=state_0.n_dim_obs,
            initial_state_mean=state_0.X,
            initial_state_covariance=state_0.P,
            transition_matrices=state_0.F,
            transition_covariance=state_0.Q,
            observation_matrices=state_0.R,
            observation_covariance=state_0.H,
            em_vars=[],
        ), state_0


@timer
def fit_kalman_states(v_ivs: List[IVSurface]) -> Dict[Equity, List[KalmanState]]:
    """If IVS span too many calculation dates, the tenors change. Kalman filter in this shape does not support matrix shape changes.
    Fitting an initial state can kinda happen for a single day only, not across weeks.
    """
    kalman_states: Dict[Equity, List[KalmanState]] = {}

    f = lambda x: x.underlying
    for i, (equity, v_ivs_by_equity) in enumerate(groupby(sorted(v_ivs, key=f), f)):
        v_ivs_by_equity = sorted(list(v_ivs_by_equity), key=lambda x: x.last_calibration_ts())
        kf, state_0 = initialize_kalman_filter(equity, v_ivs_by_equity)

        v_kalman_states = []

        X, P = kf.initial_state_mean, kf.initial_state_covariance

        for k, ivs in enumerate(sorted(v_ivs_by_equity, key=lambda x: x.last_calibration_ts())):
            z = get_measurement(ivs)
            if z.shape != X.shape:
                z = np.full(len(state_0.tenors) * 3, np.nan, dtype=float)

                tenor_to_ix = {tenor: i for i, tenor in enumerate(tenors_all)}

                for tenor, (theta, rho, psi) in ivs.params.items():
                    i = tenor_to_ix[tenor] * 3
                    z[i:i + 3] = [theta, rho, psi]

            X, P = kf.filter_update(X, P, observation=z, observation_matrix=kf.observation_matrices)

            # X2, P2 = kf.filter_update(X, P, observation=z, observation_matrix=H)
            # x_mine, p_mine = kf_update(P, H, z, X, Q, R)
            # if False:
            #     assert np.allclose(x_mine, X2)
            #     assert np.allclose(p_mine, P2)
            # X = X2
            # P = P2

            v_kalman_states.append(
                KalmanState(
                    ivs.last_calibration_ts(),
                    list(ivs.params.keys()),
                    z,
                    X,
                    P
                )
            )

        kalman_states[equity] = v_kalman_states

    return kalman_states


# @njit
def kf_update(P: np.ndarray, H: np.ndarray, z: np.ndarray, X: np.ndarray, Q: np.ndarray, R: np.ndarray):
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


def run():
    from options.frame_builder import available_sym_dates
    from estimators.earnings_iv_drop_ssvi_surface.train_estimator import ScopePrePost, get_ivs_mp

    tickers_ho = ['FDX']

    sym_dates_ho = available_sym_dates(','.join(tickers_ho))
    for sym_date in sym_dates_ho:
        v_ivs = get_ivs_mp([sym_date], mp=False, scope=ScopePrePost.all)
        # group them by calibrated calculation date...
        kalman_states = fit_kalman_states(v_ivs)


if __name__ == '__main__':
    run()
    print(f'Done.')
