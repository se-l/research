import itertools

import numpy as np
import pandas as pd # replace with JL package
import plotly.graph_objects as go # replace with JL package
import plotly.express as px # replace with JL package

from pprint import pprint
from decimal import Decimal
from functools import lru_cache # replace with JL package
from datetime import datetime, date, timedelta, time
from typing import List, Dict, Tuple, Union, Set

from pandas.tseries.holiday import USFederalHolidayCalendar # needed
from plotly.subplots import make_subplots # replace with JL package
from dataclasses import dataclass
from pyomo.core import ConcreteModel # no JL
from pyomo.environ import value # no JL

from estimators.earnings_iv_drop_ssvi_surface.train_estimator import prep_x_w_return  # need to call python
from estimators.utils import score # jl
from options.frame_builder import get_option_frame  # jl
from estimators.earnings_iv_drop_ssvi_surface.estimator_lgbm import ESSVITreeEstimator  #  jl
from connector.api_minlp.protos.RequestSSVICalibration_pb2 import SSVIParamsPb #jl
from options.surfaces.processors import get_ivs_by_date  #   # jl
from options.types.SSVIParams import SSVITenorParams, SSVISurfParams # jl
from options.types.cash import Cash # jl
from options.types.dividend import get_dividends # jl
from options.types.earnings_config import EarningsConfig # jl
# from derivatives.kalman import kalman_my_bid_col, kalman_my_ask_col # need to call python
kalman_my_ask_col = 'fill_iv_kalman_my_ask'
kalman_my_bid_col = 'fill_iv_kalman_my_bid'
from options.types.enums import OptionRight #jl
from options.types.quote_side import QuoteSide #jl
from options.types.scenario import Scenario #jl
from options.types.security import Security, SecurityDataSnap #jl
from options.types.stress_test_ds_result import StressTestDsResult # jl
from options.helper import earnings_download_dates, exclude_outlier_quotes, add_trade_days, date_to_sod, date_to_eod_expiry, get_v_tenor, dividends2amount_times,  val_from_df, atm_iv, get_moneyness_fwd_ln, get_tenor # can all move to julia if not already in helper.jl
from options.types.sym_date import SymDate # jl
from options.volatility.pf_opt_minlp_pyomo import derive_portfolio_milnp # jl
from shared.modules.logger import info, warning # jl
from options.types.portfolio import Portfolio # Jl
from shared.constants import EarningsPreSessionDates # jl
from options.types.equity import Equity # jl
from options.types.option_contract import OptionContract # jl
from options.types.option import Option # jl
from fino.fino_load import jl
from shared.yield_curve import YieldCurve # jl
from shared.plotting import show # some util to show plot in browser
from options.types.iv_surface_essvi import IVSurface # jl
from shared.utils.decorators import time_it  # use jl function


@dataclass
class NLVSnapshot:
    ts: pd.Timestamp
    nlv: float


@dataclass
class IVTransitionMatrices:
    m_dnlv01_worst_buy: np.ndarray
    m_dnlv01_worst_sell: np.ndarray
    m_dnlv01_best_buy: np.ndarray
    m_dnlv01_best_sell: np.ndarray
    m_dnlv01_estimated_buy: np.ndarray
    m_dnlv01_estimated_sell: np.ndarray
    v_delta0_mid: np.ndarray


@dataclass
class SolverResult:
    er: 'EarningsReleaseSimulation'
    pf: Portfolio
    pyo_model: ConcreteModel
    pyo_inst: ConcreteModel

def get_max_obj_value(res: SolverResult):
    return float(value(res.pyo_inst.obj) or 0)

bday_us = pd.tseries.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())


class EarningsReleaseSimulation:
    def __init__(self, cfg: EarningsConfig, scoped_symbols: List[str] = None):
        self.cfg = cfg
        self.equity = Equity(cfg.sym)
        self.release_date = cfg.release_date
        self.pf: Portfolio = cfg.portfolio

        self.sym = self.equity.symbol
        self._option_frame = None
        self._df = None
        self._df0 = None
        self._df1 = None
        self._v_ts_pre_release = None
        self._v_ts_post_release = None
        self._ivs1: Dict[Tuple[float, float] | None, IVSurface] = {}
        self._option_universe: Set[Option] = set()
        self._scoped_options: List[Option] = []
        self._iv_transition_matrices: Dict[float | None, IVTransitionMatrices] = {}
        self.nlv0_bid: List[float] = []
        self.nlv0_ask: List[float] = []
        self._ivs0_bid = None
        self._ivs0_ask = None
        self._xy = {}
        self.scoped_symbols = scoped_symbols

    def __hash__(self):
        return hash((self.sym, self.release_date))

    @property
    def option_frame(self):
        if self._option_frame is None:
            dates = earnings_download_dates(self.release_date)
            self._option_frame = get_option_frame(self.equity, dates, resolution=self.cfg.resolution, seq_ret_threshold=self.cfg.seq_ret_threshold)
            # self._option_frame.df_option_trades = trade_transformations(self._option_frame.df_option_trades, self.cfg)
            self._option_frame.df_option_trades = exclude_outlier_quotes(self._option_frame.df_option_trades, self.pf, self.equity)
            self._option_frame.df_options = exclude_outlier_quotes(self._option_frame.df_options, self.pf, self.equity)
        return self._option_frame

    @property
    def df(self):
        if self._df is None:
            self._df = self.option_frame.df_options
            info(f'df: shape: {self._df.shape}')
            assert self._df['mid_price'].isna().sum() == 0
        return self._df

    @property
    def v_ts_pre_release(self) -> List[pd.Timestamp]:
        if self._v_ts_pre_release is None:
            ts = self.df.index.get_level_values('ts').unique()

            # Adjustment. hedges can choose on which date. sell high IV is before release to center around underlying price.
            if self.release_date in {el.date() for el in ts}:
                self._v_ts_pre_release = [i for i in ts if i.date() == self.release_date]  # preparing on day of expiry
            else:
                # v_ts_pre_release = [i for i in ts if i.date() <= self.release_date] # preparing before release date
                self._v_ts_pre_release = [i for i in ts if i.date() == (self.release_date - bday_us).date()]  # preparing options the night before

            self._v_ts_pre_release = [i for i in self._v_ts_pre_release if i.hour >= 10]

        return self._v_ts_pre_release

    @property
    def v_ts_post_release(self) -> List[pd.Timestamp]:
        if self._v_ts_post_release is None:
            ts = self.df.index.get_level_values('ts').unique()
            self._v_ts_post_release = [i for i in ts if i.date() >= (self.release_date + bday_us).date()]

            self._v_ts_post_release = [i for i in self._v_ts_post_release if i.hour >= 12]
            first_2_days = sorted(list(set([i.date() for i in self._v_ts_post_release])))[:2]
            self._v_ts_post_release = [i for i in self._v_ts_post_release if i.date() in first_2_days]

        return self._v_ts_post_release

    @property
    def ts_pre_release(self):
        # print(f'Release date: {release_date}. ts_pre_release: {ts_pre_release}')
        return self.v_ts_pre_release[1]

    @property
    def ts_post_release(self):
        return self.v_ts_post_release[0] if self.v_ts_post_release else None
        # if ts_post_release:
        #     print(f'TS Post Release - Min: {np.min(v_ts_post_release)} Max: {np.max(v_ts_post_release)}')

    @property
    def calc_date0(self):
        return self.ts_pre_release.date()

    @property
    @lru_cache(maxsize=1)
    def calc_date1(self) -> date:
        return add_trade_days(self.release_date, 1)

    @property
    def df0(self):
        if self._df0 is None:
            self._df0 = self.df.loc[self.v_ts_pre_release]
            info(f'df0: self._df0 shape: {self._df0.shape} - self.ts_pre_release: {self.ts_pre_release}')
        return self._df0

    @property
    def df1(self):
        if self._df1 is None:
            self._df1 = self.df.loc[self.v_ts_post_release] if self.v_ts_post_release else None
        return self._df1

    @property
    def s0(self):
        return self.df0.loc[self.ts_pre_release].iloc[0]['spot']

    @property
    def s1(self):
        return self.df1.loc[self.ts_post_release].iloc[0]['spot'] if self.v_ts_post_release else np.nan
        # print(f'''Spot0: {s0}; Spot1: {s1}; Stock jumped by dS: {round(s1 - s0, 1)} | {round(100 * (s1 / s0 - 1), 1)} %''')

    @property
    def expiries(self) -> List[date]:
        return list(sorted(set(self.df0.index.get_level_values('expiry').values)))

    @property
    def exp_jump1(self):
        return sorted([i for i in self.expiries if i >= self.release_date])[0]

    @property
    def option_universe(self) -> Set[Option]:
        if not self._option_universe:
            t0 = datetime.now().timestamp()
            sym = self.equity.symbol
            self._option_universe = set()

            # Here, should apply scope filters
            info(f'option universe: len df0 index: {len(self.df0.index)}')
            for ix in self.df0.index:
                ts, expiry, strike, right = ix
                if expiry <= self.release_date:
                    continue
                self._option_universe.add(get_option(sym, expiry, strike, right, self.calc_date0))
            info(f'Added {len(self._option_universe)} options to universe in {datetime.now().timestamp() - t0:.2f} seconds.')

        return self._option_universe

    @property
    def ivs0_bid(self) -> IVSurface:
        if self._ivs0_bid is None:
            v_ivs_bid = get_ivs_by_date(self.equity, sorted(set([ts.date() for ts in self.v_ts_pre_release])), self.cfg.resolution, self.cfg.seq_ret_threshold, side=QuoteSide.bid)
            self._ivs0_bid = v_ivs_bid[-1]
        return self._ivs0_bid

    @property
    def ivs0_ask(self) -> IVSurface:
        if self._ivs0_ask is None:
            v_ivs_ask = get_ivs_by_date(self.equity, sorted(set([ts.date() for ts in self.v_ts_pre_release])), self.cfg.resolution, self.cfg.seq_ret_threshold, side=QuoteSide.ask)
            self._ivs0_ask = v_ivs_ask[-1]
        return self._ivs0_ask

    def set_ivs0_params(self, ssvi_params_pb: List[SSVIParamsPb]):
        params = SSVISurfParams({datetime.strptime(p.tenor_dt, '%Y-%m-%d').date(): (p.model_params.theta, p.model_params.rho, p.model_params.psi) for p in ssvi_params_pb})
        self._ivs0_ask = IVSurface(self.equity, tag='ask').set_params(params)
        self._ivs0_bid = IVSurface(self.equity, tag='bid').set_params(params)

    def get_prep_x_w_return(self, sym_date: SymDate, alpha: float, ret: float):
        key = (sym_date, alpha, ret)
        if self._xy.get(key) is None:
            self._xy[key] = prep_x_w_return(
                [sym_date],
                estimators=ESSVITreeEstimator.load_model(self.cfg.earnings_iv_drop_regressor_model_name_version, alpha),
                alpha=alpha,
                ret=ret
            )
        return self._xy[key]

    def set_ivs1(self, ret, alpha):
        if abs(ret) > 0.5:
            warning(f'Outsized return. Likely wrong argument: {ret}')
        x = self.get_prep_x_w_return(
            sym_date=SymDate(self.sym, self.release_date),
            alpha=alpha,
            ret=ret
        )

        estimators = ESSVITreeEstimator.load_model(self.cfg.earnings_iv_drop_regressor_model_name_version, alpha)

        if x.empty:
            raise f'Failed to load data for {self.sym} on {self.release_date}.'

        # I dont have a tenor_dt in train set. So would want to predict the metrics for each.
        # Either reconstruct from float values or prevent its removal... lgb just picks needs column. additional ones dont hurt...
        df_params = pd.DataFrame({
            "tenor_dt": x["tenor_dt"].values,
            **{metric: est.predict(x) for metric, est in estimators.items()},
        })

        params: SSVISurfParams = SSVISurfParams(
            {
                tenor_dt: SSVITenorParams(**row[["theta", "rho", "psi"]].to_dict())
                for tenor_dt, row in df_params.groupby("tenor_dt").mean(numeric_only=True).iterrows()
            }
        )

        self._ivs1[(ret, alpha)] = IVSurface(self.equity, tag=f"alpha: {alpha}").set_params(params)

        if False:#self.cfg.plot:
            # True Surface
            params_y = SSVISurfParams(
                {
                    tenor_dt: SSVITenorParams(*row[['theta_y', 'rho_y', 'psi_y']])
                    for tenor_dt, row in self._xy[key].groupby("tenor_dt").mean(numeric_only=True).iterrows()
                }
            )
            IVSurface(self.equity).set_last_calibration_ts(self.calc_date1).set_params(params_y).plot_surface_vanilla(self.calc_date1)

            # Predicted Surface
            self._ivs1[(ret, alpha)].plot_surface_vanilla(self.calc_date1)


    def get_ivs1(self, ret: float, alpha=None) -> IVSurface:
        _alpha = alpha or 0.5
        if self._ivs1.get((ret, _alpha)) is None:
            self.set_ivs1(ret, _alpha)
        return self._ivs1[(ret, _alpha)]

    def plot_ivs0(self, x):
        params0: SSVISurfParams = SSVISurfParams(
            {
                tenor_dt: SSVITenorParams(**row[["theta", "rho", "psi"]].to_dict())
                for tenor_dt, row in x.groupby("tenor_dt").mean(numeric_only=True).iterrows()
            }
        )
        (IVSurface(self.equity)
         .set_last_calibration_ts(self.calc_date0)
         .set_params(params0)
         .plot_surface_vanilla(self.calc_date0, expiries=sorted(params0.keys())))

    def plot_return_vs_model_params(self):
        fig = make_subplots(3, 1, subplot_titles=[''], row_heights=[0.5, 0.5, 0.5], vertical_spacing=0.1)

        returns = np.linspace(-0.2, 0.2, 21)

        ix_col = 1

        for ret in returns:
            ivs = self.get_ivs1(ret)
            expiries = list(ivs.params.keys())

            thetas = [ivs.params[tenor].theta for tenor in expiries]
            rhos = [ivs.params[tenor].rho for tenor in expiries]
            psis = [ivs.params[tenor].psi for tenor in expiries]

            fig.add_trace(go.Scatter(x=expiries, y=thetas, mode='lines+markers', name=f'{ret:.2f} theta'), row=1, col=ix_col)
            fig.add_trace(go.Scatter(x=expiries, y=rhos, mode='lines+markers', name=f'{ret:.2f} rho'), row=2, col=ix_col)
            fig.add_trace(go.Scatter(x=expiries, y=psis, mode='lines+markers', name=f'{ret:.2f} psi'), row=3, col=ix_col)

        show(fig)


    def set_matrices(self, alpha=None):
        t0 = datetime.now().timestamp()

        m_dnlv01_worst_buy, m_dnlv01_worst_sell, m_dnlv01_best_buy, m_dnlv01_best_sell = self.get_nlv_delta_matrix(alpha)

        v_delta0_bid, v_delta0_ask = get_v_delta_bid_ask(self.s0, self.ivs0_bid, self.ivs0_ask, self.scoped_options, self.calc_date0)

        m_spread_pc_credit = self.get_m_spread_pc_credit(len(self.scoped_options))
        # m_spread_pc_credit = get_m_spread_pc_credit(self.cfg, self.scoped_options, self.calc_date0, self.s0)
        m_dnlv01_estimated_buy, m_dnlv01_estimated_sell = get_estimate_m_dnlv_buy_sell(
            m_dnlv01_worst_buy, m_dnlv01_best_buy, m_dnlv01_worst_sell, m_dnlv01_best_sell, m_spread_pc_credit
        )
        v_delta0_mid = (v_delta0_bid + v_delta0_ask) / 2

        self._iv_transition_matrices[alpha] = IVTransitionMatrices(
            m_dnlv01_worst_buy, m_dnlv01_worst_sell, m_dnlv01_best_buy, m_dnlv01_best_sell, m_dnlv01_estimated_buy, m_dnlv01_estimated_sell, v_delta0_mid
        )
        info(f'Set matrices in {datetime.now().timestamp() - t0:.2f} seconds.')

    def get_iv_transition_matrices(self, alpha: float = None) -> IVTransitionMatrices:
        if self._iv_transition_matrices.get(alpha) is None:
            self.set_matrices(alpha)
        return self._iv_transition_matrices[alpha]

    @property
    def scoped_options(self) -> List[Option]:
        if not self._scoped_options:
            t0 = datetime.now().timestamp()
            # scoped_expiries = get_scoped_expiries(self.expiries)
            get_p0 = lambda o: val_from_df(self.df0.loc[self.v_ts_pre_release], o.expiry, o.optionContract.strike, o.right, 'mid_price', 0)
            self._scoped_options = [o for o in self.option_universe if
                                    option_in_pf_scope(o, self.cfg, self.calc_date0, self.s0, get_p0(o), pf=self.pf) and
                                    # option_in_pf_scope(o, self.cfg, self.calc_date0, self.s0, get_p0(o), scoped_expiries=scoped_expiries, pf=self.pf) and
                                    # self.get_ivs1(0).is_calibrated_slice(o.expiry) and
                                    # (o.symbol in self.scoped_symbols if self.scoped_symbols else True)
                                    (o.symbol in self.scoped_symbols if self.scoped_symbols is not None else True)
                                    ] + [self.equity]
            info(f'Scoped securities: #{len(self._scoped_options)} out of # {len(self.option_universe)} in {datetime.now().timestamp() - t0:.2f} seconds.')
        # info(f"{self._scoped_options}")
        return self._scoped_options

    def solve(self) -> SolverResult:
        info(f'EarningsReleaseSimulation.solve() for {self.sym} on {self.release_date}')

        res = derive_portfolio_milnp(
            self.scoped_options,
            m_dnlv01_buy=self.get_iv_transition_matrices().m_dnlv01_estimated_buy,
            m_dnlv01_sell=self.get_iv_transition_matrices().m_dnlv01_estimated_sell,
            cfg=self.cfg,
            f_weight_ds=get_f_weight_ds(),
            pf=self.pf,
        )
        if self.pf:
            new_holdings = {h.symbol: h.quantity - self.pf.get(h.symbol, 0) for h in res.pf if (h.quantity - self.pf.get(h.symbol, 0)) != 0}
            info(f'Added holdings by solver: {new_holdings}')
        return SolverResult(self, res.pf, res.pyo_model, res.pyo_inst)

    def stress_tests(self, pf: Portfolio, alphas: List[float] = None) -> List[StressTestDsResult]:
        from connector.api_minlp.common import get_stress_test_ds
        _alphas = alphas or [0.9, None, 0.1]
        v_stress_test_ds = []
        for alpha in _alphas:
            m = self.get_iv_transition_matrices(alpha)
            v_stress_test_ds.extend([
                get_stress_test_ds(self.scoped_options, pf, m.m_dnlv01_worst_buy, m.m_dnlv01_worst_sell, m.v_delta0_mid, self.cfg, self.s0, f'Spread: {Scenario.worst}, alpha: {alpha}'),
                get_stress_test_ds(self.scoped_options, pf, m.m_dnlv01_estimated_buy, m.m_dnlv01_estimated_sell, m.v_delta0_mid, self.cfg, self.s0, f'Spread: {Scenario.estimated} alpha: {alpha}'),
                get_stress_test_ds(self.scoped_options, pf, m.m_dnlv01_best_buy, m.m_dnlv01_best_sell, m.v_delta0_mid, self.cfg, self.s0, f'Spread: {Scenario.best} alpha: {alpha}')
            ])

        return v_stress_test_ds

    def v_nlv(self, pf_in: Portfolio, v_ts=None, delta_hedge=False, scenario: Scenario | str = Scenario.mid) -> List[NLVSnapshot]:
        """Needs delta hedging to be realistic. Unless equity close to zero. easier to compare to non-trading algo on T+1"""
        v_nlv = []
        pf = Portfolio(pf_in.holdings)
        if pf.underlying not in pf.holdings:
            pf.add_holding(Equity(pf.underlying), 0)
        else:
            pf.add_holding(Cash('USD'), pf[pf.underlying] * self.df.iloc[0]['spot'])

        v_ts = v_ts or self.df.index.get_level_values('ts').unique()
        first_nlv = None

        for i, ts in enumerate(v_ts):
            if ts.time() < time(9, 33):
                continue
            snap = market_snapshot(self.df, ts, pf)
            if snap is None:
                continue

            nlv = pf.nlv(snap, scenario=scenario)
            if first_nlv is None:
                first_nlv = nlv
            v_nlv.append(NLVSnapshot(ts, nlv - first_nlv))

            if delta_hedge:
                spot = snap[Equity(pf.underlying)].spot
                delta_total = int(pf.delta_total(ts.date(), spot, self.df.loc[v_ts].loc[:ts]))
                pf.add_holding(Equity(pf.underlying), -delta_total)
                pf.add_holding(Cash('USD'), delta_total * spot)
                info(f'v_nlv: Delta hedge: {delta_total} at {ts}')

        return v_nlv

    @time_it
    def get_nlv_delta_matrix(self, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        opts = [o for o in self.scoped_options if isinstance(o, Option)]
        equity_mask = [not isinstance(o, Option) for o in self.scoped_options]
        n_total = len(self.scoped_options)

        # ── collect per-option arrays ──────────────────────────────────────
        strikes = np.array([float(o.strike) for o in opts], dtype=np.float32)
        expiries = [o.expiry for o in opts]
        rights = np.array([1 if o.right == OptionRight.call else 0 for o in opts], dtype=np.uint8)
        multiplier = np.float32(Option.multiplier)

        # t0 tenors
        tenors_0 = get_v_tenor(
            np.array([date_to_eod_expiry(e) for e in expiries]),
            date_to_sod(self.calc_date0),
        ).astype(np.float32)

        # t1 tenors - given it's future, option with expiry-=calc_date0 have negative tenor, hence 0 extrinsic value.
        tenors_1 = np.maximum(
            get_v_tenor(
                np.array([date_to_eod_expiry(e) for e in expiries]),
                date_to_sod(self.calc_date1),
            ).astype(np.float32),
            np.float32(0.01 / 365.0),   # floor at 1/100th day
        )

        # t0 market bid/ask prices
        prices_bid_0 = np.array([
            val_from_df(self.df0.loc[self.v_ts_pre_release], o.expiry, o.optionContract.strike, o.right, 'bid_close', 0)
            for o in opts
        ], dtype=np.float32)
        prices_ask_0 = np.array([
            val_from_df(self.df0.loc[self.v_ts_pre_release], o.expiry, o.optionContract.strike, o.right, 'ask_close', 0)
            for o in opts
        ], dtype=np.float32)

        spots_0 = np.full(len(opts), self.s0, dtype=np.float32)

        # yield curve + dividends (shared across options, same underlying)
        yield_curve = YieldCurve().get_last_zero_curve(self.calc_date0, self.equity)

        dividends = get_dividends(self.sym, self.calc_date0)
        div_amounts, div_times = dividends2amount_times(dividends, self.calc_date0)

        # ── build SSVI model IV matrix: (n_ds × n_options) ────────────────
        # Each row is one dS scenario; columns are options
        # self.ivs0_ask.plot_surface_vanilla(self.calc_date0)
        # self.ivs0_bid.plot_surface_vanilla(self.calc_date0)
        # self.get_ivs1(1, alpha).plot_surface_vanilla(self.calc_date1)

        m_ivs_1 = np.array([
            [
                iv(self.get_ivs1(ds_ret - 1, alpha), o, self.s0 * ds_ret, self.calc_date1)
                for o in opts
            ]
            for ds_ret in self.cfg.v_ds_ret
        ], dtype=np.float32)
        info(f'PY get_nlv_delta_matrix.m_ivs_1: size={m_ivs_1.shape} sum={m_ivs_1.sum()})')

        v_spots_1 = np.array([self.s0 * ds for ds in self.cfg.v_ds_ret], dtype=np.float32)
        info(f's0: {self.s0} - size(ds)={len(self.cfg.v_ds_ret)}')

        # ── call Julia (one function, two GPU batches inside) ──────────────
        def to_jl_f32(arr):
            return jl.Vector[jl.Float32](arr.astype(np.float32))

        def to_jl_u8(arr):
            return jl.Vector[jl.UInt8](arr.astype(np.uint8))

        nlv0_bid_jl, nlv0_ask_jl, wl, ws, bl, bs = jl.Fino.EarningsReleaseSSVI.get_nlv_delta_matrix_inner(
            to_jl_f32(prices_bid_0),
            to_jl_f32(prices_ask_0),
            to_jl_f32(spots_0),
            to_jl_f32(strikes),
            to_jl_f32(tenors_0),
            to_jl_u8(rights),
            to_jl_f32(np.array(yield_curve.rates, dtype=np.float32)),
            to_jl_f32(np.array(yield_curve.times, dtype=np.float32)),
            to_jl_f32(np.array(div_amounts, dtype=np.float32)),
            to_jl_f32(np.array(div_times, dtype=np.float32)),
            multiplier,
            to_jl_f32(v_spots_1),
            to_jl_f32(tenors_1),
            jl.Matrix[jl.Float32](m_ivs_1),  # (n_ds × n_options)
        )

        # ── convert results back; re-insert equity column as v_spots_1 - self.s0 ─────────
        def _to_full(jl_mat_or_vec, n_total, equity_mask, n_ds=None):
            arr = np.array(jl_mat_or_vec, dtype=np.float64)
            if n_ds is not None:
                full = np.zeros((n_ds, n_total), dtype=np.float64)
                opt_ix = [i for i, m in enumerate(equity_mask) if not m]
                equity_ix = [i for i, m in enumerate(equity_mask) if m]
                full[:, opt_ix] = arr
                # Fill equity positions with spot price changes: v_spots_1 - s0
                for idx, eq_col in enumerate(equity_ix):
                    full[:, eq_col] = v_spots_1 - self.s0
            else:
                full = np.zeros(n_total, dtype=np.float64)
                opt_ix = [i for i, m in enumerate(equity_mask) if not m]
                # equity_ix = [i for i, m in enumerate(equity_mask) if m]
                full[opt_ix] = arr
            return full
        #
        # def _to_full(jl_mat_or_vec, n_total, equity_mask, n_ds=None):
        #     arr = np.array(jl_mat_or_vec, dtype=np.float64)
        #     if n_ds is not None:
        #         full = np.zeros((n_ds, n_total), dtype=np.float64)
        #         opt_ix = [i for i, m in enumerate(equity_mask) if not m]
        #         full[:, opt_ix] = arr
        #     else:
        #         full = np.zeros(n_total, dtype=np.float64)
        #         opt_ix = [i for i, m in enumerate(equity_mask) if not m]
        #         full[opt_ix] = arr
        #     return full

        n_ds = len(self.cfg.v_ds_ret)
        self.nlv0_bid = _to_full(nlv0_bid_jl, n_total, equity_mask, n_ds)
        self.nlv0_ask = _to_full(nlv0_ask_jl, n_total, equity_mask, n_ds)
        info(f'PY nlv0_bid: size={self.nlv0_bid.shape} sum={self.nlv0_bid.sum()})')
        info(f'PY nlv0_ask: size={self.nlv0_ask.shape} sum={self.nlv0_ask.sum()})')

        dnlv01_worst_long = _to_full(wl, n_total, equity_mask, n_ds)
        dnlv01_worst_short = _to_full(ws, n_total, equity_mask, n_ds)
        dnlv01_best_long = _to_full(bl, n_total, equity_mask, n_ds)
        dnlv01_best_short = _to_full(bs, n_total, equity_mask, n_ds)

        info(f'PY dnlv01_worst_long: size={dnlv01_worst_long.shape} sum={dnlv01_worst_long.sum()})')
        info(f'PY dnlv01_worst_short: size={dnlv01_worst_short.shape} sum={dnlv01_worst_short.sum()})')
        info(f'PY dnlv01_best_long: size={dnlv01_best_long.shape} sum={dnlv01_best_long.sum()})')
        info(f'PY dnlv01_best_short: size={dnlv01_best_short.shape} sum={dnlv01_best_short.sum()})')

        return dnlv01_worst_long, dnlv01_worst_short, dnlv01_best_long, dnlv01_best_short

    @time_it
    def get_nlv_delta_matrix_old(self, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """slow - need to port to julia and vectorize"""
        self.nlv0_bid = nlv0_bid = np.array([self.ivs0_bid * 100 for o in self.scoped_options])
        # self.nlv0_bid = nlv0_bid = np.array([nlv(o, self.s0, iv(self.ivs0_bid, o, self.s0, self.calc_date0), 1, self.calc_date0) for o in self.scoped_options])
        self.nlv0_ask = nlv0_ask = np.array([self.ivs0_ask * 100 for o in self.scoped_options])
        # self.nlv0_ask = nlv0_ask = np.array([nlv(o, self.s0, iv(self.ivs0_ask, o, self.s0, self.calc_date0), 1, self.calc_date0) for o in self.scoped_options])

        nlv1 = np.array([[nlv(o, self.s0 * ds_ret, iv(self.get_ivs1(ds_ret - 1, alpha), o, self.s0 * ds_ret, self.calc_date1), 1, self.calc_date1) for o in self.scoped_options] for ds_ret in self.cfg.v_ds_ret])

        dnlv01_worst_long = nlv1 - nlv0_ask
        dnlv01_worst_short = nlv1 - nlv0_bid
        dnlv01_best_long = nlv1 - nlv0_bid
        dnlv01_best_short = nlv1 - nlv0_ask

        return dnlv01_worst_long, dnlv01_worst_short, dnlv01_best_long, dnlv01_best_short

    def get_v_spread_pc_credit(self, n):
        return np.full(n, 0.5)

    def get_m_spread_pc_credit(self, n):
        return np.full((len(self.cfg.v_ds_ret), n), 0.5)

    @lru_cache(maxsize=100)
    def _get_v_buy_sell_iv(self, s0: float, calc_date0: date) -> Tuple[np.ndarray, np.ndarray]:
        v_iv0_ask = np.array([iv(self.ivs0_ask, o, s0, calc_date0) for o in self.scoped_options])
        v_iv0_bid = np.array([iv(self.ivs0_bid, o, s0, calc_date0) for o in self.scoped_options])

        v_spread_pc_credit = self.get_v_spread_pc_credit(len(self.scoped_options))
        spread_iv0 = v_iv0_ask - v_iv0_bid

        v_buy_iv = v_iv0_ask - spread_iv0 * v_spread_pc_credit
        v_sell_iv = v_iv0_bid + spread_iv0 * v_spread_pc_credit
        return v_buy_iv, v_sell_iv

    def get_assumed_fill_iv(self, pf: Portfolio, s0: float, calc_date0: date) -> Dict[Option, float]:
        """Presuming for now a mid IV fill, therefore presumed fill is simply the SSVI surfaces IV"""
        v_buy_iv, v_sell_iv = self._get_v_buy_sell_iv(s0, calc_date0)

        scoped_options_idx = {str(o): i for i, o in enumerate(self.scoped_options)}
        dct_buy_ivs = {h.symbol: v_buy_iv[scoped_options_idx[str(h.symbol)]] for h in pf if h.quantity != 0}
        dct_sell_ivs = {h.symbol: v_sell_iv[scoped_options_idx[str(h.symbol)]] for h in pf if h.quantity != 0}

        return {**dct_buy_ivs, **dct_sell_ivs}


def get_estimated_stress_test(tests: List[StressTestDsResult]):
    return next(iter([st for st in tests if st.tag == f'Spread: {Scenario.estimated} alpha: {None}']), None)


def market_snapshot(df: pd.DataFrame, ts: pd.Timestamp, pf: Portfolio) -> Dict[Security, SecurityDataSnap] | None:
    v_ts = [el for el in df.index.get_level_values('ts').unique() if el.date() == ts.date() and el <= ts]
    _df = df.loc[v_ts]
    snaps = {}
    for holding in pf:
        if isinstance(holding.symbol, Option):
            if ts.date() > holding.symbol.optionContract.expiry:
                snaps[holding.symbol] = SecurityDataSnap(holding.symbol, ts, np.nan, np.nan, np.nan)
                continue
            o = Option(holding.symbol.optionContract, ts.date())
            try:
                ps = _df.loc[(slice(None), o.expiry, o.optionContract.strike, o.right)].iloc[-1]
                snaps[o] = SecurityDataSnap(o.symbol, ts, ps['spot'], ps['bid_close'], ps['ask_close'])
            except (IndexError, KeyError) as e:
                spot = _df.iloc[-1]['spot']
                intrinsic_value = o.intrinsic_value(spot)
                warning(f'No market data provided for Symbol={holding.symbol}, Quantity={holding.quantity} up to {ts}. Pricing at intrinsic value: {intrinsic_value}.')
                # min_tenors_iv = _df.loc[(slice(None), o.expiry, o.optionContract.strike, o.right)]
                # o.npv(0.3, spot, ts.date())
                # o.npv(1.0, spot, ts.date())
                snaps[o] = SecurityDataSnap(o.symbol, ts, spot, intrinsic_value, intrinsic_value)
        elif isinstance(holding.symbol, Equity):
            ps = _df.iloc[-1]
            snaps[holding.symbol] = SecurityDataSnap(holding.symbol, ts, ps['spot'])

    return snaps


def get_earnings_release_pf(cfg: EarningsConfig) -> dict:
    """
        Refinement:
        - The ultimate short is exercised 1 day at EOD before release date. Also here can improve, research when the time value is high.
        - Generalize. also investigate portfolios earning the term structure IV crush.

        # Questions:
        # - What's the optimal T-x days to sell near expiry options? Understands how far values change. How intrinsic value changes as release date comes up.
        # - Optimal near/long hedge ratios? Risk / Pnl Trade off... relate risk space and cost to hedge?
        """
    er = EarningsReleaseSimulation(cfg)
    # er.plot_return_vs_model_params()

    # Spot price series1 internal value, eventually weights also...
    if cfg.plot:
        plot_spot(er.df, er.df0, er.exp_jump1, er.release_date, er.ts_pre_release, er.s0, cfg.sym)

    if cfg.run_solver:
        solver_res = er.solve()
        pf = solver_res.pf
    else:
        pf = er.pf

    # presumed_fill_ivs = get_assumed_fill_iv(m_spread_pc_credit, pf.get_holdings(), er.scoped_options, er.ivs0_bid, er.ivs0_ask, er.s0, er.calc_date0)
    # pprint(f'presumed_fill_ivs: {presumed_fill_ivs}')

    if cfg.plot:
        # Stressing the portfolio for a few more dIV level changes for plots
        v_stress_test_ds = er.stress_tests(pf)
        stress_test_ds_estimated = get_estimated_stress_test(v_stress_test_ds)
        plot_stressed_pnl(er, pf, stress_test_ds_estimated.delta_total_across_ds, v_stress_test_ds)

        show(get_chart_nlv_t(er, pf))

        if date.today() > er.release_date:
            show(get_chart_iv_predicted_vs_actual(er))

            score_iv_predicted_vs_actual(er)

        # plot_iv_over_time(pf, df, v_ts_post_release, ts_pre_release, ts_post_release, release_date, sym)

        print('marginal_scaled_objective_by_holding')
        for sec, v in stress_test_ds_estimated.marginal_weighted_objective_by_holding.items():
            q_holding = pf.get(sec, 0)
            print(f'{sec}: {(q_holding * v) / abs(q_holding):.2f}')
        print('-' * 30)

    # print(f'Done earnings release {sym}')
    # pprint_spread_usd_by_sym(m_dnlv01_best_buy, m_dnlv01_worst_buy, m_dnlv01_best_sell, m_dnlv01_worst_sell, pf, scoped_options)

    return pf.holdings


def get_chart_nlv_t(er: EarningsReleaseSimulation, pf: Portfolio, delta_hedge=False, scenarios: List[Scenario] = None) -> go.Figure:
    """Needs delta hedging to be realistic...."""
    scenarios = scenarios or [Scenario.best, Scenario.mid, Scenario.worst]

    fig = go.Figure()
    for scenario in scenarios:
        v_pnl: List[NLVSnapshot] = er.v_nlv(pf, delta_hedge=delta_hedge, scenario=scenario)
        x = [el.ts for el in v_pnl]
        y = [el.nlv for el in v_pnl]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f'PL {scenario}',
                                 marker=dict(size=4), line=dict(width=1)))
    fig.update_layout(xaxis_title='ts', yaxis_title='NLV')
    return fig


def get_chart_iv_predicted_vs_actual(er: EarningsReleaseSimulation, min_max_mny=(-0.3, 0.3)) -> go.Figure:
    """
    Each tenor is a line. One for predicted, one for realized.
    predicted (MID) will be a smooth line. realized will be scatter throughout the day
    x is ln(K/F), y is iv.

    one for call, one for put
    """
    ivs1 = er.get_ivs1(0, alpha=0.1)
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Call', 'Put'])
    for i, (right, df_right) in enumerate(er.df1.groupby('right')):
        ix_col = i + 1
        n_colors = len(er.expiries)
        color_scale = px.colors.sample_colorscale("rainbow", [n / (n_colors - 1) for n in range(n_colors)])

        for k, tenor_dt in enumerate(er.expiries):

            if not ivs1.is_calibrated_slice(tenor_dt):
                continue

            # Actual
            _df = df_right[df_right.index.get_level_values('expiry') == tenor_dt].sort_values('moneyness_fwd_ln')

            x = (get_moneyness_fwd_ln(
                er.equity,
                _df.index.get_level_values('strike').astype(float).values,
                _df['spot'].values.astype(float),
                _df['tenor'].values,
                date_to_sod(er.calc_date1)
                ))
            ix_scope = (x > min_max_mny[0]) & (x < min_max_mny[1])
            x = x[ix_scope]
            y = _df['mid_iv'][ix_scope]
            ix_sample = sorted(np.random.choice(range(len(x)), min(200, len(x)), replace=False))
            x = x[ix_sample]
            y = y.iloc[ix_sample]

            # assert all(sorted(x) == x)

            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'{right} {tenor_dt} Actual', marker=dict(size=3, color=color_scale[k])), row=1, col=ix_col)

            # Predicted
            y_pred = ivs1.iv(x, tenor_dt, date_to_sod(er.calc_date1))
            fig.add_trace(go.Scatter(x=x, y=y_pred, mode='lines+markers', name=f'{right} {tenor_dt} Predicted', marker=dict(size=3, color=color_scale[k])), row=1, col=ix_col)

            # Need MAE, RMSE by right and tenor...

    return fig


def score_iv_predicted_vs_actual(er: EarningsReleaseSimulation, min_max_mny=(-0.5, 0.5)):
    v_y = []
    v_y_pred = []
    v_weight = []

    for right, df_right in er.df1.groupby('right'):
        for tenor_dt in er.expiries:
            if not er.get_ivs1(0).is_calibrated_slice(tenor_dt):
                continue
            # Actual
            _df = df_right[df_right.index.get_level_values('expiry') == tenor_dt].sort_values('strike')

            x = get_moneyness_fwd_ln(er.equity, _df.index.get_level_values('strike').astype(float).values, _df['spot'].values.astype(float), _df['tenor'].values, date_to_sod(er.calc_date1))
            ix_scope = (x > min_max_mny[0]) & (x < min_max_mny[1])
            x = x[ix_scope]
            y = _df['mid_iv'][ix_scope]
            weight = _df['vega_mid_iv'].values[ix_scope]

            # Predicted
            y_pred = er.get_ivs1(0).iv(x, tenor_dt, date_to_sod(er.calc_date1))

            v_y.extend(y)
            v_y_pred.extend(y_pred)
            v_weight.extend(weight)

    return score(v_y, v_y_pred, v_weight)


def get_f_weight_ds():
    """Rewarding risk reduction around implied jump dS move more than large dS moves"""
    # f_weight_ds = get_f_weight_ds_ret(implied_jump * 2, plot=cfg.plot)
    # f_weight_ds = get_f_weight_ds_ret(0.1, plot=cfg.plot)
    f_weight_ds = None
    return f_weight_ds


def pprint_spread_usd_by_sym(m_dnlv01_best_buy, m_dnlv01_worst_buy, m_dnlv01_best_sell, m_dnlv01_worst_sell, pf, scoped_options):
    m_dnlv01_spread_buy = m_dnlv01_best_buy - m_dnlv01_worst_buy
    m_dnlv01_spread_sell = m_dnlv01_best_sell - m_dnlv01_worst_sell

    v_spread_buy = np.maximum(np.array([1 if pf.portfolio.get(o, 0) > 0 else 0 for o in scoped_options]), 0) * m_dnlv01_spread_buy[10]
    v_spread_sell = np.minimum(np.array([-1 if pf.portfolio.get(o, 0) < 0 else 0 for o in scoped_options]), 0) * m_dnlv01_spread_sell[10]
    sym_q_buy = [(scoped_options[i], spread_usd) for i, spread_usd in enumerate(v_spread_buy) if spread_usd != 0]
    sym_q_sell = [(scoped_options[i], spread_usd) for i, spread_usd in enumerate(v_spread_sell) if spread_usd != 0]

    pprint(dict(sym_q_buy))
    pprint(dict(sym_q_sell))


def iv(ivs: IVSurface, o: Option, s: float, calc_date: date) -> float:
    if isinstance(o, Option):
        return ivs.iv(
            get_moneyness_fwd_ln(
                o.equity,
                o.strike,
                s,
                get_tenor(o.expiry, calc_date),
                date_to_sod(calc_date)
            ),
            o.expiry,
            calc_date
        )[0]
    else:
        # We can also assign an IV to the underlying, but that's not of interest right. Needs some function across options..
        return 0


# @time_it
def get_v_delta_bid_ask(s0: float, ivs0_bid: IVSurface, ivs0_ask: IVSurface, scoped_options: List[Option], calc_date0: date) -> Tuple[np.ndarray, np.ndarray]:
    v_delta0_bid = np.array([delta(o, s0, iv(ivs0_bid, o, s0, calc_date0), 1, calc_date0) for o in scoped_options])
    v_delta0_ask = np.array([delta(o, s0, iv(ivs0_ask, o, s0, calc_date0), 1, calc_date0) for o in scoped_options])
    return v_delta0_bid, v_delta0_ask


def get_estimate_m_dnlv_buy_sell(m_dnlv01_worst_buy, m_dnlv01_best_buy, m_dnlv01_worst_sell, m_dnlv01_best_sell, m_spread_pc_credit: np.ndarray):
    m_spread_buy = m_dnlv01_best_buy - m_dnlv01_worst_buy
    m_spread_sell = m_dnlv01_best_sell - m_dnlv01_worst_sell

    m_dnlv01_buy_estimated = m_dnlv01_worst_buy + m_spread_buy * m_spread_pc_credit
    m_dnlv01_sell_estimated = m_dnlv01_worst_sell + m_spread_sell * m_spread_pc_credit
    # return m_dnlv01_worst_buy, m_dnlv01_worst_sell
    return m_dnlv01_buy_estimated, m_dnlv01_sell_estimated


def get_scoped_expiries(expiries: List[date]) -> List[date]:
    # Drop some weeklies to reduce scope. Pick 2 closest expiries. Then last of each month
    scoped_expiries = expiries[:2]
    for i, exp in itertools.groupby(expiries, lambda x: x.month):
        months_last_exp = list(exp)[-1]
        if months_last_exp.month not in [dt.month for dt in scoped_expiries]:
            scoped_expiries.append(months_last_exp)
    # print(f'Scoped expiries: #{len(scoped_expiries)} out of #{len(expiries)}, {scoped_expiries}')
    return scoped_expiries


def median(ps: pd.Series, skipna=True):
    _ps = ps[~ps.isna()] if skipna else ps
    if len(_ps) == 0:
        return np.nan
    return sorted(_ps)[len(_ps) // 2]


def get_df_median(df0):
    """ps0 where mid_iv == median"""
    ps_lst = []
    for expiry, s_df in df0.groupby('expiry'):
        for strike, s_df2 in s_df.groupby('strike'):
            for right, s_df3 in s_df2.groupby('right'):
                mid_iv = s_df3['mid_iv']
                mid_iv_median = median(mid_iv)
                df_median = s_df3.loc[mid_iv == mid_iv_median]
                if len(df_median) > 0:
                    ps_lst.append(df_median.iloc[0])
                else:
                    print('None el')
    df = pd.concat(ps_lst, axis=1).transpose()
    df.index = df.index.set_names(['ts', 'expiry', 'strike', 'right'])
    return df


def title_stressed_pnl_plot(pf: Portfolio, deltaTotalPf0, ts_pre_release):
    # Title subscript of holdings
    sym = pf.underlying
    holdings_txt = ''
    for k, h in enumerate(pf):
        holdings_txt += f'{h.symbol} {h.quantity}, '
        if k > 0 and k % 6 == 0:
            holdings_txt += '<br>'
    holdings_txt += f' DeltaTotalPf0Total: {deltaTotalPf0:.2f}'
    return f'{sym} dS dIV dPL ts0: {ts_pre_release}<br><sup>{holdings_txt}</sup>'


def get_ds_dpnl_traces(s0, stress_tests: List[StressTestDsResult], marker_size=4):
    """Traces using the iv crush model."""
    for stress_test_ds in stress_tests:
        x = [100 * (x - 1) for x in stress_test_ds.ds_dnlv.keys()]
        # y = [dnlv + s0 * (ds_ret - 1) * -stress_test_ds.delta_total_across_ds for ds_ret, dnlv in stress_test_ds.ds_dnlv.items()]
        y = [dnlv for ds_ret, dnlv in stress_test_ds.ds_dnlv.items()]
        yield go.Scatter(x=x, y=y, mode='markers', name=f'dIV: model {stress_test_ds.tag} fill', marker=dict(size=marker_size))


def get_chart_ds_dspnl(s0: float, stress_tests_ds: List[StressTestDsResult], marker_size=4):
    fig = go.Figure()
    for t in get_ds_dpnl_traces(s0, stress_tests_ds, marker_size=marker_size):
        fig.add_trace(t)
    fig.update_layout(xaxis_title=f'dS %', yaxis_title='dPL')
    return fig


def plot_stressed_pnl(er: EarningsReleaseSimulation, pf: Portfolio, delta_total_pf0, v_stress_test_ds: List[StressTestDsResult], marker_size=4):
    sym = pf.underlying

    fig_ds_iv = get_chart_ds_dspnl(er.s0, v_stress_test_ds, marker_size=marker_size)
    title = title_stressed_pnl_plot(pf, delta_total_pf0, er.ts_pre_release.isoformat())
    info(title)

    show(fig_ds_iv, f'''{sym}_{er.release_date.strftime('%y%m%d')}_figdSdIV_iteration.html''')


def median_iv(df, expiry, strike, right, col_nm='fill_iv'):
    return df.loc[(slice(None), expiry, strike, right), col_nm].median()


def plot_spot(df, df0, exp_jump1, release_date, ts_pre_release, s0, sym, marker_size=4):
    figSpotTimeValue = make_subplots(rows=2, cols=1, subplot_titles=['Spot', 'Time Value'])
    figSpotTimeValue.add_trace(go.Scatter(x=df.index.get_level_values('ts'), y=df['spot'], mode='lines', name='Spot'), row=1, col=1)
    figSpotTimeValue.add_vline(x=datetime.fromisoformat(release_date.isoformat()) + timedelta(hours=16), line_dash='dash', line_color='red', row=1, col=1)
    figSpotTimeValue.update_layout(title=f'{sym} Spot', xaxis_title='Time', yaxis_title='Spot')

    ############## Extrinsic value over time for near expiry - IS WRONG
    strikes = pd.Series(np.unique(df0.loc[(ts_pre_release, exp_jump1, slice(None), slice(None))].index.get_level_values('strike').astype(float)))
    strikes_atm = strikes.iloc[(strikes - s0).abs().sort_values().head(2).index].values
    for right, s_df in df.loc[slice(None), exp_jump1, strikes_atm, slice(None)].groupby('right'):
        strike = s_df.index.get_level_values('strike').astype(float)
        spot = s_df['spot']
        if right == 'call':
            intrinsic_value = spot - strike
            time_value = s_df['mid_price'] - intrinsic_value
        elif right == 'put':
            intrinsic_value = strike - spot
            time_value = s_df['mid_price'] - intrinsic_value
        else:
            raise ValueError('Right must be call or put')
        figSpotTimeValue.add_trace(go.Scatter(x=s_df.index.get_level_values('ts'), y=time_value, mode='markers', name=f'{right}', marker=dict(size=marker_size)), row=2, col=1)
        figSpotTimeValue.add_vline(x=datetime.fromisoformat(release_date.isoformat()) + timedelta(hours=16), line_dash='dash', line_color='red', row=2, col=1)

    show(figSpotTimeValue, f'{sym}_{release_date.strftime("%y%m%d")}_spot_extrinsic_value.html')


def option_in_pf_scope(o: Option, cfg: EarningsConfig, calcDate0, s0, p0, scoped_expiries=None, exclude: List[Option] = None, pf: Portfolio = None):
    if pf and str(o) in [str(h) for h in pf.holdings.keys()]:
        return True

    if exclude and o in exclude:
        return False

    # Want large tenors as their IV decreases less after release date.
    if scoped_expiries and o.expiry not in scoped_expiries:
        return False

    if get_tenor(o.expiry, calcDate0) < cfg.min_tenor:
        return False

    if get_tenor(o.expiry, calcDate0) > cfg.max_tenor:
        return False

    moneyness = float(o.strike) / s0
    if (cfg.moneyness_limits and
            (moneyness < cfg.moneyness_limits[0] or moneyness > cfg.moneyness_limits[1])):
        return False

    iv0 = o.iv(p0, s0, calcDate0)
    if iv0 == 0 or np.isnan(iv0):
        return False

    if cfg.abs_delta_limits:
        delta = abs(o.delta(iv0, s0, calcDate0))
        if delta < cfg.abs_delta_limits[0] or delta > cfg.abs_delta_limits[1]:
            return False

    return True


def get_v_iv_vega(df_trade, df_quote, o: Option, q: int, sub_sample_size: int, calc_date: date, quote_col_nm, trade_col_nm='fill_iv', scenario: Scenario | str = Scenario.worst):
    try:
        v_ix = df_trade.loc[(slice(None), o.expiry, o.optionContract.strike, o.right)].index
        v_ix = np.random.choice(v_ix, sub_sample_size) if len(v_ix) > sub_sample_size else v_ix
        v_iv_vega = np.array(
            [(ps[trade_col_nm], o.vega(ps[trade_col_nm], ps['spot'], calc_date)) for ix, ps in df_trade.loc[(v_ix, o.expiry, o.optionContract.strike, o.right)].iterrows()])
        if len(v_iv_vega) == 0:
            raise ValueError(f'No trade data for {o} in df0_trade')
    except (KeyError, ValueError) as e:
        print(f'No trade data for {o}: {q} in df0_trade. Using {scenario} quote data instead. {e}')
        v_ix = df_quote.loc[(slice(None), o.expiry, o.optionContract.strike, o.right)].index
        v_ix = np.random.choice(v_ix, sub_sample_size) if len(v_ix) > sub_sample_size else v_ix
        v_iv_vega = np.array(
            [(ps[quote_col_nm], o.vega(ps[quote_col_nm], ps['spot'], calc_date)) for ix, ps in df_quote.loc[(v_ix, o.expiry, o.optionContract.strike, o.right)].iterrows()])
    return v_iv_vega


def simulate_pf_pnl_kalman(pf: Portfolio, df0_quote, df0_trade, df1_quote, df1_trade, sample_size=500, scenario=Scenario.worst) -> np.ndarray:
    """
    Don't use mid prices, but actual fills. If none available, worst quote.

    For each day and for each security, n prices are sampled.
    There'd be n**2 dnlv combinations.

    Plot actual trade fills during that period, instead of limit. My fill estimates have to match actual pass fills!
    Simulate better and worse fill depending on liquidity.
    Worst/Best/Mid
    """
    col_nm_fill_iv = 'fill_iv'
    calc_date0 = df0_quote.index.get_level_values('ts')[0].date()
    calc_date1 = df1_quote.index.get_level_values('ts')[0].date()
    sub_sample_size = int(np.ceil(sample_size ** 0.5))

    v_dlnv = np.zeros(sample_size)
    for sec, q in pf.items():
        if isinstance(sec, Equity):
            p0 = df0_quote['spot'].iloc[-1]
            p1 = df1_quote['spot'].iloc[0]
            v_dlnv += q * (p1 - p0)
        else:
            o = sec

            # Buying at t0 if q > 0
            quote_iv_col_nm = 'ask_iv' if q > 0 else 'bid_iv' if scenario == Scenario.worst else 'mid_iv'
            iv_col_nm = kalman_my_bid_col if q > 0 else kalman_my_ask_col
            _df0_trade = df0_trade[~df0_trade[iv_col_nm].isna()]
            if q > 0:
                _df0_trade = _df0_trade[_df0_trade[kalman_my_bid_col] >= _df0_trade[col_nm_fill_iv]]
            else:
                _df0_trade = _df0_trade[_df0_trade[kalman_my_ask_col] <= _df0_trade[col_nm_fill_iv]]
            v_iv_vega0 = get_v_iv_vega(_df0_trade, df0_quote, o, q, sub_sample_size, calc_date0, quote_col_nm=quote_iv_col_nm, trade_col_nm=iv_col_nm, scenario=scenario)
            v_iv_vega0 = np.array([(iv, vega) for iv, vega in v_iv_vega0 if not any((np.isnan(v) for v in (iv, vega)))])
            if len(v_iv_vega0) == 0:
                print(f'No data for v_iv_vega0 {o}: {q}. Assume intrinsic value. 0 IV => 0 vega.')
                v_iv_vega0 = np.array([(0, 0)])

            # Selling at t0 if q > 0
            quote_iv_col_nm = 'ask_iv' if q < 0 else 'bid_iv' if scenario == Scenario.worst else 'mid_iv'
            iv_col_nm = kalman_my_bid_col if q < 0 else kalman_my_ask_col
            _df1_trade = df1_trade[~df1_trade[iv_col_nm].isna()]
            if q < 0:
                _df1_trade = _df1_trade[_df1_trade[kalman_my_bid_col] >= _df1_trade[col_nm_fill_iv]]
            else:
                _df1_trade = _df1_trade[_df1_trade[kalman_my_ask_col] <= _df1_trade[col_nm_fill_iv]]
            v_iv_vega1 = get_v_iv_vega(df1_trade, df1_quote, o, q, sub_sample_size, calc_date1, quote_col_nm=quote_iv_col_nm, trade_col_nm=iv_col_nm, scenario=scenario)
            v_iv_vega1 = np.array([(iv, vega) for iv, vega in v_iv_vega1 if not any((np.isnan(v) for v in (iv, vega)))])
            if len(v_iv_vega1) == 0:
                print(f'No data for v_iv_vega1 {o}: {q}. Ignoring this. one. Assume intrinsic value. 0 IV => 0 vega.')
                v_iv_vega1 = np.array([(0, 0)])

            tmp = [100 * q * (iv1 - iv0) * (vega0 + vega1) / 2 for (iv0, vega0), (iv1, vega1) in itertools.product(v_iv_vega0, v_iv_vega1) if
                   not any((np.isnan(v) for v in (iv1, iv0, vega0, vega1)))]
            if len(tmp) == 0:
                print(f'No data for {o}: {q}. Ignoring this. one. Bad. Should assume cannot buy or sell, hence market value simply.')
                continue
            v_dlnv += np.random.choice(tmp, sample_size)

    return v_dlnv


def npv(sec: Union[Option, Equity], s0, iv: float, calc_date: date) -> float:
    return sec.npv(iv, s0, calc_date) if isinstance(sec, Option) else s0


def nlv(sec: Union[Option, Equity], s0, iv: float, q, calc_date: date) -> float | None:
    if iv is None or np.isnan(iv):
        return None

    if isinstance(sec, Option):
        return sec.npv(iv, s0, calc_date) * q * 100
    else:
        return q * s0


def delta(sec: Union[Option, Equity], s0, iv: float, q, calc_date: date) -> float:
    if isinstance(sec, Option):
        return sec.delta(iv, s0, calc_date) * q * 100
    else:
        return q * s0


@lru_cache(maxsize=None)
def get_option(underlying: str, expiry: date, strike: Decimal, right: str, calc_date0: date) -> Option:
    return Option(OptionContract('', underlying, expiry, strike, right), calc_date0)


@lru_cache(maxsize=1000)
def d_iv_power_law(d_iv_tenor_1, expiry: date, calc_date: date, ts_time_power=0.5):
    tenor_ = get_tenor(expiry, calc_date)
    if tenor_ <= 0:
        return d_iv_tenor_1
    return d_iv_tenor_1 / tenor_ ** ts_time_power


def add_plot_term_structure(df, calc_date: date, fig, row, cfg: EarningsConfig, dIV_levels=(0, -0.01), ts_time_power=0.5, name_prefix=''):
    for dIV_level in dIV_levels:
        x = []
        y = []
        for expiry, ss_df in df.groupby('expiry'):
            tenor_ = get_tenor(expiry, calc_date)
            ts = ss_df.index.get_level_values('ts').unique()[0]
            atm_iv_val = atm_iv(ss_df.loc[ts], expiry, ss_df.iloc[0]['spot'])
            if atm_iv_val <= 0 or np.isnan(atm_iv_val):
                continue

            if dIV_level is None:  # predicted
                df_mny_t = pd.DataFrame([[get_moneyness_fwd_ln(Equity(cfg.sym), ss_df.iloc[0]['strike'], ss_df.iloc[0]['spot'], tenor_, calc_date), tenor_]],
                                        columns=['moneyness_fwd_ln', 'tenor'])
                d_iv_ratio = cfg.earnings_iv_drop_regressor.predict(df_mny_t)[0][0]
                dIV_tenor = atm_iv_val * d_iv_ratio
                # dIV_tenor = ss_df.loc[(ts, expiry, slice(None), slice(None)), 'd_iv_ratio'].unique().mean() * atm_iv_val
            else:  # fixed simulation
                dIV_tenor = d_iv_power_law(dIV_level, expiry, calc_date, ts_time_power=ts_time_power)
            iv = atm_iv_val + dIV_tenor
            if iv > 5:
                pass

            x.append((expiry - df.index.get_level_values('ts').min().date()).days)
            y.append(iv)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f'{name_prefix}IV Term Structure @ {calc_date} dIV={dIV_level}', marker=dict(size=3)), row=row, col=1)


def add_plot_iv(df, ts, pf, fig, row, col, marker_size_iv=3, marker_size_holdings=8):
    calcDate = ts.date()
    holding_expiries = list(set([sec.expiry for sec in pf.portfolio.keys() if isinstance(sec, Option)]))

    for expiry in holding_expiries:
        s_df = df.loc[(ts, expiry, slice(None), slice(None))]

        # Plot IVs
        for right, ss_df in s_df.groupby(level='right'):
            sample_df = ss_df[ss_df['mid_iv'].fillna(0) != 0]
            fig.add_trace(go.Scatter(x=sample_df['moneyness'], y=sample_df['mid_iv'], mode='lines+markers', name=f'IV {expiry} {right}', marker=dict(size=marker_size_iv)), row=row,
                          col=col)

    right_symbol_map = {'call': 'x-dot', 'put': 'circle-dot'}
    for right, s_df in df.loc[ts].groupby(level='right'):
        holdings = {sec: q for sec, q in pf.portfolio.items() if isinstance(sec, Option) and sec.right == right}
        s = s_df.iloc[0]['spot']
        x_moneyness = [float(sec.optionContract.strike) / s for sec in holdings.keys()]
        y_iv = [sec.iv(s_df.loc[(sec.expiry, sec.optionContract.strike, sec.right), 'mid_price'], s, calcDate) for sec in holdings.keys()]
        # if 0 in y_iv:
        #     raise ValueError(f'Zero IV for {right} {ts} {x_moneyness} {y_iv}')

        fig.add_trace(go.Scatter(
            x=x_moneyness, y=y_iv, mode='markers', name=f'IV Holdings {right}',
            text=[f'{str(sec)}: {q}' for sec, q in holdings.items()],
            marker=dict(size=marker_size_holdings, symbol=right_symbol_map[right])), row=row, col=col)


def trial_plot_quantile_surfaces():
    cfg = EarningsConfig('ORCL', date(2024, 3, 11), plot=True, plot_last=True,
                         moneyness_limits=(0.8, 1.2), abs_delta_limits=(0.1, 0.9), seq_ret_threshold=0.002,
                         min_tenor=0.0, max_tenor=1, add_equity_holdings=True, portfolio=Portfolio(), run_solver=True,
                         n_contracts=20, earnings_iv_drop_regressor_model_name_version='v1d_2024-08-27',
                         )
    er = EarningsReleaseSimulation(cfg)

    fig = go.Figure()
    for alpha in [0.9, None, 0.1]:
        ivs = er.get_ivs1(0, alpha)
        x = np.linspace(-0.2, 0.2, 100)
        y = []
        z = []
        for tenor_dt in er.expiries:
            z += list(ivs.iv(x, tenor_dt, er.calc_date1))
            y += [get_tenor(tenor_dt, er.calc_date1)] * len(x)
        fig.add_trace(go.Scatter3d(x=x.tolist() * len(er.expiries), y=y, z=z, mode='markers', marker=dict(size=2), name=f'{alpha}' if alpha else 'mid'))

        ivs.plot_model_params(er.calc_date1, f'alpha_{alpha}_model_params.html')
    show(fig)


def run():
    # test_plot_quantile_surfaces()

    tickers = 'PEP'
    for take in [-1]:
        for sym in tickers.split(','):
            release_date = EarningsPreSessionDates(sym)[take]
            print(release_date)
            pf = Portfolio({Option(OptionContract.from_ib_symbol(s), release_date): q for s, q in {
                # FDX Jun26'26 347.5 PUT
            }.items()})
            # pf.add_holding(Equity(sym), -334)

            # cfg = EarningsConfig(sym, release_date, plot=True, plot_last=True,
            #                      moneyness_limits=(0.0, 3), abs_delta_limits=(0.0, 1), seq_ret_threshold=0.002,
            #                      min_tenor=0.0, max_tenor=2, add_equity_holdings=False, portfolio=pf, run_solver=False,
            #                      n_contracts=10,
            #                      )
            # Pre Go-Live money to be earned estimates...
            cfg = EarningsConfig(sym, release_date, plot=False, plot_last=False,
                                 moneyness_limits=(0.75, 1.25), abs_delta_limits=(0.05, 0.95), seq_ret_threshold=0.002,
                                 min_tenor=0.0, max_tenor=1, add_equity_holdings=False, portfolio=pf, run_solver=True,
                                 n_contracts=20,
                                 )
            get_earnings_release_pf(cfg)


if __name__ == '__main__':
    run()
