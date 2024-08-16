import os
import pickle
import copy
import datetime
import multiprocessing
import time
import numpy as np
import pandas as pd
import QuantLib as ql
import warnings

from itertools import chain
from typing import List, Dict, Tuple
from functools import partial
from dataclasses import dataclass, asdict
from scipy.optimize import basinhopping
from scipy.stats import iqr
from sklearn.neighbors import KernelDensity
from scipy.interpolate import LinearNDInterpolator

from shared.paths import Paths
from options.helper import plot_vol_surface, repair_prices, str2ql_option_right, make_eu_option, np1d_from_df, plot_surface, delta_bsm, \
    set_ql_calculation_date, find_loc_every_x_pc, to_ql_dt
from options.typess.enums import TickType, SecurityType, Resolution
from options.typess.equity import Equity
from options.typess.option_contract import OptionContract
from options.typess.option import Option

warnings.filterwarnings('ignore')


@dataclass
class VolSurfaceData:
    strikes: np.array
    tenors: np.array
    volatilities: np.array


@dataclass
class SurfaceData:
    """May want to use 1D arrays to support non-rectangular surfaces"""
    x: np.array
    y: np.array
    z: np.array  # of shape (len(x), len(y))
    spot: float = 0.0


@dataclass
class HestonParams:
    v0: float
    kappa: float
    theta: float
    rho: float
    sigma: float

    def params(self):
        return self.v0, self.kappa, self.theta, self.sigma, self.rho

    def feller(self):
        return 2 * self.kappa * self.theta - self.sigma ** 2


@dataclass
class CalibrationParams:
    ts: datetime.datetime
    spot: float
    f_min_strike: float
    f_max_strike: float
    option_right: str
    min_dte: int
    heston_parameter_bounds: list
    initial_heston_params: HestonParams
    rate: float = 0.0
    dividend_yield: float = 0.0
    calendar: object = None
    day_count: object = None
    yield_ts: object = None
    dividend_ts: object = None

    def to_serializable(self):
        cp = copy.copy(self)
        cp.calendar = None
        cp.day_count = None
        cp.yield_ts = None
        cp.dividend_ts = None
        return cp

    def load_unserializable_objects(self):
        calculation_date = self.ts.date()
        calculation_date_ql = ql.Date(calculation_date.day, calculation_date.month, calculation_date.year)
        self.calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        self.day_count = ql.Actual365Fixed()
        self.yield_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date_ql, self.rate, self.day_count))
        self.dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date_ql, self.dividend_yield, self.day_count))

    def __post_init__(self):
        self.load_unserializable_objects()


@dataclass
class SurfaceFrameData:
    df: pd.DataFrame
    calibration_params: CalibrationParams

    def strikes(self):
        return sorted(set(self.df['strike']))

    def expiries(self):
        return sorted(set(self.df['expiry']))

    def ah_vol_surface(self) -> ql.AndreasenHugeVolatilityAdapter:
        self.calibration_params.load_unserializable_objects()
        spot_quote = ql.QuoteHandle(ql.SimpleQuote(self.calibration_params.spot))
        calibration_set = get_calibration_set(self.df, self.calibration_params)
        ah_vol_interpolation = ql.AndreasenHugeVolatilityInterpl(calibration_set, spot_quote, self.calibration_params.yield_ts,
                                                                 self.calibration_params.dividend_ts)
        ah_vol_surface = ql.AndreasenHugeVolatilityAdapter(ah_vol_interpolation)
        return ah_vol_surface

    def to_serializable(self):
        sfd = copy.copy(self)
        sfd.calibration_params = sfd.calibration_params.to_serializable()
        return sfd


@dataclass
class HestonCalibrationResult:
    ts: datetime.datetime
    params: HestonParams
    error: float
    surface_frame_data: SurfaceFrameData
    calibration_params: CalibrationParams
    surface_frame_data_raw: SurfaceFrameData


def vol_surface_data2vega_surface_data(surface_frame_data: SurfaceFrameData, calibration_params: CalibrationParams) -> SurfaceData:
    ah_vol_surface = surface_frame_data.ah_vol_surface()
    strikes = surface_frame_data.strikes()
    expiries = [e for e in surface_frame_data.expiries() if e < ah_vol_surface.maxDate().to_date()]
    tenors = np.array([(dt - calibration_params.ts.date()).days / 365 for dt in expiries])
    vega_srf = np.zeros((len(expiries), len(strikes)))
    for i, expiry in enumerate(expiries):
        for j, strike in enumerate(strikes):
            vol = ah_vol_surface.blackVol(ql.Date(expiry.day, expiry.month, expiry.year), strike)
            vega_srf[i, j] = vega_bsm(strike, expiry, vol, calibration_params)
    return SurfaceData(strikes, tenors, vega_srf, calibration_params.spot)


def gen_df_q(quotes, ps_spot, calibration_params: CalibrationParams):
    calculation_ts = calibration_params.ts

    dfs = []
    for c, df in quotes.items():
        if c.right != calibration_params.option_right or c.expiry <= calculation_ts.date():
            # Filtering for calls only. Calibrating each option right's surface separately
            continue
        # df = df[~df.index.duplicated()]
        try:
            ps = df.loc[calculation_ts]
        except KeyError:
            if len(df.index) > 0 and df.index[0] < calculation_ts < df.index[-1]:
                print(f'KeyError: {c} - {calculation_ts}')
            continue

        ps['mid_close'] = (ps['bid_close'] + ps['ask_close']) / 2
        ps.name = c
        dfs.append(ps)
    df_quotes = pd.concat(dfs, axis=1).transpose()
    # print(df_quotes.shape)

    spot = ps_spot.loc[calculation_ts]
    min_strike = spot * calibration_params.f_min_strike
    max_strike = spot * calibration_params.f_max_strike

    df_quotes['strike'] = [float(x.strike) for x in df_quotes.index]
    df_quotes['expiry'] = [x.expiry for x in df_quotes.index]
    df_quotes['tenor'] = (df_quotes['expiry'] - calculation_ts.date()).apply(lambda x: x.days / 365)
    df_quotes['mid_price_underlying'] = spot
    df_quotes['mid_iv'] = [Option(ix).iv(price, spot, calculation_ts.date()) for ix, price in
                           df_quotes['mid_close'].items()]

    df_quotes = df_quotes[(df_quotes['strike'] > min_strike) & (df_quotes['strike'] < max_strike)]
    df_quotes = df_quotes[
        (df_quotes['expiry'] - calculation_ts.date()).apply(lambda x: x.days) > calibration_params.min_dte]
    # print(df_quotes.isna().sum())
    return df_quotes


def get_andreasen_vol_srf(df_q_r, calibration_params: CalibrationParams, plot=False):
    sfd = SurfaceFrameData(df_q_r, calibration_params)
    ah_vol_surface = sfd.ah_vol_surface()

    if plot:
        min_strike = calibration_params.f_min_strike
        max_strike = calibration_params.f_max_strike
        tenors = np.array([(dt - calibration_params.ts.date()).days / 365 for dt in sorted(set(df_q_r['expiry']))])
        plot_vol_surface(ah_vol_surface, funct='blackVol', plot_years=np.arange(tenors.min(), tenors.max(), 0.05),
                         plot_strikes=np.arange(min_strike, max_strike, 1))
    return ah_vol_surface


def get_andreasen_local_vol_srf(df_q_r, calibration_params: CalibrationParams, plot=False):
    spotQuote = ql.QuoteHandle(ql.SimpleQuote(calibration_params.spot))
    tenors = np.array([(dt - calibration_params.ts.date()).days / 365 for dt in sorted(set(df_q_r['expiry']))])

    calibrationSet = get_calibration_set(df_q_r, calibration_params)
    ahInterpolation = ql.AndreasenHugeVolatilityInterpl(calibrationSet, spotQuote, calibration_params.yield_ts,
                                                        calibration_params.dividend_ts)
    ahLocalSurface = ql.AndreasenHugeLocalVolAdapter(ahInterpolation)
    if plot:
        min_strike = calibration_params.f_min_strike
        max_strike = calibration_params.f_max_strike
        plot_vol_surface(ahLocalSurface, funct='localVol', plot_years=np.arange(tenors.min(), tenors.max(), 0.05),
                         plot_strikes=np.arange(min_strike, max_strike, 1))
    return ahLocalSurface


def get_calibration_set(df_q_r, calibration_params: CalibrationParams):
    # calculation_ts = calibration_params.ts
    calibrationSet = ql.CalibrationSet()
    # calculation_date = ql.Date(calculation_ts.day, calculation_ts.month, calculation_ts.year)
    for iv, expiry, strike in df_q_r[['mid_iv', 'expiry', 'strike']].values:
        if np.isnan(iv):
            continue
        payoff = ql.PlainVanillaPayoff(str2ql_option_right(calibration_params.option_right), strike)
        #  calibration_params.calendar.advance(calculation_date, ql.Period(dte(calculation_ts.date(), expiry), ql.Days))
        exercise = ql.EuropeanExercise(to_ql_dt(expiry))
        calibrationSet.push_back((ql.VanillaOption(payoff, exercise), ql.SimpleQuote(iv)))
    return calibrationSet


def dte(calculation_date, expiry):
    return (expiry - calculation_date).days


def setup_helpers(engine, expiration_dates, strikes,
                  data, ref_date, spot, yield_ts,
                  dividend_ts, calendar):
    heston_helpers = []
    grid_data = []
    for i, date in enumerate(expiration_dates):
        for j, s in enumerate(strikes):
            t = (date - ref_date)
            p = ql.Period(t, ql.Days)
            vols = data[i][j]
            helper = ql.HestonModelHelper(
                p, calendar, spot, s,
                ql.QuoteHandle(ql.SimpleQuote(vols)),
                yield_ts, dividend_ts)
            helper.setPricingEngine(engine)
            heston_helpers.append(helper)
            grid_data.append((date, s))
    return heston_helpers, grid_data


def cost_function_generator(model, helpers, norm=False):
    def cost_function(params):
        params_ = ql.Array(list(params))
        model.setParams(params_)
        error = [h.calibrationError() for h in helpers]
        if norm:
            return np.sqrt(np.sum(np.abs(error)))
        else:
            return error

    return cost_function


def calibration_report(helpers, grid_data, detailed=False) -> float:
    avg = 0.0
    if detailed:
        print("%15s %25s %15s %15s %20s" % (
            "Strikes", "Expiry", "Market Value",
            "Model Value", "Relative Error (%)"))
        print("=" * 100)
    for i, opt in enumerate(helpers):
        err = (opt.modelValue() / opt.marketValue() - 1.0)
        date, strike = grid_data[i]
        if detailed:
            print("%15.2f %25s %14.5f %15.5f %20.7f " % (
                strike, str(date), opt.marketValue(),
                opt.modelValue(),
                100.0 * (opt.modelValue() / opt.marketValue() - 1.0)))
        avg += abs(err)
    avg = avg * 100.0 / len(helpers)
    if detailed: print("-" * 100)
    summary = "Calibration Report: Average Abs Error (%%) : %5.9f" % (avg)
    print(summary)
    return avg


def setup_model(_yield_ts, _dividend_ts, _spot, init_condition: HestonParams) -> Tuple[ql.HestonModel, ql.AnalyticHestonEngine]:
    v0, kappa, theta, sigma, rho = init_condition.params()
    print(f'Setting up model with init conditions: spot: {_spot}\nv0={v0}, kappa={kappa}, theta={theta}, sigma={sigma}, rho={rho}, ')
    process = ql.HestonProcess(_yield_ts, _dividend_ts, ql.QuoteHandle(ql.SimpleQuote(_spot)), v0, kappa, theta, sigma, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)
    return model, engine


def calibrate_heston_model(strikes, tenors, surface_frame_data: SurfaceFrameData, calibration_params: CalibrationParams,
                           surface_frame_data_raw: SurfaceFrameData, plot=False) -> HestonCalibrationResult:
    calculation_date = ql.Date(calibration_params.ts.day, calibration_params.ts.month, calibration_params.ts.year)

    # As interpolated by Andreasen
    vol_surface = surface_frame_data.ah_vol_surface()
    _tenors = [t for t in tenors if t < vol_surface.maxTime()]
    tenors_dt = [calibration_params.calendar.advance(calculation_date, ql.Period(int(p * 365), ql.Days)) for p in
                 _tenors]

    method_to_call = getattr(vol_surface, 'blackVol')
    X, Y = np.meshgrid(strikes, _tenors)
    volatilities = np.array([method_to_call(float(y), float(x))
                             for xr, yr in zip(X, Y)
                             for x, y in zip(xr, yr)]
                            ).reshape(len(X), len(X[0]))

    model5, engine5 = setup_model(
        calibration_params.yield_ts, calibration_params.dividend_ts, calibration_params.spot,
        init_condition=calibration_params.initial_heston_params
    )
    heston_helpers5, grid_data5 = setup_helpers(
        engine5, tenors_dt, strikes, volatilities,
        calculation_date, calibration_params.spot, calibration_params.yield_ts, calibration_params.dividend_ts,
        calibration_params.calendar
    )

    heston_handle = ql.HestonModelHandle(model5)
    heston_vol_surface = ql.HestonBlackVolSurface(heston_handle)
    # plot_vol_surface(heston_vol_surface, plot_years=np.arange(tenors.min(), tenors.max(), 0.05), plot_strikes=np.arange(minStrike, maxStrike, 1))

    mybound = MyBounds()
    initial_condition = list(model5.params())
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": calibration_params.heston_parameter_bounds}
    cost_function = cost_function_generator(
        model5, heston_helpers5, norm=True)
    sol = basinhopping(cost_function, initial_condition, niter=5,
                       minimizer_kwargs=minimizer_kwargs,
                       stepsize=0.005,
                       accept_test=mybound,
                       interval=10)

    theta, kappa, rho, sigma, v0 = list(model5.params())  # returns theta, kappa, rho, sigma, v0
    hestonParams = HestonParams(v0, kappa, theta, sigma, rho)
    print("v0=%f, kappa=%f, theta=%f, sigma=%f, rho=%f" % (v0, kappa, theta, sigma, rho))
    error = calibration_report(heston_helpers5, grid_data5)  #, detailed=True)
    if plot:
        min_strike = calibration_params.spot * calibration_params.f_min_strike
        max_strike = calibration_params.spot * calibration_params.f_max_strike
        # Andreasen
        plot_vol_surface(vol_surface, plot_years=np.arange(_tenors.min(), _tenors.max(), 0.05),
                         plot_strikes=np.arange(min_strike, max_strike, 1))
        # Heston
        plot_vol_surface(heston_vol_surface, plot_years=np.arange(_tenors.min(), _tenors.max(), 0.05),
                         plot_strikes=np.arange(min_strike, max_strike, 1))

    return HestonCalibrationResult(calibration_params.ts, hestonParams, error, None,  # surface_frame_data.to_serializable(),
                                   calibration_params.to_serializable(), surface_frame_data_raw.to_serializable())


def calibrate_heston_model_from_market_data(ts: pd.Timestamp, quotes: Dict[OptionContract, pd.DataFrame],
                                            ps_spot: pd.Series, calibration_params: CalibrationParams):
    """
    Heston Model Calibration Using QuantLib Python and Scipy Optimize
    https://gouthamanbalaraman.com/blog/heston-calibration-scipy-optimize-quantlib-python.html
    :param ts:
    :param quotes:
    :param ps_spot:
    :param calibration_params:
    :return:
    """
    t0 = time.time()
    calculation_ts = ts
    calculation_date = ts.date()
    spot = ps_spot.loc[calculation_ts]
    min_strike = spot * calibration_params.f_min_strike
    max_strike = spot * calibration_params.f_max_strike
    print(f'calculation_ts={calculation_ts}, min_strike={min_strike}, spot={spot}, max_strike={max_strike}')

    calibration_params.ts = ts
    calibration_params.spot = spot
    calibration_params.load_unserializable_objects()

    df_q = gen_df_q(quotes, ps_spot, calibration_params)

    strikes = np.array(sorted(set(df_q['strike'])))
    tenors = np.array([(dt - calculation_date).days / 365 for dt in sorted(set(df_q['expiry']))])

    sfd_raw = SurfaceFrameData(df_q, calibration_params)

    # Repairing Quotes satisfying arb-free conditions better
    repair = False
    if repair:  # calibration_params.option_right == 'call':
        df_q_r = repair_prices(df_q, calculation_ts.date(), right=calibration_params.option_right, n_repairs=2)
    else:
        df_q_r = df_q
    sfd_repaired = SurfaceFrameData(df_q_r, calibration_params)

    calibration_result = calibrate_heston_model(strikes, tenors, sfd_repaired, calibration_params, sfd_raw, plot=False)
    print(f'Calibration took {(time.time() - t0)/60} minutes')

    ts_str = calibration_params.ts.strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(Paths.path_calibration, 'tmp', f'heston_calibration_results_{calibration_params.option_right}_{ts_str}.pkl'), 'wb') as f:
        pickle.dump(calibration_result, f)

    return calibration_result


def mp_calibrate_heston(timestamps: List[pd.Timestamp], quotes: Dict[OptionContract, pd.DataFrame], ps_spot, calibration_params: CalibrationParams):
    f = partial(calibrate_heston_model_from_market_data, quotes=quotes, ps_spot=ps_spot,
                calibration_params=calibration_params.to_serializable())
    # return [f(ts) for ts in timestamps]
    with multiprocessing.Pool(min(int(multiprocessing.cpu_count()/2), len(timestamps), 12)) as pool:
        hestonCalibrationResults = pool.map(f, timestamps)
    return hestonCalibrationResults


def sq(i, calendar):
    print(calendar)
    return OptionContract(i, 'call', 100, datetime.date(2023, 9, 1), 100)


def mp_n(n):
    f = partial(sq, calendar='sadf')
    with multiprocessing.Pool(min(multiprocessing.cpu_count(), n)) as pool:
        return pool.map(f, range(n))


class MyBounds(object):
    def __init__(self, xmin=None, xmax=None):
        xmin = xmin or [0., 0.01, 0.01, -1, 0]
        xmax = xmax or [1, 15, 1, 1, 1.0]
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


# def plot_raw_estimated_srf(df_q_r, calibration_params: CalibrationParams):
#     dft = df_q_r.groupby(['strike', 'expiry'])['mid_iv'].aggregate('first').unstack().sort_index()[
#         list(sorted(set(df_q_r['expiry'])))]
#     # Removing columns only containing  NaN values
#     dft = dft.loc[:, dft.columns[~dft.apply(lambda ps: ps.isna().sum() == len(dft))]]
#     ivs_made_plottable = ql.Matrix(dft.bfill().ffill().values.tolist())
#
#     blackVolSurface = ql.BlackVarianceSurface(calculation_date, calibration_params.calendar, tenorsQlDt, strikes, ivs_made_plottable,
#                                               day_count)
#     blackVolSurface.setInterpolation("bicubic")
#     # blackVolHandle = ql.BlackVolTermStructureHandle(blackVolSurface)
#
#     plot_vol_surface(blackVolSurface, plot_years=np.arange(tenors.min(), tenors.max(), 0.05),
#                      plot_strikes=np.arange(min(strikes), max(strikes), 1))

def vega_bsm(strike, tenor, vol, calibration_params: CalibrationParams):
    calculation_date = calibration_params.ts.date()
    calculation_date_ql = ql.Date(calculation_date.day, calculation_date.month, calculation_date.year)
    set_ql_calculation_date(calculation_date_ql)

    maturity_date = tenor if isinstance(tenor, datetime.date) else calibration_params.ts.date() + datetime.timedelta(days=int(tenor * 365))
    if vol is not None and np.isnan(vol):
        return np.nan
    spotQ = ql.QuoteHandle(ql.SimpleQuote(calibration_params.spot))
    volQuoteHandle = ql.QuoteHandle(ql.SimpleQuote(vol))
    put_call = str2ql_option_right(calibration_params.option_right)

    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date_ql, calibration_params.calendar, volQuoteHandle, calibration_params.day_count))
    eu_option = make_eu_option(put_call, strike, maturity_date)
    bsmProcess = ql.BlackScholesMertonProcess(spotQ, calibration_params.dividend_ts, calibration_params.yield_ts, vol_ts)
    analytical_engine = ql.AnalyticEuropeanEngine(bsmProcess)
    eu_option.setPricingEngine(analytical_engine)
    return eu_option.vega()


def iv_surface_data_from_heston_params(strikes: np.array, tenors: np.array, heston_params: HestonParams, calibration_params: CalibrationParams) -> SurfaceData:
    spot_quote = ql.QuoteHandle(ql.SimpleQuote(calibration_params.spot))
    # YieldTermStructureHandle riskFreeTS, YieldTermStructureHandle dividendTS, QuoteHandle s0, Real v0, Real kappa, Real theta, Real sigma, Real rho
    heston_process = ql.HestonProcess(calibration_params.yield_ts, calibration_params.dividend_ts, spot_quote, *heston_params.params())
    heston_model = ql.HestonModel(heston_process)

    heston_handle = ql.HestonModelHandle(heston_model)
    heston_vol_surface = ql.HestonBlackVolSurface(heston_handle)
    volatilities = np.zeros((len(tenors), len(strikes)))
    for i, tenor in enumerate(tenors):
        for j, strike in enumerate(strikes):
            maturity_date = tenor if isinstance(tenor, datetime.date) else calibration_params.ts.date() + datetime.timedelta(days=int(tenor * 365))
            maturity_date = ql.Date(maturity_date.day, maturity_date.month, maturity_date.year)
            volatilities[i, j] = heston_vol_surface.blackVol(maturity_date, strike)

    return SurfaceData(strikes, tenors, volatilities, calibration_params.spot)


def heston_surface_from_heston_params(heston_params: HestonParams, calibration_params: CalibrationParams) -> ql.HestonBlackVolSurface:
    spot_quote = ql.QuoteHandle(ql.SimpleQuote(calibration_params.spot))
    # YieldTermStructureHandle riskFreeTS, YieldTermStructureHandle dividendTS, QuoteHandle s0, Real v0, Real kappa, Real theta, Real sigma, Real rho
    heston_process = ql.HestonProcess(calibration_params.yield_ts, calibration_params.dividend_ts, spot_quote, *heston_params.params())
    heston_model = ql.HestonModel(heston_process)
    heston_handle = ql.HestonModelHandle(heston_model)
    return ql.HestonBlackVolSurface(heston_handle)


def bw_silverman(v):
    return 0.9 * min(v.std(), iqr(v)/1.35)*(len(v)**-0.2)


def most_probable_heston_params(heston_calibration_results: List[HestonCalibrationResult]) -> HestonParams:
    hestonParamAttributes = ['v0', 'kappa', 'theta', 'rho', 'sigma']
    vectors = [np.array([getattr(r.params, attr) for r in heston_calibration_results]) for attr in hestonParamAttributes]
    probable_params = []
    for v in vectors:
        X = v.reshape(-1, 1)
        kde = KernelDensity(kernel='tophat', bandwidth=bw_silverman(X)).fit(X)
        probable_params.append(X[np.exp(kde.score_samples(X)).argmax()][0])
    return HestonParams(*probable_params)

def heston_param_probability_quantiles(heston_calibration_results: List[HestonCalibrationResult], quantiles=(0.10, 0.90)) -> Dict[str, Tuple[float, float]]:
    hestonParamAttributes = ['v0', 'kappa', 'theta', 'rho', 'sigma']
    vectors = {attr: np.array([getattr(r.params, attr) for r in heston_calibration_results]) for attr in hestonParamAttributes}
    attr_q = {}
    for attr, v in vectors.items():
        v.sort()
        X = v.reshape(-1, 1)
        kde = KernelDensity(kernel='tophat', bandwidth=bw_silverman(X)).fit(X)
        df = pd.DataFrame(np.vstack([v, np.exp(kde.score_samples(X))]).transpose(), columns=['x', 'score'])
        df['score_cumsum_norm'] = df['score'].cumsum() / df['score'].cumsum().max()
        qlow = df['x'].iloc[(~(df['score_cumsum_norm'] < quantiles[0])).argmax()-1]
        qhigh = df['x'].iloc[(df['score_cumsum_norm'] > quantiles[1]).argmax()]
        attr_q[attr] = (qlow, qhigh)
        # df.set_index('x')[['score', 'score_cumsum_norm']].plot()
    return attr_q

def utitlity_surface_between_heston_models(vega_surface: SurfaceData, heston_params0: HestonParams, heston_params1: HestonParams, calibration_params: CalibrationParams):
    """
    For every time step, calculate the difference between the two models' IVs. The 0 should fit as good as possible the current arb free surface.
    Srf 1 is a surface constructed with more likely params the surface likely returns to.
    Multiply the difference by the vega surface to get the utility surface.
    """
    srf0 = iv_surface_data_from_heston_params(vega_surface.x, vega_surface.y, heston_params0, calibration_params)
    srf1 = iv_surface_data_from_heston_params(vega_surface.x, vega_surface.y, heston_params1, calibration_params)
    srf_diff = srf1.z - srf0.z
    return SurfaceData(srf0.x, srf0.y, srf_diff, calibration_params.spot)
    # Somehow vega surface too often zerp
    return SurfaceData(srf0.x, srf0.y, srf_diff * vega_surface.z, calibration_params.spot)


def get_utility_surface(heston_calibration_result: HestonCalibrationResult, probable_heston_params: HestonParams) -> SurfaceData:
    r = heston_calibration_result
    vega_srf = vol_surface_data2vega_surface_data(r.surface_frame_data_raw, r.calibration_params)
    return utitlity_surface_between_heston_models(vega_srf, r.params, probable_heston_params, r.calibration_params)


@dataclass
class Trade:
    utility: float
    quantity: float
    strike: float
    expiry: datetime.date
    ts0: datetime.datetime
    price0: float
    delta0: float
    vega0: float
    iv_heston_kde0: float
    iv_heston0: float
    iv_ah0: float
    iv_market0: float
    spot0: float

    repaired2ah_pc_err: float
    repaired2heston_pc_err: float
    raw2heston_pc_err: float
    ah2heston_pc_err: float

    v00: float = 0
    kappa0: float = 0
    theta0: float = 0
    rho0: float = 0
    sigma0: float = 0
    has_heston_outlier: Dict[str, bool] = None

    ts1: datetime.datetime = None
    price1: float = None  # at same expiry date
    iv_heston1: float = None
    iv_ah1: float = None
    iv_market1: float = None  # at same moneyness
    spot1: float = None

    v01: float = 0
    kappa1: float = 0
    theta1: float = 0
    rho1: float = 0
    sigma1: float = 0


    @property
    def moneyness0(self):
        return self.strike / self.spot0

    @property
    def expiry_ql(self):
        return ql.Date(self.expiry.day, self.expiry.month, self.expiry.year)


def price(srf, strike, expiry):
    strike_match = nearest_strike(srf, strike)
    try:
        return srf.loc[strike_match, expiry] if strike_match else np.nan
    except KeyError:
        return np.nan


def nearest_strike(srf: pd.DataFrame, strike):
    return next(iter([i for i in srf.index if abs(i - strike) < 0.1]), None)


def nearest_expiry(srf: pd.DataFrame, expiry: datetime.date):
    return next(iter([i for i in srf.columns if abs(i - expiry).days <= 1]), None)


def calibration_error(srf, df_in: pd.DataFrame) -> float:
    return calibration_error_df(srf, df_in)['error'].abs().mean()


def calibration_error_df(srf, df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df['expiry_ql'] = [ql.Date(dt.day, dt.month, dt.year) for dt in df['expiry']]
    df['iv_ah'] = [srf.blackVol(expiry_ql, strike) for expiry_ql, strike in df[['expiry_ql', 'strike']].values]
    df['error'] = df['iv_ah'] / df['mid_iv'] - 1
    return df


def srf_pricer_interpolator(df: pd.DataFrame, ts: datetime.datetime, plot=False):
    x = np1d_from_df(df, 'strike')
    y = np1d_from_df(df, 'expiry')
    z = np1d_from_df(df, 'mid_iv')
    ix_an = ~np.isnan(z)
    x = x[ix_an]
    y = y[ix_an]
    z = z[ix_an]
    y = np.array([(dt - ts.date()).days / 365 for dt in y])
    interp = LinearNDInterpolator(list(zip(x, y)), z)

    if plot:
        X = np.linspace(min(x), max(x))
        Y = np.linspace(min(y), max(y))
        X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
        Z = interp(X, Y)
        plot_surface(Z, X[0], Y[:, 0])
    return interp


def bt_heston_param_mean_reversion(min_util_long=0.05, min_util_short=-0.05):
    """
    Given n calibration results and a min abs utility threshold, go long/short a particular (strike, tenor) pair if the utility is above/below the threshold.
    Would want to trade calibration errors...
    If trading based on heston perturbations, may want to put this in perspective with calibration error. Is increased parameter disturbance correlated with increased calibration error?
    Further, raw data cleaning and AH interpolation further removed opportunities to arb value. There is no continuous market IV surface to test against... AH without cleaning at most...
    """

    with open(os.path.join(Paths.path_calibration, 'hestonCalibrationResults_put_2024-01-28T130807.pkl'), 'rb') as f:
        heston_calibration_results: List[HestonCalibrationResult] = pickle.load(f)
    heston_calibration_results = [r for r in heston_calibration_results if r and r.error < 100]

    for r in heston_calibration_results:
        r.calibration_params.load_unserializable_objects()
    probable_heston_params = most_probable_heston_params(heston_calibration_results)
    heston_param_q_val = heston_param_probability_quantiles(heston_calibration_results, quantiles=(0.1, 0.9))
    print(f'probable_heston_params: {probable_heston_params}')
    print(f'Average calibration error: {np.average([r.error for r in heston_calibration_results])}')

    srf_ah_iv_0 = None
    srf_price_0 = None
    srf_heston_iv_0 = None
    spot0 = None
    new_trades = []
    trades = []

    # heston_probable_surface = heston_surface_from_heston_params(probable_heston_params, r.calibration_params)
    # plot_vol_surface(heston_probable_surface, np.arange(0.01, 2, 0.2), sorted(dft['strike'].unique()))
    #
    # trial_params = HestonParams(
    #     probable_heston_params.v0,
    #     probable_heston_params.kappa,
    #     probable_heston_params.theta,
    #     probable_heston_params.rho,
    #     probable_heston_params.sigma
    # )
    # heston_trial_surface = heston_surface_from_heston_params(trial_params, r.calibration_params)
    # plot_vol_surface(heston_trial_surface, np.arange(0.01, 2, 0.2), sorted(dft['strike'].unique()))

    # plot_vol_surface(srf_heston_iv_1, np.arange(0.1, 2, 0.2), sorted(dft['strike'].unique()))
    # plot_vol_surface(srf_ah_iv_1, np.arange(0.1, 2, 0.2), sorted(dft['strike'].unique()))

    for i, r in enumerate(heston_calibration_results):
        spot1 = r.calibration_params.spot
        srf_heston_iv_kde = heston_surface_from_heston_params(probable_heston_params, r.calibration_params)
        srf_heston_iv_1 = heston_surface_from_heston_params(r.params, r.calibration_params)
        srf_ah_iv_1: ql.AndreasenHugeVolatilityAdapter = r.surface_frame_data_raw.ah_vol_surface()
        # ah_calibration_error = calibration_error(srf_ah_iv_1, r.surface_frame_data.df)

        util_surface1 = get_utility_surface(r, probable_heston_params)
        srf_price_raw_1 = r.surface_frame_data_raw.df.groupby(['strike', 'expiry'])['mid_close'].aggregate('first').unstack().sort_index()
        # repaired_iv_interpolator1 = srf_pricer_interpolator(r.surface_frame_data.df, r.calibration_params.ts)
        raw_iv_interpolator1 = srf_pricer_interpolator(r.surface_frame_data_raw.df, r.calibration_params.ts)

        # Track IV Earned and reset position
        if i > 0:
            for t in new_trades:
                try:
                    tenor1 = max((t.expiry - r.calibration_params.ts.date()).days / 365, 0.001)
                    t.ts1 = r.calibration_params.ts
                    # This BT only calculates the IV earned, not the price earned. Therefore, moneyness must be equal across srf0 and srf1
                    t.spot1 = spot1
                    strike1 = t.moneyness0 * spot1
                    t.price1 = price(srf_price_raw_1, t.strike, t.expiry)
                    t.iv_heston1 = srf_heston_iv_1.blackVol(t.expiry_ql, strike1)
                    t.iv_ah1 = srf_ah_iv_1.blackVol(t.expiry_ql, strike1)
                    t.iv_market1 = interpolate_pt(raw_iv_interpolator1, strike1, tenor1)[0]

                    t.v01 = r.params.v0
                    t.kappa1 = r.params.kappa
                    t.theta1 = r.params.theta
                    t.rho1 = r.params.rho
                    t.sigma1 = r.params.sigma

                    # tenor0 = max((expiryQl - srf_ah_iv_0.referenceDate()) / 365, 0.001)
                except Exception as e:
                    print(e)
            trades += new_trades

        new_trades = []

        """
        probable peak: x
        norm dist to sum 1, cumsum, proceed only if quantile cumsum < 10% or > 90%
        Can also handle multiple peaks in between well!!
        
        if not heston paramaters exceptionally perturbed:
        continue
        """
        heston_param_outlier = {attr: val for attr, val in asdict(r.params).items() if val < heston_param_q_val[attr][0] or val > heston_param_q_val[attr][1]}

        if True:  # heston_param_outlier:
            print(f'{r.ts} Heston param outlier: {heston_param_outlier}')
            for i, tenor in enumerate(util_surface1.y):
                for j, strike in enumerate(util_surface1.x):
                    if int(tenor * 365) < 14:  # don't get into those near ones. Heston bad at it anyway
                        continue

                    off_expiry = r.calibration_params.ts.date() + datetime.timedelta(days=int(tenor * 365))
                    expiry = nearest_expiry(srf_price_raw_1, off_expiry)
                    # Only test tenors, strikes that are actually available in raw data...
                    if nearest_strike(srf_price_raw_1, strike) not in srf_price_raw_1.index or expiry not in srf_price_raw_1.columns:
                        continue

                    util = util_surface1.z[i, j]
                    if util > min_util_long:
                        quantity = 1
                    elif util < min_util_short:
                        quantity = -1
                    else:
                        quantity = 0
                    if quantity != 0:
                        expiry_ql = ql.Date(expiry.day, expiry.month, expiry.year)
                        iv_market1 = interpolate_pt(raw_iv_interpolator1, strike, tenor)[0]
                        iv_repaired1 = 1  # interpolate_pt(repaired_iv_interpolator1, strike, tenor)[0]
                        iv_ah1 = srf_ah_iv_1.blackVol(expiry_ql, strike)
                        iv_heston1 = srf_heston_iv_1.blackVol(expiry_ql, strike)
                        iv_heston_kde1 = srf_heston_iv_kde.blackVol(expiry_ql, strike)

                        new_trades.append(
                            Trade(util, quantity, strike, expiry, r.calibration_params.ts,
                                  price(srf_price_raw_1, strike, expiry),
                                  delta_bsm(strike, expiry, spot1, r.calibration_params.option_right, iv_market1, r.calibration_params.ts.date(), r.calibration_params.calendar, r.calibration_params.day_count, r.calibration_params.yield_ts, r.calibration_params.dividend_ts),
                                  vega_bsm(strike, expiry, iv_market1, r.calibration_params),
                                  iv_heston_kde1,
                                  iv_heston1,
                                  iv_ah1,
                                  iv_market1,
                                  r.calibration_params.spot,
                                  iv_ah1 / iv_repaired1 - 1,
                                  iv_heston1 / iv_repaired1 - 1,
                                  iv_heston1 / iv_market1 - 1,
                                  iv_heston1 / iv_ah1 - 1,
                                  r.params.v0, r.params.kappa, r.params.theta, r.params.rho, r.params.sigma, heston_param_outlier
                                  )
                        )
        else:
            print(f'{r.ts} No heston param outliers')

        # srf_ah_iv_0: ql.AndreasenHugeVolatilityAdapter = r.surface_frame_data.ah_vol_surface()
        # srf_price_0 = r.surface_frame_data.df.groupby(['strike', 'expiry'])['mid_close'].aggregate('first').unstack().sort_index()
        # srf_heston_iv_0 = heston_surface_from_heston_params(r.params, r.calibration_params)

    dft = pd.DataFrame(trades)
    # dft['calibration_error'] = dft['iv_heston0'] / dft['iv_ah0'] - 1
    dft['ah_iv_earned'] = (dft['quantity'] * (dft['iv_ah1'] - dft['iv_ah0']))
    dft['heston_iv_earned'] = (dft['quantity'] * (dft['iv_heston1'] - dft['iv_heston0']))
    dft['market_iv_earned'] = (dft['quantity'] * (dft['iv_market1'] - dft['iv_market0']))
    dft = dft[~dft['market_iv_earned'].isna()]
    dft['dS'] = dft['spot1'] - dft['spot0']
    dft['dte'] = (dft['expiry'] - dft['ts0'].dt.date).apply(lambda x: x.days)

    dft['dv0'] = dft['v01'] - dft['v00']
    dft['dkappa'] = dft['kappa1'] - dft['kappa0']
    dft['dtheta'] = dft['theta1'] - dft['theta0']
    dft['drho'] = dft['rho1'] - dft['rho0']
    dft['dsigma'] = dft['sigma1'] - dft['sigma0']

    adjusting_twd_mean = lambda ps, mean: ps.iloc[1] > ps.iloc[0] if ps.iloc[0] < mean else ps.iloc[1] < ps.iloc[0]
    adjusting_twd_mean = partial(adjusting_twd_mean, mean=probable_heston_params.v0)
    dft[['v00', 'v01']].apply(adjusting_twd_mean, axis=1).apply(lambda tf: 1 if tf else -1)

    dft['move2mean_v0'] = dft['dv0'].abs() * dft[['v00', 'v01']].apply(partial(adjusting_twd_mean, mean=probable_heston_params.v0), axis=1).apply(lambda tf: 1 if tf else -1)
    dft['move2mean_kappa'] = dft['dkappa'].abs() * dft[['kappa0', 'kappa1']].apply(partial(adjusting_twd_mean, mean=probable_heston_params.kappa), axis=1).apply(lambda tf: 1 if tf else -1)
    dft['move2mean_theta'] = dft['dtheta'].abs() * dft[['theta0', 'theta1']].apply(partial(adjusting_twd_mean, mean=probable_heston_params.theta), axis=1).apply(lambda tf: 1 if tf else -1)
    dft['move2mean_rho'] = dft['drho'].abs() * dft[['rho0', 'rho1']].apply(partial(adjusting_twd_mean, mean=probable_heston_params.rho), axis=1).apply(lambda tf: 1 if tf else -1)
    dft['move2mean_sigma'] = dft['dsigma'].abs() * dft[['sigma0', 'sigma1']].apply(partial(adjusting_twd_mean, mean=probable_heston_params.sigma), axis=1).apply(lambda tf: 1 if tf else -1)

    dft['div_market_heston'] = dft['iv_market0'] - dft['iv_heston0']
    dft['div_market_ah'] = dft['iv_market0'] - dft['iv_ah0']

    dft['option_pnl'] = dft['quantity'] * (dft['price1'] - dft['price0']) * 100
    dft['equity_hedge_pnl'] = -dft['quantity'] * (dft['spot1'] - dft['spot0']) * dft['delta0'] * 100
    dft['total_pnl'] = dft['option_pnl'] + dft['equity_hedge_pnl']

    print(f'# Trades: {len(trades)}')
    print(f'Raw 2 Heston MAE: {dft["raw2heston_pc_err"].abs().mean() * 100}%')
    # print(f'Repaired 2 Heston MAE: {dft["repaired2heston_pc_err"].abs().mean() * 100}%')
    # print(f'AH MAE: {dft["repaired2ah_pc_err"].abs().mean() * 100}%')
    print(f'Heston MAE: {dft["ah2heston_pc_err"].abs().mean() * 100}%')

    dfs = dft
    dfs = dfs[~dfs['option_pnl'].isna()]
    dfs = dfs[
        ((dfs['iv_market0'] > dfs['iv_ah0']) & (dfs['quantity'] < 0)) |
        ((dfs['iv_market0'] < dfs['iv_ah0']) & (dfs['quantity'] < 0))
    ]  # THIS

    # dfs = dft[dft['quantity'] > 0]
    # dfs = dft[dft['move2mean_v0'] < 0.4]
    # dfs = dft[dft['calibration_error'].abs() < 0.15]
    # dfs = dfs[(dfs['iv_market0'] < dfs['iv_ah0']) & (dfs['iv_market0'] < dfs['iv_heston0']) & (dfs['quantity'] > 0)]

    # dfs = dfs[(dfs['iv_market0'] > dfs['iv_heston0']) & (dfs['quantity'] < 0)]  # Best PnL per trade...
    # dfs = dfs[(dfs['iv_market0'] > dfs['iv_heston0']) & (dfs['iv_market0'] < dfs['iv_heston1']) & (dfs['quantity'] < 0)]  # Best PnL per trade... Volume too low.
    # dfs = dfs[(dfs['iv_market0'] < dfs['iv_heston0']) & (dfs['quantity'] > 0)]  # low pnl per trade, both call & put.

    # dfs = dfs[(dfs['iv_market0'] > dfs['iv_ah0']) & (dfs['quantity'] < 0)]  # THIS
    # dfs = dfs[(dfs['iv_market0'] < dfs['iv_ah0']) & (dfs['quantity'] > 0)]  # THIS. also good for put. not for call
    #
    # dfs = dfs[(dfs['iv_market0'] > dfs['iv_ah0']) & (dfs['iv_market0'] > dfs['iv_heston0']) & (dfs['quantity'] < 0)]  # lower volume, better pnl.
    # dfs = dfs[(dfs['iv_market0'] < dfs['iv_ah0']) & (dfs['iv_market0'] < dfs['iv_heston0']) & (dfs['quantity'] > 0)]  # For calls: better than only AH filter.

    print(f'# Sample trades: {len(dfs)}, Market IV earned - Total: {dfs["market_iv_earned"].sum()}, Per Trade: {dfs["market_iv_earned"].mean()}')
    print(f'# Sample trades: {len(dfs)}, Option PnL: {dfs["option_pnl"].sum()}, Hedge PnL: {dfs["equity_hedge_pnl"].sum()}, Total PnL: {dfs["total_pnl"].sum()}, PnL per trade: {dfs["total_pnl"].mean()}')

    # Put a learner on this, finding best way to split attributes...

    # print(f'# Sample trades: {len(dfs)}, AH IV earned - Total: {dfs["ah_iv_earned"].sum()}, Per Trade: {dfs["ah_iv_earned"].mean()}')
    # print(f'# Sample trades: {len(dfs)}, Heston IV earned - Total: {dfs["heston_iv_earned"].sum()}, Per Trade: {dfs["heston_iv_earned"].mean()}')
    # pd.set_option('display.max_columns', None)
    # # dfs[['ts0', 'quantity', 'delta0', 'market_iv_earned', 'option_pnl', 'equity_hedge_pnl', 'total_pnl']]
    # dft[dft['delta0'] == 0][['ts0', 'strike', 'expiry', 'spot0', 'price0', 'iv_market0']]
    # calc_dt = datetime.date(2024, 1, 11)
    # flat = ql.YieldTermStructureHandle(ql.FlatForward(ql.Date(calc_dt.day, calc_dt.month, calc_dt.year), 0, ql.Actual365Fixed()))
    # delta_bsm(78.0, datetime.date(2024, 1, 27), 78.94, 'call', 0.22, calc_dt, ql.UnitedStates(ql.UnitedStates.NYSE), ql.Actual365Fixed(), flat, flat)

    # Ideally report grouped by option_right, expiry, moneyness, tenor, quantity
    if False:
        dfs.set_index("ts0").sort_index()[['option_pnl', 'equity_hedge_pnl', 'total_pnl']].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index("delta0").sort_index()[['market_iv_earned']].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index('div_market_heston').sort_index()[["market_iv_earned"]].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index("div_market_ah").sort_index()[['market_iv_earned']].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index("market_iv_earned").sort_index()[['repaired2ah_pc_err']].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index("market_iv_earned").sort_index()[['ah2heston_pc_err']].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index("market_iv_earned").sort_index()[['repaired2heston_pc_err']].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index("market_iv_earned").sort_index()[['quantity']].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index("market_iv_earned").sort_index()[['dte']].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index("market_iv_earned").sort_index()[['strike']].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index("market_iv_earned").sort_index()[['utility']].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index("market_iv_earned").sort_index()[['iv_ah0']].plot(style='o', alpha=0.5, figsize=(15, 10))

        # dfs.set_index("market_iv_earned").sort_index()[['dv0']].plot(style='o', alpha=0.5, figsize=(15, 10))
        # dfs.set_index("market_iv_earned").sort_index()[['dkappa']].plot(style='o', alpha=0.5, figsize=(15, 10))
        # dfs.set_index("market_iv_earned").sort_index()[['dtheta']].plot(style='o', alpha=0.5, figsize=(15, 10))
        # dfs.set_index("market_iv_earned").sort_index()[['drho']].plot(style='o', alpha=0.5, figsize=(15, 10))
        # dfs.set_index("market_iv_earned").sort_index()[['dsigma']].plot(style='o', alpha=0.5, figsize=(15, 10))

        dfs.set_index("market_iv_earned").sort_index()[['v00']].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index("market_iv_earned").sort_index()[['kappa0']].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index("market_iv_earned").sort_index()[['theta0']].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index("market_iv_earned").sort_index()[['rho0']].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index("market_iv_earned").sort_index()[['sigma0']].plot(style='o', alpha=0.5, figsize=(15, 10))

        dfs.set_index('move2mean_v0').sort_index()[["market_iv_earned"]].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index('move2mean_kappa').sort_index()[["market_iv_earned"]].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index('move2mean_theta').sort_index()[["market_iv_earned"]].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index('move2mean_rho').sort_index()[["market_iv_earned"]].plot(style='o', alpha=0.5, figsize=(15, 10))
        dfs.set_index('move2mean_sigma').sort_index()[["market_iv_earned"]].plot(style='o', alpha=0.5, figsize=(15, 10))

        linregress(dfs['market_iv_earned'], dfs['move2mean_sigma'])

        dfs[dfs['move2mean_v0'] > 0]['market_iv_earned'].sum()
        dfs[dfs['move2mean_kappa'] > 0]['market_iv_earned'].sum()
        dfs[dfs['move2mean_theta'] > 0]['market_iv_earned'].sum()
        dfs[dfs['move2mean_rho'] > 0]['market_iv_earned'].sum()
        dfs[dfs['move2mean_sigma'] > 0]['market_iv_earned'].sum()
        for c in ['move2mean_v0', 'move2mean_kappa', 'move2mean_theta', 'move2mean_rho', 'move2mean_sigma']:
            print(c, dfs[dfs[c] > 0]['market_iv_earned'].sum())
            print(c, dfs[dfs[c] < 0]['market_iv_earned'].sum())


def calibrate_heston_models_over_time():
    from options.client import Client

    start = datetime.date(2023, 9, 1)
    end = datetime.date(2024, 1, 22)
    # end = datetime.date(2023, 9, 10)
    resolution = Resolution.minute
    client = Client()
    sym = 'dell'
    equity = Equity(sym)
    option_right = 'put'
    min_dte = 14
    f_min_strike = 0.9
    f_max_strike = 1.1
    # model5.params()  # returns theta, kappa, rho, sigma, v0
    bounds = [(0, 1), (0.01, 30), (0.01, 15.), (-1, 1), (0, 1.0)]
    # probable_heston_params: HestonParams(v0=0.10641492676975998, kappa=1.7086747299788199, theta=0.17318132042556636, rho=-0.3207602890545346, sigma=1.3744017528238948)
    initialHestonParams = HestonParams(0.1, 1.7, 0.17, -0.32, 1.37)
    rate = 0.0435  # https://www.bloomberg.com/markets/rates-bonds/government-bonds/us
    dividend_yield = 0.0182  # https://www.morningstar.com/stocks/xnys/dell/quote

    trades = client.history([equity], start, end, resolution, TickType.trade, SecurityType.equity)
    ps_spot = trades[sym]['close']
    # removing non-RTH hours from spot because not present in quotes
    ps_spot = ps_spot[(ps_spot.index.time >= datetime.time(10, 0)) & (ps_spot.index.time <= datetime.time(15, 0))]
    # ret = np.log(ps_spot / ps_spot.shift(1))

    contracts = client.central_volatility_contracts(equity, start, end, n=300)
    contracts = list(chain(*contracts.values()))
    print(f'# Contracts loaded: {len(contracts)}')

    quotes = client.history(contracts, start, end, resolution, TickType.quote, SecurityType.option)
    quotes = {OptionContract.from_contract_nm(k): v for k, v in quotes.items()}

    # ix_quotes = pd.concat([df[~df.index.duplicated()] for df in quotes.values()], axis=1).index
    # print(len(ix_quotes))

    loc = find_loc_every_x_pc(ps_spot, 0.005)
    print('Calibrating for', len(loc), 'time steps')

    # FFill every loc that's not within a quote df
    for c, df in quotes.items():
        dft = pd.concat([df, pd.DataFrame(index=loc, columns=df.columns)]).sort_index().fillna(method='ffill')
        dft = dft.loc[~dft.index.duplicated()]
        quotes[c] = dft.loc[loc]

    calibration_params = CalibrationParams(
        datetime.datetime(2000, 1, 1),
        0,
        f_min_strike,
        f_max_strike,
        option_right,
        min_dte,
        bounds,
        initialHestonParams,
        rate,
        dividend_yield
    )

    hestonCalibrationResults = mp_calibrate_heston(loc, quotes, ps_spot, calibration_params)
    concat_calibrated_tmp_files()
    print('Done.')

def concat_calibrated_tmp_files():
    option_right = None
    hestonCalibrationResults = []
    for _dir, dirs, fns in os.walk(os.path.join(Paths.path_calibration, 'tmp')):
        for fn in fns:
            if option_right is None:
                option_right = 'put' if '_put_' in fn else 'call'
            with open(os.path.join(_dir, fn), 'rb') as f:
                hestonCalibrationResults.append(pickle.load(f))
    ts_str = datetime.datetime.now().isoformat().replace(':', '')[:17]

    with open(os.path.join(Paths.path_calibration, f'hestonCalibrationResults_{option_right}_{ts_str}.pkl'), 'wb') as f:
        pickle.dump(hestonCalibrationResults, f)

    for _dir, dirs, fns in os.walk(os.path.join(Paths.path_calibration, f'calibration', 'tmp')):
        for fn in fns:
            os.remove(os.path.join(_dir, fn))


if __name__ == '__main__':
    # print(mp_n(5))
    # calibrate_heston_models_over_time()
    bt_heston_param_mean_reversion(0, 0)
    # concat_calibrated_tmp_files()
