import math

import QuantLib as ql
import numpy as np
import pandas as pd

from datetime import datetime, date, timedelta
from typing import Iterable, Callable, Dict, List, Tuple, Any, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from math import fabs, erf, erfc

from QuantLib import DateVector, DoubleVector
from numba import njit
from numpy import ndarray, dtype
from numpy._typing import NDArray

from options.typess.dividend import get_dividends, Dividend
from options.typess.equity import Equity
from options.typess.security import Security, SecurityDataSnap
from options.typess.scenario import Scenario
from shared.constants import DiscountRateMarket
from options.ql_helper import engined_option
from options.typess.enums import OptionPricingModel, OptionRight, TickType, Resolution
from options.typess.option_contract import OptionContract
import numba as nb
import merlin

from shared.yield_curve import YieldCurve, ZeroCurveData


@dataclass
class GreekParameters:
    price_underlying: float
    volatility: float
    calculation_date: date


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float


@dataclass
class OptionSerialized:
    optionContract: OptionContract
    calculationDate: date
    price_underlying: float
    volatility: float

@dataclass
class Style:
    European = 'European'
    American = 'American'
    AmericanDiv = 'AmericanDiv'


class Option(Security):
    # Ideally refactored to combine with OptionContract
    multiplier = 100
    accuracy = 1.0e-4
    max_iterations = 100
    minVol = 0.0001
    maxVol = 4.0

    def __init__(self, option_contract: OptionContract, calculation_date: date, price_underlying: float = 0, volatility: float = 0, option_constructor: ql.OneAssetOption = ql.VanillaOption, style=Style.American, optionPricingModel: OptionPricingModel=None, dividends=None, q=False):
        self.q = q
        self.style = style
        if dividends is None:
            dividends = []

        self.optionContract = option_contract
        self.expiry = self.optionContract.expiry
        self.right = self.optionContract.right
        self.price = 0

        self.calculationDate = calculation_date
        self.calculationDateQl = ql.Date(calculation_date.day, calculation_date.month, calculation_date.year)
        self.strike = float(option_contract.strike)
        self.optionType = {OptionRight.put: ql.Option.Put,
                           OptionRight.call: ql.Option.Call}[option_contract.right]
        self.maturityDate = ql.Date(self.optionContract.expiry.day, self.optionContract.expiry.month, self.optionContract.expiry.year)

        self.calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        self.dayCount = ql.Actual365Fixed()
        self.payoff = ql.PlainVanillaPayoff(self.optionType, self.strike)
        self.exercise = ql.EuropeanExercise(self.maturityDate) if style == Style.European else ql.AmericanExercise(self.calculationDateQl, self.maturityDate)

        self.underlyingQuote = ql.SimpleQuote(price_underlying)
        self.underlyingQuoteHandle = ql.QuoteHandle(self.underlyingQuote)

        self.volQuote = ql.SimpleQuote(volatility)
        self.volQuoteHandle = ql.QuoteHandle(self.volQuote)

        self.riskFreeRateQuote = ql.SimpleQuote(DiscountRateMarket)
        self.riskFreeRateQuoteHandle = ql.QuoteHandle(self.riskFreeRateQuote)

        self.dividendSchedule = []
        args = [self.payoff, self.exercise]
        if style == Style.AmericanDiv:
            self.dividendDates = DateVector([ql.Date(d.ex_date.day, d.ex_date.month, d.ex_date.year) for d in dividends if d.ex_date <= self.optionContract.expiry])
            self.dividendAmounts = DoubleVector([d.amount for d in dividends if d.ex_date <= self.optionContract.expiry])
            self.dividendSchedule: ql.DividendSchedule = ql.DividendSchedule([ql.FixedDividend(a, d) for a, d in zip(self.dividendAmounts, self.dividendDates)])
            args += [self.dividendDates, self.dividendAmounts]
        else:
            from options.helper import get_dividend_yield
            # Annualized Yields
            self.dividendRateQuote = ql.SimpleQuote(get_dividend_yield(option_contract.underlying_symbol))
            # self.dividendRateQuote = ql.SimpleQuote(0)
            self.dividendRateQuoteHandle = ql.QuoteHandle(self.dividendRateQuote)

        self.option = engined_option(option_constructor(*args), self.get_bsm(), optionPricingModel=optionPricingModel or OptionPricingModel.AnalyticEuropeanEngine, dividend_schedule=self.dividendSchedule)

    def __eq__(self, other):
        return self.optionContract.__repr__() == other.optionContract.__repr__() if isinstance(other, Option) else False

    def __ge__(self, other):
        return self.optionContract.__repr__() >= other.optionContract.__repr__() if isinstance(other, Option) else False

    def __gt__(self, other):
        return self.optionContract.__repr__() > other.optionContract.__repr__() if isinstance(other, Option) else False

    def __lt__(self, other):
        return self.optionContract.__repr__() < other.optionContract.__repr__() if isinstance(other, Option) else False

    def __le__(self, other):
        return self.optionContract.__repr__() <= other.optionContract.__repr__() if isinstance(other, Option) else False

    @property
    def symbol(self): return self.optionContract.symbol or self.__repr__()
    @property
    def underlying_symbol(self): return self.optionContract.underlying_symbol

    @property
    def equity(self): return self.optionContract.equity

    def csv_name(self, tick_type: TickType, resolution: Resolution, dt: date = None):
        self.optionContract.csv_name(tick_type, resolution, dt)

    def zip_name(self, tick_type: TickType, resolution: Resolution, dt: date = None):
        if date is None:
            raise ValueError('date must be provided')
        return self.optionContract.zip_name(tick_type, resolution, dt)

    def SetPriceUnderlying(self, priceUnderlying: float):
        self.underlyingQuote.setValue(priceUnderlying)

    def SetVolatility(self, volatility: float):
        self.volQuote.setValue(volatility)

    def SetPriceAndVolatility(self, price: float):
        self.price = price
        self.volQuote.setValue(self.iv(price, self.underlyingQuote.value(), self.calculationDate))

    def SetPrice(self, price: float):
        self.price = price

    def SetEvaluationDateToCalcDate(self, calculation_date_ql: ql.Date | date = None):
        if isinstance(calculation_date_ql, date):
            _calculationDateQl = ql.Date(calculation_date_ql.day, calculation_date_ql.month, calculation_date_ql.year)
        elif calculation_date_ql is None:
            _calculationDateQl = self.calculationDateQl
        else:
            _calculationDateQl = calculation_date_ql

        if ql.Settings.instance().evaluationDate != _calculationDateQl:
            ql.Settings.instance().evaluationDate = _calculationDateQl
            self.calculationDateQl = _calculationDateQl
            self.calculationDate = date(_calculationDateQl.year(), _calculationDateQl.month(), _calculationDateQl.dayOfMonth())

    # @lru_cache(maxsize=2**12)
    def iv(self, price_option: float, price_underlying: float, calculation_date: date) -> float:
        """
        Calculate implied volatility from a series of option prices.
        param ps: A series of option prices.
        return: A series of implied volatilities.
        """
        if np.isnan(price_option) or np.isnan(price_underlying):
            raise ValueError('Price or underlying is NaN')

        _calculationDateQl = ql.Date(calculation_date.day, calculation_date.month, calculation_date.year) if calculation_date else self.calculationDateQl
        self.SetEvaluationDateToCalcDate(_calculationDateQl)

        if price_underlying != self.underlyingQuote.value():
            self.underlyingQuote.setValue(price_underlying)

        iv_ = self._iv(price_option, self.get_bsm(), self.maxVol)
        if iv_ == 0 and price_option > self.intrinsic_value():
            iv_ = self._iv(price_option, self.get_bsm(), 600)
        return iv_

    def _iv(self, priceOption, bsm, maxVol):
        # impliedVolatility(VanillaOption self, Real targetValue, ext::shared_ptr< GeneralizedBlackScholesProcess > const & process, Real accuracy=1.0e-4, Size maxEvaluations=100, Volatility minVol=1.0e-4, Volatility maxVol=4.0) -> Volatility
        # impliedVolatility(VanillaOption self, Real targetValue, ext::shared_ptr< GeneralizedBlackScholesProcess > const & process, DividendSchedule dividends, Real accuracy=1.0e-4, Size maxEvaluations=100, Volatility minVol=1.0e-4, Volatility maxVol=4.0) -> Volatility
        try:
            return self.option.impliedVolatility(priceOption, bsm, self.accuracy, self.max_iterations, self.minVol, maxVol)
        except Exception as e:
            if 'root not bracketed' not in str(e):
                print(e)
            return 0

    def ivs(self, prices_option: np.ndarray, prices_underlying: np.ndarray, date_times: np.ndarray[datetime]) -> Iterable[float]:
        assert len(prices_option) == len(prices_underlying) == len(date_times)
        return (self.iv(priceOption, priceUnderlying, dt.date()) for priceOption, priceUnderlying, dt in zip(prices_option, prices_underlying, date_times))

    def greeks(self, params: GreekParameters) -> Greeks:
        self.underlyingQuote.setValue(params.price_underlying)
        self.volQuote.setValue(params.volatility)

        try:
            delta = self.am_option.delta()
            gamma = self.am_option.gamma()
            vega = self.eu_option.vega()
            theta = self.eu_option.theta()
        except Exception as e:
            print(e)
            delta = gamma = vega = theta = 0

        return Greeks(delta, gamma, vega, theta)

    def greeks_vec(self, prices_underlying: np.ndarray, volatilities: np.ndarray, date_times: np.ndarray[datetime]) -> Iterable[Greeks]:
        assert len(prices_underlying) == len(volatilities) == len(date_times)
        return (self.greeks(GreekParameters(priceUnderlying, vola, dt.date())) for (priceUnderlying, vola, dt) in zip(prices_underlying, volatilities, date_times))

    def get_bsm(self):
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(self.calculationDateQl, self.riskFreeRateQuoteHandle, self.dayCount))
        flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(self.calculationDateQl, self.calendar, self.volQuoteHandle, self.dayCount))
        if self.style == Style.AmericanDiv:
            return ql.BlackScholesProcess(self.underlyingQuoteHandle, flat_ts, flat_vol_ts)
        else:
            dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(self.calculationDateQl, self.dividendRateQuoteHandle, self.dayCount))
            return ql.BlackScholesMertonProcess(self.underlyingQuoteHandle, dividend_yield, flat_ts, flat_vol_ts)


    @lru_cache(maxsize=2**12)
    def npv(self, vol: float, price_underlying: float, calculation_date: date):
        """
        # For very large IV values, npv() starts failing... for far OTM/ITM, assume value to be intrinsic only...
        """
        self.SetEvaluationDateToCalcDate(calculation_date)
        if price_underlying != self.underlyingQuote.value():
            self.underlyingQuote.setValue(price_underlying)
        if vol != self.volQuote.value():
            self.volQuote.setValue(vol)

        # delta = self.eu_option.delta()
        # if abs(delta) < 0.02:
        #     npv = 0
        # elif abs(delta) > 0.98:
        #     npv = self.underlyingQuote.value() - self.strike if self.right == 'call' else self.strike - self.underlyingQuote.value()
        # else:
        if calculation_date == self.expiry:
            return self.intrinsic_value()
        return self.eu_option.NPV()

    def nlv(self, market_data: Dict[Security, SecurityDataSnap], q=1, scenario=Scenario.mid):
        if scenario == Scenario.mid:
            price = (market_data[self].bid + market_data[self].ask) / 2
        elif scenario == Scenario.best:
            price = market_data[self].bid if q > 0 else market_data[self].ask
        elif scenario == Scenario.worst:
            price = market_data[self].ask if q > 0 else market_data[self].bid
        else:
            raise ValueError(f'Invalid scenario: {scenario}')
        return price * self.multiplier * q

    @lru_cache(maxsize=2**12)
    def delta(self, vol, priceUnderlying, calculationDate):
        self.SetEvaluationDateToCalcDate(calculationDate)
        if priceUnderlying != self.underlyingQuote.value():
            self.underlyingQuote.setValue(priceUnderlying)
        if vol != self.volQuote.value():
            self.volQuote.setValue(vol)
        return self.eu_option.delta()

    @lru_cache(maxsize=2 ** 12)
    def vega(self, vol, priceUnderlying, calculationDate):
        # code duplication... fix later
        self.SetEvaluationDateToCalcDate(calculationDate)
        if priceUnderlying != self.underlyingQuote.value():
            self.underlyingQuote.setValue(priceUnderlying)
        if vol != self.volQuote.value():
            self.volQuote.setValue(vol)
        return self.eu_option.vega()

    def intrinsic_value(self, price_underlying: float = None):
        if price_underlying:
            self.underlyingQuote.setValue(price_underlying)
        if self.right == 'call':
            return max(self.underlyingQuote.value() - self.strike, 0)
        elif self.right == 'put':
            return max(self.strike - self.underlyingQuote.value(), 0)

    def is_otm(self, priceUnderlying: float):
        return self.intrinsic_value(priceUnderlying) == 0

    def extrinsic_value(self, vol, priceUnderlying, calculationDate):
        return self.npv(vol, priceUnderlying, calculationDate) - self.intrinsic_value(priceUnderlying)

    # def summary(self):
    #     return {
    #         'Contract': str(self.optionContract),
    #         'IV': self.volQuote.value(),
    #         'NPV': self.npv(),
    #         'IntrinsicValue': self.intrinsic_value(),
    #         'ExtrinsicValue': self.extrinsic_value(),
    #         'Delta': self.delta(),
    #         'SpotUnderlying': self.underlyingQuote.value(),
    #     }

    @staticmethod
    def price(s: float | np.ndarray, k: float | np.ndarray, t: float | np.ndarray, iv: float | np.ndarray, r: float | np.ndarray, q: float | np.ndarray, right: str | OptionRight):
        if right == OptionRight.call:
            return price_call(s, k, t, iv, r, q)
        elif right == OptionRight.put:
            return price_put(s, k, t, iv, r, q)

    @classmethod
    def pv(cls, k, t, r):
        return k * np.exp(-r * t)

    def __repr__(self):
        return self.optionContract.ib_symbol()

    def __hash__(self):
        return hash(self.__repr__())

    def __getstate__(self):
        return OptionSerialized(self.optionContract, self.calculationDate, self.underlyingQuote.value(), self.volQuote.value()).__dict__

    def __setstate__(self, state: OptionSerialized.__dict__):
        option = Option(state['optionContract'], state['calculationDate'], state['price_underlying'], state['volatility'])
        self.__dict__.update(option.__dict__)


def dividends2amount_times(dividends: List[Dividend], calculation_date: date) -> Tuple[List[float], List[float]]:
    return (
        [d.amount for d in dividends],
        [((d.ex_date - calculation_date).days + 1) / 365 for d in dividends]
    )


@nb.njit(fastmath=True)
def get_d1(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q: float | NDArray[np.float64]):
    return (np.log(s / k) + (r - q + iv ** 2 / 2) * t) / (iv * np.sqrt(t))


@nb.njit(fastmath=True)
def get_d2(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q: float | NDArray[np.float64]):
    return (np.log(s / k) + (r - q - iv ** 2 / 2) * t) / (iv * np.sqrt(t))
    # return get_d1(s, k, t, iv, r, q) - iv * np.sqrt(t)


@nb.njit(fastmath=True)
def get_d1_derivative(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q: float | NDArray[np.float64]):
    d1 = get_d1(s, k, t, iv, r, q)
    return np.exp(-d1**2/2) / np.sqrt(2 * np.pi)


@nb.njit(fastmath=True)
def price_call(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q: float | NDArray[np.float64]):
    d1 = get_d1(s, k, t, iv, r, q)
    d2 = d1 - iv * np.sqrt(t)
    return s * np.exp(-q * t) * ndtr_numba_v(d1) - k * np.exp(-r * t) * ndtr_numba_v(d2)


@nb.njit(fastmath=True)
def price_put(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q: float | NDArray[np.float64]):
    d1 = get_d1(s, k, t, iv, r, q)
    d2 = d1 - iv * np.sqrt(t)
    return k * np.exp(-r * t) * ndtr_numba_v(-d2) - s * np.exp(-q * t) * ndtr_numba_v(-d1)


@nb.njit(fastmath=True)
def delta_call(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q):
    d1 = get_d1(s, k, t, iv, r, q)
    return np.exp(-q * t) * ndtr_numba_v(d1)


@nb.njit(fastmath=True)
def delta_put(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q):
    d1 = get_d1(s, k, t, iv, r, q)
    return np.exp(-q * t) * (ndtr_numba_v(d1) - 1)


@nb.njit(fastmath=True)
def gamma(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q: float | NDArray[np.float64]):
    d1_drv = get_d1_derivative(s, k, t, iv, r, q)
    return np.exp(-q * t) * d1_drv / (s * iv * np.sqrt(t))


@nb.njit(fastmath=True)
def get_vega(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], iv: float | NDArray[np.float64], r: float | NDArray[np.float64], q):
    d1_drv = get_d1_derivative(s, k, t, iv, r, q)
    return s * np.exp(-q * t) * np.sqrt(t) * d1_drv / 100

def get_price_cuda(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], v_is_call: NDArray[int], iv: float | NDArray[np.float64], dividends: List[Dividend], calculation_date: date | NDArray[date], yield_curve:ZeroCurveData = None, equity: Equity=None, cpu=False) -> ndarray[Any, dtype[Any]]:
    if not isinstance(calculation_date, (date, datetime)):
        v = np.zeros_like(s)
        unique_dt, inverse = np.unique(calculation_date, return_inverse=True)
        for i, dt in enumerate(unique_dt):
            ix = inverse == i
            v[ix] = get_price_cuda(s[ix], k[ix], t[ix], v_is_call[ix], iv[ix], dividends=dividends, calculation_date=dt, yield_curve=yield_curve, equity=equity, cpu=cpu)
        return v
    else:
        yield_curve = yield_curve or YieldCurve().get_zero_curve(calculation_date, equity)
        div_amounts, div_times = dividends2amount_times(dividends, calculation_date)
        if not cpu:
            return np.array(merlin.get_v_fd_price(
                spots=s, strikes=k, tenors=t, sigmas=iv, v_is_call=v_is_call, rates_curve=yield_curve.rates, rates_times=yield_curve.times, div_amounts=div_amounts, div_times=div_times,
                time_steps=200, space_steps=200
            ))
        else:
            return np.asarray(merlin.get_v_fd_price_cpu(
                spots=s, strikes=k, tenors=t, sigmas=iv, v_is_call=v_is_call, rates_curve=yield_curve.rates, rates_times=yield_curve.times, div_amounts=div_amounts, div_times=div_times,
                time_steps=200, space_steps=200
            ))

def get_delta_cuda(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], v_is_call: NDArray[int], iv: NDArray[np.float64], dividends: List[Dividend], calculation_date: date | NDArray[date], yield_curve:ZeroCurveData = None, equity: Equity=None) -> ndarray[Any, dtype[Any]]:
    if not isinstance(calculation_date, date):
        v = np.zeros_like(s)
        unique_dt, inverse = np.unique(calculation_date, return_inverse=True)
        for i, dt in enumerate(unique_dt):
            ix = inverse == i
            v[ix] = get_delta_cuda(s[ix], k[ix], t[ix], v_is_call[ix], iv=iv[ix], dividends=dividends, calculation_date=dt, yield_curve=yield_curve, equity=equity)
        return v
    else:
        yield_curve = yield_curve or YieldCurve().get_zero_curve(calculation_date, equity)
        div_amounts, div_times = dividends2amount_times(dividends, calculation_date)
        return np.array(merlin.get_v_fd_delta(
            spots=s, strikes=k, tenors=t, v_is_call=v_is_call, sigmas=iv, rates_curve=yield_curve.rates, rates_times=yield_curve.times, div_amounts=div_amounts, div_times=div_times,
        ))

def get_vega_cuda(s: float | NDArray[np.float64], k: float | NDArray[np.float64], t: float | NDArray[np.float64], v_is_call: NDArray[int], iv: NDArray[np.float64], dividends: List[Dividend], calculation_date: date | NDArray[date], yield_curve:ZeroCurveData = None, equity: Equity=None) -> ndarray[Any, dtype[Any]]:
    if not isinstance(calculation_date, date):
        v = np.zeros_like(s)
        unique_dt, inverse = np.unique(calculation_date, return_inverse=True)
        for i, dt in enumerate(unique_dt):
            ix = inverse == i
            v[ix] = get_vega_cuda(s[ix], k[ix], t[ix], v_is_call[ix], iv[ix], dividends=dividends, calculation_date=dt, yield_curve=yield_curve, equity=equity)
        return v
    else:
        yield_curve = yield_curve or YieldCurve().get_zero_curve(calculation_date, equity)
        div_amounts, div_times = dividends2amount_times(dividends, calculation_date)
        return np.array(merlin.get_v_fd_vega(
            spots=s, strikes=k, tenors=t, v_is_call=v_is_call, sigmas=iv, rates_curve=yield_curve.rates, rates_times=yield_curve.times, div_amounts=div_amounts, div_times=div_times,
        ))


_SQRT2 = math.sqrt(2)
NPY_SQRT1_2 = 1.0 / np.sqrt(2)


@njit(cache=True, fastmath=True)
def ndtr_numba_v(arr: NDArray[np.float64]):
    if isinstance(arr, float):
        return ndtr_numba(arr)

    res = np.zeros_like(arr)
    for i, a in enumerate(arr):
        res[i] = ndtr_numba(a)
    return res


@njit(cache=True, fastmath=True)
def ndtr_numba(a):
    if np.isnan(a):
        return np.nan

    x = a * NPY_SQRT1_2
    z = fabs(x)

    if (z < NPY_SQRT1_2):
        y = 0.5 + 0.5 * erf(x)

    else:
        y = 0.5 * erfc(z)

        if (x > 0):
            y = 1.0 - y

    return y


def cdf(x):
    """mu=0, sigma=1"""
    return 0.5 * (1 + math.erf(x / _SQRT2))


@dataclass
class TestSample:
    calculationDate: date
    expiry: date
    strike: Decimal
    right: OptionRight | str
    price: float = None
    iv: float = None
    contract: OptionContract = None

    def __post_init__(self):
        self.contract = OptionContract('contract', 'FDX', self.expiry, self.strike, self.right)
        self.option = Option(self.contract, self.calculationDate)


def price_calcs():
    from options.helper import get_tenor
    calculationDate = date(2024, 3, 14)

    for sample in [
        TestSample(calculationDate, date(2024, 4, 5), Decimal('28.0'), OptionRight.put, iv=0.32),
        TestSample(calculationDate, date(2025, 4, 5), Decimal('28.0'), OptionRight.put, iv=0.32),
        TestSample(calculationDate, date(2024, 4, 5), Decimal('38.0'), OptionRight.put, iv=0.32),
        TestSample(calculationDate, date(2024, 4, 5), Decimal('28.0'), OptionRight.put, iv=0.16),

        TestSample(calculationDate, date(2024, 4, 5), Decimal('28.0'), OptionRight.call, iv=0.32),
        TestSample(calculationDate, date(2025, 4, 5), Decimal('28.0'), OptionRight.call, iv=0.32),
        TestSample(calculationDate, date(2024, 4, 5), Decimal('38.0'), OptionRight.call, iv=0.32),
        TestSample(calculationDate, date(2024, 4, 5), Decimal('28.0'), OptionRight.call, iv=0.16),
    ]:
        ql.Settings.instance().evaluationDate = ql.Date(calculationDate.day, calculationDate.month, calculationDate.year)
        s = 28.15
        option = sample.option

        ql_price = option.npv(sample.iv, s, calculationDate)

        t = np.array([get_tenor(sample.expiry, calculationDate)])
        r = option.riskFreeRateQuote.value()
        q = option.dividendRateQuote.value()
        f: Callable = price_put if sample.right == OptionRight.put else price_call
        py_price = f(s, float(sample.strike), t, sample.iv, r, q)[0]

        iv_py_price = option.iv(py_price, s, calculationDate)
        iv_ql_price = option.iv(ql_price, s, calculationDate)

        print(f'QL Price: {ql_price}, '
              f'py_price: {py_price}, '
              f'diff: {ql_price - py_price}, '
              f'diff IV: {iv_ql_price - iv_py_price}, '
              )


def test_price_iv_price_loop():
    from options.helper import df2iv
    calculation_date = date(2024, 6, 26)
    s = 296.5
    # sample = TestSample(calculation_date, date(2026, 1, 16), Decimal('400.0'), OptionRight.call, price=11.50)
    for sample in [
        TestSample(calculation_date, date(2026, 1, 16), Decimal('400.0'), OptionRight.call, price=11.50),
        ]:
        ql.Settings.instance().evaluationDate = ql.Date(calculation_date.day, calculation_date.month, calculation_date.year)

        option = sample.option
        rate = option.riskFreeRateQuote.value()
        dividends = get_dividends(equity.symbol.upper(), v_calc_date[0], v_calc_date[-1])

        # t = np.array([get_tenor(sample.expiry, calculation_date)])
        # r = option.riskFreeRateQuote.value()
        # q = option.dividendRateQuote.value()
        # f: Callable = Option.price_put if sample.right == OptionRight.put else Option.price_call
        # py_price = f(s, float(sample.strike), t, sample.iv, r, q)[0]

        # iv_py_price = option.iv(py_price, s, calculation_date)
        iv_ql_price = option.iv(sample.price, s, calculation_date)

        df = pd.DataFrame({
            'ts': [datetime.fromisoformat(calculation_date.isoformat())],
            'expiry': [date(2026, 1, 16)],
            'strike': [400.0],
            'right': [OptionRight.call],

            'spot': [s],
            'price': [11.50],
        }).set_index(['ts', 'expiry', 'strike', 'right'])
        # ps2iv_ivs = df['bid_iv'] = df2iv(df, price_col_nm='price', rate=rate, dividends=dividends, calculation_date=)
        # ps2iv_ivs = df.apply(partial(ps2iv, price_col=f'price', calendar=calendar, day_count=day_count, rate=rate, dividend=dividend_yield), axis=1)

        ql_p_of_iv_ql = option.npv(iv_ql_price, s, calculation_date)
        # ql_p_of_ps2iv_ql = option.npv(ps2iv_ivs[0], s, calculation_date)

        print(
            # f'py_price: {py_price}, '
            # f'diff: {ql_price - py_price}, '
            f'Diff price: {sample.price - ql_p_of_iv_ql}, '
            )


def speed_compare_cdf_calc():
    from scipy.special import ndtr
    from scipy.stats import norm
    import timeit

    for x in np.arange(-5, 5, 0.01):
        print(f'{x}: {ndtr(x)} {ndtr(x) == norm.cdf(x) == ndtr_numba(x)}')
        # print(f'{x}: {ndtr(x)} {ndtr(x) == norm.cdf(x) == ndtr_numba(x)}')

    def cdf_slow():
        return norm.cdf(np.arange(-5, 5, 0.1))

    def cdf_fast():
        return ndtr(np.arange(-5, 5, 0.1))

    def cdf_faster():
        return ndtr_numba_v(np.arange(-5, 5, 0.1))

    n = 1_000_000
    print(timeit.timeit('cdf_slow()', setup="from __main__ import cdf_slow", number=n), 'us')
    # 30
    print(timeit.timeit('cdf_fast()', setup="from __main__ import cdf_fast", number=n), 'us')
    # 3
    print(timeit.timeit('cdf_faster()', setup="from __main__ import cdf_faster", number=n), 'us')
    # 3


def _to_ql_date(d: date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)

def build_ql_zero_curve(
    calculation_date: date,
    curve: ZeroCurveData,
    day_count: ql.DayCounter = ql.Actual365Fixed(),
    calendar: ql.Calendar = ql.UnitedStates(ql.UnitedStates.NYSE),
) -> ql.YieldTermStructureHandle:
    """
    Build a QuantLib ZeroCurve from (date, zero_rate) points.
    Rates are interpreted as continuously-compounded zero rates with Actual/365.
    """
    calc = _to_ql_date(calculation_date)
    ql.Settings.instance().evaluationDate = calc

    dates = ql.DateVector([_to_ql_date(calculation_date + timedelta(days=t*365)) for t in curve.times] if curve else [calc, _to_ql_date(calculation_date + timedelta(days=999))])
    rates = ql.DoubleVector(curve.rates if curve else [0, 0])

    zc = ql.ZeroCurve(dates, rates, day_count, calendar)
    return ql.YieldTermStructureHandle(zc)

def _make_american_option_with_discrete_divs(
    calculation_date: date,
    expiry: date,
    strike: float,
    right: OptionRight,
) -> ql.OneAssetOption:
    """
    Some QuantLib-Python builds don't expose ql.DividendVanillaOption.
    We support both:
      1) ql.DividendVanillaOption (if available)
      2) ql.VanillaOption, with the dividend schedule passed to the FD engine (if your binding supports it)
    """
    calc_dt = _to_ql_date(calculation_date)
    exp_dt = _to_ql_date(expiry)

    option_type = ql.Option.Call if right == OptionRight.call else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(option_type, float(strike))
    exercise = ql.AmericanExercise(calc_dt, exp_dt)
    return ql.VanillaOption(payoff, exercise)

def _build_dividend_schedule(
    cash_dividends: Sequence[Dividend],
    expiry: date,
) -> ql.DividendSchedule:
    divs = [d for d in cash_dividends if d.ex_date <= expiry]
    div_dates = [_to_ql_date(d.ex_date) for d in divs]
    div_amounts = [float(d.amount) for d in divs]
    return ql.DividendSchedule([ql.FixedDividend(a, dt) for a, dt in zip(div_amounts, div_dates)])

def price_american_option_ql_discrete_divs(
    calculation_date: date,
    expiry: date,
    spot: float,
    strike: float,
    iv: float,
    right: OptionRight,
    cash_dividends: Sequence[Dividend],
    zero_curve: ZeroCurveData | None,
    steps_time: int = 200,
    steps_grid: int = 200,
) -> float:
    """
    American option with discrete cash dividends + explicit yield curve.

    Important:
      - If your QuantLib-Python build does NOT expose DividendVanillaOption, we fall back to
        VanillaOption and try FD engines that accept a DividendSchedule.
      - If none of the engine constructors in your build accept a dividend schedule, you’ll need
        to upgrade QuantLib-Python/QuantLib (or use a different approach).
    """
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    day_count = ql.Actual365Fixed()

    calc_dt = _to_ql_date(calculation_date)
    ql.Settings.instance().evaluationDate = calc_dt

    option = _make_american_option_with_discrete_divs(
        calculation_date=calculation_date,
        expiry=expiry,
        strike=strike,
        right=right,
    )

    dividend_schedule = _build_dividend_schedule(cash_dividends, expiry)

    spot_h = ql.QuoteHandle(ql.SimpleQuote(float(spot)))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(calc_dt, calendar, ql.QuoteHandle(ql.SimpleQuote(float(iv))), day_count)
    )
    r_ts = build_ql_zero_curve(calculation_date, zero_curve, day_count=day_count, calendar=calendar)
    process = ql.BlackScholesProcess(spot_h, r_ts, vol_ts)
    engine = ql.FdBlackScholesVanillaEngine(process, dividend_schedule, steps_time, steps_grid)

    option.setPricingEngine(engine)
    return float(option.NPV())



def _pct_diff(a: float, b: float) -> float:
    """
    Percent difference vs reference a:
        100 * (b - a) / a
    """
    if a == 0.0:
        return np.inf if b != 0.0 else 0.0
    return 100.0 * (b - a) / a


def run_american_ql_vs_cuda_tests() -> None:
    calculation_date = date(2024, 6, 27)
    spot = 100.0

    # cash_divs = get_dividends('FDX', calculation_date, calculation_date + timedelta(days=900))
    cash_divs = []
    yield_curve = YieldCurve().get_zero_curve(calculation_date)
    # yield_curve = ZeroCurveData([0,3], [0.05, 0.05])
    # yield_curve = ZeroCurveData([0,3], [0.0, 0.0])

    # ~20 test cases varying right, iv, strike, tenor
    rights = [OptionRight.call, OptionRight.put]
    ivs = [0.12, 0.25, 0.35, 0.5]
    strikes = [70.0, 85.0, 100.0, 115.0, 130.0]
    tenors_days = [30, 60, 90, 180, 365, 730]

    # Pick 20 deterministic combinations (no randomness)
    cases = []
    for i in range(20):
        right = rights[i % len(rights)]
        iv = ivs[(i // 2) % len(ivs)]
        strike = strikes[i % len(strikes)]
        days = tenors_days[(i // 3) % len(tenors_days)]
        expiry = calculation_date + timedelta(days=int(days))
        cases.append((right, iv, strike, days, expiry))

    print("Running %d American option tests: QuantLib (discrete divs + curve) vs get_price_cuda", len(cases))

    for idx, (right, iv, strike, days, expiry) in enumerate(cases, start=1):
        # QuantLib
        price_ql = price_american_option_ql_discrete_divs(
            calculation_date=calculation_date,
            expiry=expiry,
            spot=spot,
            strike=strike,
            iv=iv,
            right=right,
            cash_dividends=cash_divs,
            zero_curve=yield_curve,
            steps_time=200,
            steps_grid=200,
        )

        # CUDA FD pricer
        t = np.array([days / 365.0], dtype=np.float64)
        v_is_call = np.array([1 if right == OptionRight.call else 0], dtype=np.int32)

        price_cuda = float(
            get_price_cuda(
                s=np.array([spot], dtype=np.float64),
                k=np.array([strike], dtype=np.float64),
                t=t,
                v_is_call=v_is_call,
                iv=np.array([iv], dtype=np.float64),
                dividends=cash_divs,
                calculation_date=calculation_date,
                yield_curve=yield_curve,
                cpu=True,
            )[0]
        )

        pct = _pct_diff(price_ql, price_cuda)
        if abs(pct) > 0.1 and price_ql >= 0.01:
            print(
                f"case={idx} spot={spot:.2f} right={right} iv={iv:.3f} K={strike:.2f} tenor={t[0]:.3f}  QL={price_ql:.6f} CUDA={price_cuda:.6f}  pct_diff(CUDA-QL)={pct:.4f}",
            )



if __name__ == '__main__':
    run_american_ql_vs_cuda_tests()
    # from options.helper import get_tenor, get_dividend_amount_times, dividends2amount_times
    # import py_vollib.black_scholes_merton.implied_volatility
    # import py_vollib_vectorized
    # from statistics import NormalDist
    # # price_calcs()
    # from datetime import timedelta
    #
    # c = OptionContract.from_ib_symbol("DAL   250705C00050000")
    # c = OptionContract.from_ib_symbol("DAL   250705P00050000")
    #
    # calculationDate = date(2024, 6, 27)
    # expiry = c.expiry
    # strike = float(c.strike)
    # right = c.right
    #
    # ql.Settings.instance().evaluationDate = ql.Date(calculationDate.day, calculationDate.month, calculationDate.year)
    # divs = get_dividends('DAL', calculationDate, c.expiry + timedelta(days=1))
    #
    # p = 15.0
    # s = 50.0
    #
    # # EU Analytical same as super fast py_vollib
    # option_eu = Option(c, calculationDate, option_constructor=ql.VanillaOption, style=Style.European, optionPricingModel=OptionPricingModel.AnalyticEuropeanEngine)
    # print("EU Q Option %s" % option_eu.iv(p, s,calculationDate))
    #
    # # American - different engine - same result - Identical - lower IV than euro because early exercise right.
    # # print("AM Q BAW Option %s" % Option(c, calculationDate, option_constructor=ql.VanillaOption, style=Style.American, optionPricingModel=OptionPricingModel.BaroneAdesiWhaleyApproximationEngine).iv(p, s, calculationDate))
    # # print("AM Q CRR Option %s" % Option(c, calculationDate, option_constructor=ql.VanillaOption, style=Style.American, optionPricingModel=OptionPricingModel.CoxRossRubinstein).iv(p, s, calculationDate))
    # print("AM Div Q FD Option %s" % Option(c, calculationDate, option_constructor=ql.VanillaOption, style=Style.American, optionPricingModel=OptionPricingModel.FdBlackScholesVanillaEngine).iv(p, s, calculationDate))
    #
    # # American no divs, no Q = identical. Put higher IV than american with DIV because no Div and Call has lower IV than with Div
    # print("AM NoDiv FD Option %s" % Option(c, calculationDate, option_constructor=ql.VanillaOption, style=Style.AmericanDiv, dividends=[], optionPricingModel=OptionPricingModel.FdBlackScholesVanillaEngine).iv(p, s, calculationDate))
    # # print("AM Div2 Q Option %s" % Option(c, calculationDate, option_constructor=ql.DividendVanillaOption, style=Style.AmericanDiv, dividends=[], optionPricingModel=OptionPricingModel.FdBlackScholesVanillaEngineDivSchedule).iv(p, s, calculationDate))
    # # print("AM Div ! BAW Option %s" % Option(c, calculationDate, option_constructor=ql.DividendVanillaOption, style=Style.AmericanDiv, dividends=[], optionPricingModel=OptionPricingModel.BaroneAdesiWhaleyApproximationEngine).iv(p, s, calculationDate))
    #
    # # Discrete Dividends lead to significantly lower IV - all Identical IV
    # # Call = Higher IV than w/o div ; Put = Lower IV than w/o div
    # # print("AM DivFixed BAW Option %s" % Option(c, calculationDate, option_constructor=ql.DividendVanillaOption, style=Style.AmericanDiv, dividends=divs, optionPricingModel=OptionPricingModel.BaroneAdesiWhaleyApproximationEngine).iv(p, s, calculationDate))
    # # The one to build now
    # print("AM DivFixed FD Option %s" % Option(c, calculationDate, option_constructor=ql.VanillaOption, style=Style.AmericanDiv, dividends=divs, optionPricingModel=OptionPricingModel.FdBlackScholesVanillaEngine).iv(p, s, calculationDate))
    # print("AM DivFixed2 Option %s" % Option(c, calculationDate, option_constructor=ql.DividendVanillaOption, style=Style.AmericanDiv, dividends=divs, optionPricingModel=OptionPricingModel.FdBlackScholesVanillaEngineDivSchedule).iv(p, s, calculationDate))


    # def get_v_iv(p: NDArray[np.float64], s: NDArray[np.float64], k: NDArray[np.float64], t: NDArray[np.float64], r: float, right: np.ndarray, q: float) -> np.ndarray:
    #     return py_vollib_vectorized.vectorized_implied_volatility(p, s, k, t, r, right, q=q, model='black_scholes_merton', return_as='numpy')

    # print("EU pyvoillib Q Option %f", get_v_iv(
    #     np.array([p]),
    #     np.array([s]),
    #     np.array([strike]),
    #     np.array([get_tenor(expiry, calculationDate)]),
    #     option_eu.riskFreeRateQuote.value(),
    #     np.array([right[0]]),
    #     option_eu.dividendRateQuote.value()
    # ))

    # model_price = option.npv(model_iv, s, calculationDate)
    # t = np.array([get_tenor(expiry, calculationDate)])
    # r = option.riskFreeRateQuote.value()
    # q = option.dividendRateQuote.value()
    # f: Callable = price_put if right == OptionRight.put else price_call
    # # model_price2 = f(s, np.array([float(strike)]), t, np.array([model_iv]), r, q)
    # # print(model_price - model_price2[0])
    #
    # print(option.iv(5.05, s, calculationDate))
    #
    # # print(f'NPV: {option.npv(0.356, 310.1, calculationDate)}')
    # #
    # # option.volQuote.setValue(0.411)
    # #
    # # print(option.iv(0.8, 45, date(2023, 6, 5)))
    # # option.underlyingQuote.setValue(17.49825)
    # # #
    # # # print(f'Delta: {option.delta()}')
    # # # print(f'Theta: {option.theta()}')
    # # #
    # # # print(f'ThetaPerDay: {option.thetaPerDay()}')
    # # # print(f'Vega: {option.vega()}')
