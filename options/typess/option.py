import math
import pickle

import QuantLib as ql
import numpy as np
import pandas as pd

from datetime import datetime, date
from typing import Iterable, Callable, Dict
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from math import fabs, erf, erfc
from numba import njit

from options.typess.security import Security, SecurityDataSnap
from options.typess.scenario import Scenario
from shared.constants import DiscountRateMarket
from options.ql_helper import engined_option
from options.typess.enums import OptionPricingModel, OptionRight, TickType, Resolution
from options.typess.option_contract import OptionContract
import numba as nb


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


class Option(Security):
    # Ideally refactored to combine with OptionContract
    multiplier = 100
    accuracy = 1.0e-4
    max_iterations = 100
    minVol = 0.0001
    maxVol = 4.0

    def __init__(self, option_contract: OptionContract, calculation_date: date, price_underlying: float = 0, volatility: float = 0):
        from options.helper import get_dividend_yield

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
        self.eu_exercise = ql.EuropeanExercise(self.maturityDate)
        self.am_exercise = ql.AmericanExercise(self.calculationDateQl, self.maturityDate)

        self.underlyingQuote = ql.SimpleQuote(price_underlying)
        self.underlyingQuoteHandle = ql.QuoteHandle(self.underlyingQuote)

        self.volQuote = ql.SimpleQuote(volatility)
        self.volQuoteHandle = ql.QuoteHandle(self.volQuote)

        self.riskFreeRateQuote = ql.SimpleQuote(DiscountRateMarket)
        self.riskFreeRateQuoteHandle = ql.QuoteHandle(self.riskFreeRateQuote)

        # Annualized Yields
        self.dividendRateQuote = ql.SimpleQuote(get_dividend_yield(option_contract.underlying_symbol))
        self.dividendRateQuoteHandle = ql.QuoteHandle(self.dividendRateQuote)

        self.am_option = engined_option(ql.VanillaOption(self.payoff, self.am_exercise), self.get_bsm(), optionPricingModel=OptionPricingModel.CoxRossRubinstein)
        self.eu_option = engined_option(ql.VanillaOption(self.payoff, self.eu_exercise), self.get_bsm(), optionPricingModel=OptionPricingModel.AnalyticEuropeanEngine)

    def __eq__(self, other):
        return self.optionContract.__repr__() == other.optionContract.__repr__()

    def __ge__(self, other):
        return self.optionContract.__repr__() >= other.optionContract.__repr__()

    def __gt__(self, other):
        return self.optionContract.__repr__() > other.optionContract.__repr__()

    def __lt__(self, other):
        return self.optionContract.__repr__() < other.optionContract.__repr__()

    def __le__(self, other):
        return self.optionContract.__repr__() <= other.optionContract.__repr__()

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
            return self.eu_option.impliedVolatility(priceOption, bsm, self.accuracy, self.max_iterations, self.minVol, maxVol)
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
        dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(self.calculationDateQl, self.dividendRateQuoteHandle, self.dayCount))
        flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(self.calculationDateQl, self.calendar, self.volQuoteHandle, self.dayCount))
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


@nb.njit(fastmath=True)
def get_d1(s: float | np.ndarray, k: float | np.ndarray, t: float | np.ndarray, iv: float | np.ndarray, r: float | np.ndarray, q: float | np.ndarray):
    return (np.log(s / k) + (r - q + iv ** 2 / 2) * t) / (iv * np.sqrt(t))


@nb.njit(fastmath=True)
def get_d2(s: float | np.ndarray, k: float | np.ndarray, t: float | np.ndarray, iv: float | np.ndarray, r: float | np.ndarray, q: float | np.ndarray):
    return (np.log(s / k) + (r - q - iv ** 2 / 2) * t) / (iv * np.sqrt(t))
    # return get_d1(s, k, t, iv, r, q) - iv * np.sqrt(t)


@nb.njit(fastmath=True)
def get_d1_derivative(s: float | np.ndarray, k: float | np.ndarray, t: float | np.ndarray, iv: float | np.ndarray, r: float | np.ndarray, q: float | np.ndarray):
    d1 = get_d1(s, k, t, iv, r, q)
    return np.exp(-d1**2/2) / np.sqrt(2 * np.pi)


@nb.njit(fastmath=True)
def price_call(s: float | np.ndarray, k: float | np.ndarray, t: float | np.ndarray, iv: float | np.ndarray, r: float | np.ndarray, q: float | np.ndarray):
    d1 = get_d1(s, k, t, iv, r, q)
    d2 = d1 - iv * np.sqrt(t)
    return s * np.exp(-q * t) * ndtr_numba_v(d1) - k * np.exp(-r * t) * ndtr_numba_v(d2)


@nb.njit(fastmath=True)
def price_put(s: float | np.ndarray, k: float | np.ndarray, t: float | np.ndarray, iv: float | np.ndarray, r: float | np.ndarray, q: float | np.ndarray):
    d1 = get_d1(s, k, t, iv, r, q)
    d2 = d1 - iv * np.sqrt(t)
    return k * np.exp(-r * t) * ndtr_numba_v(-d2) - s * np.exp(-q * t) * ndtr_numba_v(-d1)


@nb.njit(fastmath=True)
def delta_call(s: float | np.ndarray, k: float | np.ndarray, t: float | np.ndarray, iv: float | np.ndarray, r: float | np.ndarray, q: float | np.ndarray):
    d1 = get_d1(s, k, t, iv, r, q)
    return np.exp(-q * t) * ndtr_numba_v(d1)


@nb.njit(fastmath=True)
def delta_put(s: float | np.ndarray, k: float | np.ndarray, t: float | np.ndarray, iv: float | np.ndarray, r: float | np.ndarray, q: float | np.ndarray):
    d1 = get_d1(s, k, t, iv, r, q)
    return np.exp(-q * t) * (ndtr_numba_v(d1) - 1)


@nb.njit(fastmath=True)
def gamma(s: float | np.ndarray, k: float | np.ndarray, t: float | np.ndarray, iv: float | np.ndarray, r: float | np.ndarray, q: float | np.ndarray):
    d1_drv = get_d1_derivative(s, k, t, iv, r, q)
    return np.exp(-q * t) * d1_drv / (s * iv * np.sqrt(t))


@nb.njit(fastmath=True)
def get_vega(s: float | np.ndarray, k: float | np.ndarray, t: float | np.ndarray, iv: float | np.ndarray, r: float | np.ndarray, q: float | np.ndarray):
    d1_drv = get_d1_derivative(s, k, t, iv, r, q)
    return s * np.exp(-q * t) * np.sqrt(t) * d1_drv / 100


_SQRT2 = math.sqrt(2)
NPY_SQRT1_2 = 1.0 / np.sqrt(2)


@njit(cache=True, fastmath=True)
def ndtr_numba_v(arr: np.ndarray):
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
        dividend_yield = option.dividendRateQuote.value()
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

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
        ps2iv_ivs = df['bid_iv'] = df2iv(df, price_col_nm='price', rate=rate, dividend_yield=dividend_yield)
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


if __name__ == '__main__':
    from options.helper import get_tenor
    from statistics import NormalDist
    # price_calcs()

    calculationDate = date(2024, 4, 5)
    expiry = date(2024, 4, 5)
    strike = Decimal('28.0')
    right = OptionRight.put
    contract = OptionContract('contract', 'FDX', expiry, strike, right)
    print(pickle.dumps(contract))
    ql.Settings.instance().evaluationDate = ql.Date(calculationDate.day, calculationDate.month, calculationDate.year)
    option = Option(contract, calculationDate)
    pickled = pickle.dumps(option)
    option2 = pickle.loads(pickled)
    print(option2 == option)
    dct1 = {option: 1}
    print(dct1[option2])
    s = 28.15
    model_iv = option.iv(1.05, s, calculationDate)
    model_price = option.npv(model_iv, s, calculationDate)
    t = np.array([get_tenor(expiry, calculationDate)])
    r = option.riskFreeRateQuote.value()
    q = option.dividendRateQuote.value()
    f: Callable = price_put if right == OptionRight.put else price_call
    model_price2 = f(s, np.array([float(strike)]), t, np.array([model_iv]), r, q)
    print(model_price - model_price2[0])

    print(option.iv(14.35, s, calculationDate))

    print(f'NPV: {option.npv(0.356, 310.1, calculationDate)}')

    option.volQuote.setValue(0.411)

    print(option.iv(0.8, 45, date(2023, 6, 5)))
    option.underlyingQuote.setValue(17.49825)
    #
    # print(f'Delta: {option.delta()}')
    # print(f'Theta: {option.theta()}')
    #
    # print(f'ThetaPerDay: {option.thetaPerDay()}')
    # print(f'Vega: {option.vega()}')
