from functools import lru_cache

import QuantLib as ql
import numpy as np

from datetime import datetime
from datetime import date
from typing import Iterable
from dataclasses import dataclass
from decimal import Decimal

from shared.constants import DiscountRateMarket, DividendYield
from options.ql_helper import engined_option
from options.typess.enums import OptionPricingModel, OptionRight
from options.typess.option_contract import OptionContract


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


class Option:
    multiplier = 100

    def __init__(self, optionContract: OptionContract, calculationDate: date = date.today(), price_underlying: float = 0, volatility: float = 0):
        self.optionContract = optionContract
        self.expiry = self.optionContract.expiry
        self.right = self.optionContract.right
        self.price = 0

        self.calculationDate = calculationDate
        self.calculationDateQl = ql.Date(calculationDate.day, calculationDate.month, calculationDate.year)
        self.strike = float(optionContract.strike)
        self.optionType = {OptionRight.put: ql.Option.Put,
                           OptionRight.call: ql.Option.Call}[optionContract.right]
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
        self.dividendRateQuote = ql.SimpleQuote(DividendYield.get(optionContract.underlying_symbol.upper(), 0))
        self.dividendRateQuoteHandle = ql.QuoteHandle(self.dividendRateQuote)

        self.am_option = engined_option(ql.VanillaOption(self.payoff, self.am_exercise), self.get_bsm(), optionPricingModel=OptionPricingModel.CoxRossRubinstein)
        self.eu_option = engined_option(ql.VanillaOption(self.payoff, self.eu_exercise), self.get_bsm(), optionPricingModel=OptionPricingModel.AnalyticEuropeanEngine)

    def SetPriceUnderlying(self, priceUnderlying: float):
        self.underlyingQuote.setValue(priceUnderlying)

    def SetVolatility(self, volatility: float):
        self.volQuote.setValue(volatility)

    def SetPriceAndVolatility(self, price: float):
        self.price = price
        self.volQuote.setValue(self.iv(price, self.underlyingQuote.value(), self.calculationDate))

    def SetPrice(self, price: float):
        self.price = price

    def SetEvaluationDateToCalcDate(self, calculationDateQl: ql.Date = None):
        if isinstance(calculationDateQl, date):
            _calculationDateQl = ql.Date(calculationDateQl.day, calculationDateQl.month, calculationDateQl.year)
        elif calculationDateQl is None:
            _calculationDateQl = self.calculationDateQl
        else:
            _calculationDateQl = calculationDateQl

        if ql.Settings.instance().evaluationDate != _calculationDateQl:
            ql.Settings.instance().evaluationDate = _calculationDateQl
            self.calculationDateQl = _calculationDateQl
            self.calculationDate = date(_calculationDateQl.year(), _calculationDateQl.month(), _calculationDateQl.dayOfMonth())

    @lru_cache(maxsize=1024)
    def iv(self, priceOption: float, priceUnderlying: float, calculationDate: date) -> float:
        """
        Calculate implied volatility from a series of option prices.
        :param ps: A series of option prices.
        :return: A series of implied volatilities.
        """
        _calculationDateQl = ql.Date(calculationDate.day, calculationDate.month, calculationDate.year) if calculationDate else self.calculationDateQl
        self.SetEvaluationDateToCalcDate(_calculationDateQl)

        if priceUnderlying != self.underlyingQuote.value():
            self.underlyingQuote.setValue(priceUnderlying)

        try:
            return self.eu_option.impliedVolatility(priceOption, self.get_bsm())
        except Exception as e:
            if 'root not bracketed' not in str(e):
                print(e)
            return 0

    def greeks(self, params: GreekParameters):
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

    def greeks_vec(self, pricesUnderlying: np.ndarray, volatilities: np.ndarray, dateTimes: Iterable[datetime]) -> Iterable[Greeks]:
        assert len(pricesUnderlying) == len(volatilities) == len(dateTimes)
        return (self.greeks(GreekParameters(priceUnderlying, vola, dt.date())) for (priceUnderlying, vola, dt) in zip(pricesUnderlying, volatilities, dateTimes))

    def get_bsm(self):
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(self.calculationDateQl, self.riskFreeRateQuoteHandle, self.dayCount))
        dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(self.calculationDateQl, self.dividendRateQuoteHandle, self.dayCount))
        flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(self.calculationDateQl, self.calendar, self.volQuoteHandle, self.dayCount))
        return ql.BlackScholesMertonProcess(self.underlyingQuoteHandle, dividend_yield, flat_ts, flat_vol_ts)

    @lru_cache(maxsize=2**10)
    def npv(self, vol: float, priceUnderlying: float, calculationDate: datetime.date):
        """
        # For very large IV values, npv() starts failing... for far OTM/ITM, assume value to be intrinsic only...
        """
        self.SetEvaluationDateToCalcDate(calculationDate)
        if priceUnderlying != self.underlyingQuote.value():
            self.underlyingQuote.setValue(priceUnderlying)
        if vol != self.volQuote.value():
            self.volQuote.setValue(vol)

        # delta = self.eu_option.delta()
        # if abs(delta) < 0.02:
        #     npv = 0
        # elif abs(delta) > 0.98:
        #     npv = self.underlyingQuote.value() - self.strike if self.right == 'call' else self.strike - self.underlyingQuote.value()
        # else:
        if calculationDate == self.expiry:
            return self.intrinsic_value()
        return self.eu_option.NPV()

    @lru_cache(maxsize=2**10)
    def delta(self, vol, priceUnderlying, calculationDate):
        self.SetEvaluationDateToCalcDate(calculationDate)
        if priceUnderlying != self.underlyingQuote.value():
            self.underlyingQuote.setValue(priceUnderlying)
        if vol != self.volQuote.value():
            self.volQuote.setValue(vol)

        return self.eu_option.delta()

    def intrinsic_value(self, priceUnderlying: float = None):
        if priceUnderlying:
            self.underlyingQuote.setValue(priceUnderlying)
        if self.right == 'call':
            return max(self.underlyingQuote.value() - self.strike, 0)
        elif self.right == 'put':
            return max(self.strike - self.underlyingQuote.value(), 0)

    def is_otm(self, priceUnderlying: float):
        return self.intrinsic_value(priceUnderlying) == 0

    def extrinsic_value(self):
        return self.npv() - self.intrinsic_value()

    def summary(self):
        return {
            'Contract': str(self.optionContract),
            'IV': self.volQuote.value(),
            'NPV': self.npv(),
            'IntrinsicValue': self.intrinsic_value(),
            'ExtrinsicValue': self.extrinsic_value(),
            'Delta': self.delta(),
            'SpotUnderlying': self.underlyingQuote.value(),
        }

    def __repr__(self):
        return self.optionContract.ib_symbol()


if __name__ == '__main__':
    calculationDate = date(2024, 2, 27)
    contract = OptionContract('contract', 'HPE', date(2026, 1, 16), Decimal('35.0'), OptionRight.put)
    ql.Settings.instance().evaluationDate = ql.Date(calculationDate.day, calculationDate.month, calculationDate.year)
    option = Option(contract)

    option.iv(15.055, 22.5, calculationDate)
    option.iv(14.915, 22.5, calculationDate)
    option.iv(14.915, 21.5, calculationDate)
    print(f'NPV: {option.npv(0.356, 310.1, calculationDate)}')
    
    option.volQuote.setValue(0.411)


    # print(option.iv(0.8, 45, date(2023, 6, 5)))
    # option.underlyingQuote.setValue(17.49825)


    print(f'Delta: {option.delta()}')
    print(f'Theta: {option.theta()}')

    # print(f'ThetaPerDay: {option.thetaPerDay()}')
    # print(f'Vega: {option.vega()}')
