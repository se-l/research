import QuantLib as ql

from options.typess.enums import OptionPricingModel


def get_bsm(calculation_date, spot_quote, hv_quote, rf_quote, dividend_rate_quote, calendar, day_count):
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, ql.QuoteHandle(rf_quote), day_count)
    )
    dividend_yield = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, ql.QuoteHandle(dividend_rate_quote), day_count)
    )
    flat_vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(calculation_date, calendar, ql.QuoteHandle(hv_quote), day_count)
    )
    return ql.BlackScholesMertonProcess(ql.QuoteHandle(spot_quote), dividend_yield, flat_ts, flat_vol_ts)


def engined_option(option, bsm_process, optionPricingModel: OptionPricingModel = OptionPricingModel.CoxRossRubinstein, steps=200) -> ql.VanillaOption:
    binomial_engine = {
        OptionPricingModel.CoxRossRubinstein: lambda: ql.BinomialVanillaEngine(bsm_process, "crr", steps),
        OptionPricingModel.AnalyticEuropeanEngine: lambda: ql.AnalyticEuropeanEngine(bsm_process),
        OptionPricingModel.FdBlackScholesVanillaEngine: lambda: ql.FdBlackScholesVanillaEngine(bsm_process),
    }[optionPricingModel]()
    option.setPricingEngine(binomial_engine)
    return option


def finite_difference_approx(quote, option, d_pct=0.01, derive='NPV', method='central', d1perturbance=None):
    h0 = quote.value()
    quote.setValue(h0 * (1 + d_pct))
    # if hasattr(option, derive):
    if derive in ['vega']:
        p_plus = finite_difference_approx(d1perturbance, option, derive='NPV')  # VEGA
    else:
        p_plus = option.__getattribute__(derive)()
    quote.setValue(h0 * (1 - d_pct))
    # if hasattr(option, derive):
    if derive in ['vega']:
        p_minus = finite_difference_approx(d1perturbance, option, derive='NPV')  # VEGA
    else:
        p_minus = option.__getattribute__(derive)()
    quote.setValue(h0)
    return (p_plus - p_minus) / (2 * h0 * d_pct)


def finite_difference_approx_time(calculation_date, spot_quote, hv_quote, rf_quote, dividend_rate_quote, calendar, day_count, derive='NPV', n_days=1, method='forward'):
    values = []
    for dt in [calculation_date, calculation_date + n_days]:
        payoff = ql.PlainVanillaPayoff(option_type, strike_price)
        am_exercise = ql.AmericanExercise(dt, maturity_date)
        option = ql.VanillaOption(payoff, am_exercise)
        bsm_process = get_bsm(dt, spot_quote, hv_quote, rf_quote, dividend_rate_quote, calendar, day_count)
        engined_option(option, bsm_process)
        # if hasattr(option, derive):
        if derive in ['vega']:
            values.append(finite_difference_approx(hv_quote, option, 0.01))  # VEGA
        else:
            values.append(option.__getattribute__(derive)())
    return (values[0] - values[-1]) / n_days
