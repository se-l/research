using Dates

# ── OptionStyle ───────────────────────────────────────────────────────────────
@enum OptionStyle begin
    option_style_american
    option_style_european
end

function option_style_from_string(s::AbstractString)::OptionStyle
    s == "american" && return option_style_american
    s == "european" && return option_style_european
    throw(ArgumentError("Unknown option style: $s"))
end


# ── OptionRight ───────────────────────────────────────────────────────────────
@enum OptionRight begin
    option_right_call
    option_right_put
end

function option_right_from_string(s::AbstractString)::OptionRight
    s in ("C", "call") && return option_right_call
    s in ("P", "put")  && return option_right_put
    throw(ArgumentError("Unknown option right: $s"))
end


# ── TickType ──────────────────────────────────────────────────────────────────
@enum TickType begin
    tick_type_trade
    tick_type_quote
    tick_type_iv_quote
    tick_type_iv_trade
    tick_type_volume
end

function tick_type_from_string(s::AbstractString)::TickType
    s == "trade"    && return tick_type_trade
    s == "quote"    && return tick_type_quote
    s == "iv_quote" && return tick_type_iv_quote
    s == "iv_trade" && return tick_type_iv_trade
    s == "volume"   && return tick_type_volume
    throw(ArgumentError("Unknown tick type: $s"))
end


# ── TickDirection ─────────────────────────────────────────────────────────────
@enum TickDirection begin
    tick_direction_up
    tick_direction_down
end

function tick_direction_from_string(s::AbstractString)::TickDirection
    s == "up"   && return tick_direction_up
    s == "down" && return tick_direction_down
    throw(ArgumentError("Unknown tick direction: $s"))
end

# numeric value helper (mirrors Python's up=1, down=-1)
    tick_direction_value(d::TickDirection) = d == tick_direction_up ? 1 : -1


# ── SecurityType ──────────────────────────────────────────────────────────────
@enum SecurityType begin
    security_type_equity
    security_type_option
end

function security_type_from_string(s::AbstractString)::SecurityType
    s == "equity" && return security_type_equity
    s == "option" && return security_type_option
    throw(ArgumentError("Unknown security type: $s"))
end

function security_type_from_ib_symbol(ib_symbol::AbstractString)::SecurityType
    ' ' in ib_symbol ? security_type_option : security_type_equity
end


# ── GreeksEuOption ────────────────────────────────────────────────────────────
# Simple string constants — no enum needed
const GREEK_DELTA = "delta"
const GREEK_VEGA  = "vega"
const GREEK_THETA = "theta"
const GREEK_GAMMA = "gamma"


# ── Resolution ────────────────────────────────────────────────────────────────
@enum Resolution begin
    resolution_daily
    resolution_hour
    resolution_minute
    resolution_second
    resolution_tick
end


# ── CsvHeader ─────────────────────────────────────────────────────────────────
# Immutable named constants — NamedTuple works well here
const CSV_HEADER_TRADE    = ["time", "open", "high", "low", "close", "volume"]
const CSV_HEADER_QUOTE    = ["time", "bid_open", "bid_high", "bid_low", "bid_close",
    "bid_size", "ask_open", "ask_high", "ask_low", "ask_close", "ask_size"]
const CSV_HEADER_IV_QUOTE = ["time", "mid_price_underlying", "bid_price", "bid_iv",
    "ask_price", "ask_iv", "delta_bid", "delta_ask"]
const CSV_HEADER_IV_TRADE = ["time", "mid_price_underlying", "price", "trade_iv", "delta"]


# ── OptionPricingModel ────────────────────────────────────────────────────────
@enum OptionPricingModel begin
    pricing_cox_ross_rubinstein
    pricing_analytic_european
    pricing_fd_bs_vanilla_div_schedule
    pricing_fd_bs_vanilla
    pricing_barone_adesi_whaley
end


# ── SkewMeasure ───────────────────────────────────────────────────────────────
@enum SkewMeasure begin
    skew_third_moment
    skew_delta25_delta50
    skew_delta25_delta25
    skew_m90_m100
    skew_m90_m110
    skew_m100_m110
end


# ── Market ────────────────────────────────────────────────────────────────────
@enum Market begin
    market_usa
end


# ── Scenario ────────────────────────────────────────────────────────────────────
@enum Scenario begin
    scenario_worst
    scenario_best
    scenario_estimated
    scenario_mid
    scenario_fills
end

# ── SSVIMetric ────────────────────────────────────────────────────────────────────
@enum SSVIMetric begin
    ssvi_theta
    ssvi_rho
    ssvi_psi
end

# ── GreekParameters ───────────────────────────────────────────────────────────
"""
    GreekParameters

Parameters for calculating Greeks.

Fields:
- `price_underlying::Float64` — spot price
- `volatility::Float64` — implied volatility
- `calculation_date::Date` — calculation date
"""
struct GreekParameters
    price_underlying::Float64
    volatility::Float64
    calculation_date::Date
end

# ── Greeks ────────────────────────────────────────────────────────────────────
"""
    Greeks

Calculated Greeks for an option.

Fields:
- `delta::Float64` — delta (∂price/∂spot)
- `gamma::Float64` — gamma (∂delta/∂spot)
- `vega::Float64` — vega (∂price/∂volatility)
- `theta::Float64` — theta (∂price/∂time)
"""
struct Greeks
    delta::Float64
    gamma::Float64
    vega::Float64
    theta::Float64
end

# ── OptionSerialized ──────────────────────────────────────────────────────────
"""
    OptionSerialized

Serializable representation of an Option (for pickling).

Fields:
- `optionContract::Any` — the OptionContract (using Any to avoid circular deps)
- `calculationDate::Date` — calculation date
- `price_underlying::Float64` — spot price at calculation
- `volatility::Float64` — volatility at calculation
"""
struct OptionSerialized
    optionContract::Any
    calculationDate::Date
    price_underlying::Float64
    volatility::Float64
end

# ── OptionStyle (enum replacement for Style class) ───────────────────────────
# Note: OptionStyle enum already exists above, but here's the string mapping:
"""
    option_style_to_string(style::OptionStyle)::String

Convert OptionStyle enum to string.
"""
function option_style_to_string(style::OptionStyle)::String
    style == option_style_american  && return "American"
    style == option_style_european  && return "European"
    error("Unknown OptionStyle: $style")
end

"""
    OPTION_STYLE_EUROPEAN::String
    OPTION_STYLE_AMERICAN::String
    OPTION_STYLE_AMERICAN_DIV::String

String constants for option styles (mirrors Python Style class).
"""
const OPTION_STYLE_EUROPEAN = "European"
const OPTION_STYLE_AMERICAN = "American"
const OPTION_STYLE_AMERICAN_DIV = "AmericanDiv"