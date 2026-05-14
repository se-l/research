# types/options.jl

using Dates
using Memoization

const DT_FMT_YMD = "yyyymmdd"

# ── OptionContract data (merged into Option) ───────────────────────────────────

"""
    Option <: Security

Combined OptionContract + Option. All pricing via PricingEngine (CUDA FD).

Fields mirror the original Python split into one flat struct.
"""
mutable struct Option <: Security
    # ── Contract fields (from OptionContract) ─────────────────────────────────
    symbol::String
    underlying_symbol::String
    expiry::Date
    strike::Float64
    right::OptionRight
    issue_date::Union{Date, Nothing}
    option_style::String          # OPTION_STYLE_AMERICAN / EUROPEAN / AMERICAN_DIV
    multiplier::Int
    equity::Equity

    # ── Pricing state (from Option) ────────────────────────────────────────────
    calculation_date::Date
    price_underlying::Float64
    volatility::Float64
    style::String                 # OPTION_STYLE_* constant
    price::Float64

    # ── Constants ──────────────────────────────────────────────────────────────
    const accuracy::Float64
    const max_iterations::Int
    const min_vol::Float64
    const max_vol::Float64

    function Option(
        symbol::String,
        underlying_symbol::String,
        expiry::Date,
        strike::Float64,
        right::OptionRight;
        issue_date::Union{Date, Nothing} = nothing,
        option_style::String = OPTION_STYLE_AMERICAN,
        multiplier::Int = 100,
        calculation_date::Date = today(),
        price_underlying::Float64 = 0.0,
        volatility::Float64 = 0.0,
        style::String = OPTION_STYLE_AMERICAN,
    )
        sym = uppercase(symbol)
        und = uppercase(underlying_symbol)
        eq  = Equity(und)
        new(
            sym, und, expiry, strike, right,
            issue_date, option_style, multiplier, eq,
            calculation_date, price_underlying, volatility, style, 0.0,
            1.0e-4, 100, 0.0001, 4.0,
        )
    end
end

# ── Security interface ─────────────────────────────────────────────────────────

function underlying_symbol(o::Option)::String
    return o.equity.symbol
end

function csv_name(o::Option, tick_type, resolution, dt::Union{Date, Nothing} = nothing)::String
    if (resolution == resolution_minute || resolution == resolution_second || resolution == resolution_tick) && dt !== nothing
        return lowercase("$(Dates.format(dt, DT_FMT_YMD))_$(o.symbol)_$(string(resolution))_$(string(tick_type))_$(o.option_style)_$(string(o.right))_$(Int(o.strike * 10000))_$(Dates.format(o.expiry, DT_FMT_YMD)).csv")
    else
        return lowercase("$(underlying_symbol(o))_$(string(tick_type))_$(o.option_style)_$(string(o.right))_$(Int(o.strike * 10000))_$(Dates.format(o.expiry, DT_FMT_YMD)).csv")
    end
end

function zip_name(o::Option, tick_type, resolution, dt::Union{Date, Nothing} = nothing)::String
    dt === nothing && throw(ArgumentError("dt must be provided for zip_name"))
    return get_zip_name(underlying_symbol(o), tick_type, resolution, dt)
end

function get_zip_name(underlying_sym::String, tick_type, resolution, dt::Date)::String
    tt = string(tick_type)
    if resolution == resolution_daily || resolution == resolution_hour
        if tick_type == tick_type_iv_quote || tick_type == tick_type_iv_trade
            return lowercase("$(underlying_sym)_$(year(dt))_$(split(tt, "_")[2])_american_iv.zip")
        end
        return lowercase("$(underlying_sym)_$(year(dt))_$(tt)_american.zip")
    else
        if tick_type == tick_type_iv_quote || tick_type == tick_type_iv_trade
            return lowercase("$(Dates.format(dt, DT_FMT_YMD))_$(split(tt, "_")[2])_american_iv.zip")
        end
        return lowercase("$(Dates.format(dt, DT_FMT_YMD))_$(tt)_american.zip")
    end
end

function nlv(o::Option, market_data::Dict, q::Float64 = 1.0, scenario::String = "mid")::Float64
    snap = market_data[o]
    if scenario == "mid"
        price = (snap.bid + snap.ask) / 2
    elseif scenario == "best"
        price = q > 0 ? snap.bid : snap.ask
    elseif scenario == "worst"
        price = q > 0 ? snap.ask : snap.bid
    else
        throw(ArgumentError("Invalid scenario: $scenario"))
    end
    return price * o.multiplier * q
end

# ── Factory methods ────────────────────────────────────────────────────────────

"""
    option_from_filename(filename, calc_date) -> Option

Parse an Option from a filename.
Supports three formats:
- Minute: `20230627_akam_minute_quote_american_put_1800000_20240119.csv`
- Short:  `fdx_american_call_2200000_20241115`
- Daily:  `are_quote_american_put_1950000_20230721.csv`
"""
function option_from_filename(filename::String, calc_date::Date = today(); issue_date::Union{Date, Nothing} = nothing)::Option
    parts = split(splitext(filename)[1], "_")
    local sym, option_style_str, option_right_str, strike_str, expiry_str

    if length(parts) == 8 && all(isdigit, parts[1])  # Minute format
        _, sym, _, _, option_style_str, option_right_str, strike_str, expiry_str = parts
    elseif length(parts) == 5  # Short format
        sym, option_style_str, option_right_str, strike_str, expiry_str = parts
    else  # Daily format
        sym, _, option_style_str, option_right_str, strike_str, expiry_str = parts
    end

    expiry       = Date(expiry_str, DateFormat("yyyymmdd"))
    right        = option_right_from_string(option_right_str)
    option_style = option_style_from_string(option_style_str)
    strike       = parse(Float64, strike_str) / 10000.0

    return Option(sym, sym, expiry, strike, right;
        issue_date = issue_date,
        option_style = option_style_to_string(option_style),
        calculation_date = calc_date,
    )
end

"""
    option_from_ib_symbol(ib_symbol, calc_date) -> Option

Parse from IB symbol format, e.g. `"PANW  250117C00540000"`.
"""
function option_from_ib_symbol(ib_symbol::String, calc_date::Date = today())::Option
    parts       = split(strip(ib_symbol))
    und_sym     = string(parts[1])
    tail        = string(parts[end])
    expiry      = Date(tail[1:6], DateFormat("yymmdd")) + Year(2000)
    right       = option_right_from_string(string(tail[7]))
    strike      = parse(Float64, tail[8:end]) / 1000.0

    return Option(ib_symbol, und_sym, expiry, strike, right;
        calculation_date = calc_date,
    )
end

"""
    option_from_contract_nm(contract_nm, calc_date) -> Option

Parse from contract name, e.g. `"fdx_american_call_2200000_20241115"`.
"""
function option_from_contract_nm(contract_nm::String, calc_date::Date = today(); issue_date::Union{Date, Nothing} = nothing)::Option
    sym, _, right_str, strike_str, expiry_str = split(contract_nm, "_")
    expiry = Date(expiry_str, DateFormat("yyyymmdd"))
    right  = option_right_from_string(right_str)
    strike = parse(Float64, strike_str) / 1000.0
    return Option(sym, sym, expiry, strike, right;
        issue_date = issue_date,
        calculation_date = calc_date,
    )
end

# ── IB symbol representation ───────────────────────────────────────────────────

function ib_symbol(o::Option)::String
    sym = rpad(underlying_symbol(o), 6)
    right_char = o.right == option_right_call ? "C" : "P"
    strike_str = lpad(string(Int(o.strike * 1000)), 8, '0')
    return uppercase("$(sym)$(Dates.format(o.expiry, "yymmdd"))$(right_char)$(strike_str)")
end

# ── Pricing via PricingEngine ──────────────────────────────────────────────────

"""
    intrinsic_value(o::Option, price_underlying::Float64 = o.price_underlying)::Float64
"""
function intrinsic_value(o::Option, price_underlying::Float64 = o.price_underlying)::Float64
    if o.right == option_right_call
        return max(price_underlying - o.strike, 0.0)
    else
        return max(o.strike - price_underlying, 0.0)
    end
end

"""
    npv(o::Option, vol::Float64, price_underlying::Float64, calc_date::Date)::Float64

Price via CUDA FD pricer (PricingEngine). Falls back to intrinsic at expiry.
"""
function npv(o::Option, vol::Float64, price_underlying::Float64, calc_date::Date)::Float64
    calc_date == o.expiry && return intrinsic_value(o, price_underlying)

    # Assumes get_v_price_fd is in scope from PricingEngine
    result = get_v_price_fd(
        Float32[price_underlying],
        Float32[o.strike],
        Float32[_tenor(o.expiry, calc_date)],
        Float32[vol],
        UInt8[o.right == option_right_call ? 1 : 0],
        Float32[], Float32[],  # rates_curve, rates_times (fill from yield curve)
        Float32[], Float32[],  # div_amounts, div_times
    )
    return Float64(result[1])
end

"""
    iv(o::Option, price_option::Float64, price_underlying::Float64, calc_date::Date)::Float64

Back-solve IV via CUDA FD bisection (get_v_iv_fd).
"""
function iv(o::Option, price_option::Float64, price_underlying::Float64, calc_date::Date)::Float64
    (isnan(price_option) || isnan(price_underlying)) && throw(ArgumentError("Price or underlying is NaN"))

    result = get_v_iv_fd(
        Float32[price_option],
        Float32[price_underlying],
        Float32[o.strike],
        Float32[_tenor(o.expiry, calc_date)],
        UInt8[o.right == option_right_call ? 1 : 0],
        Float32[], Float32[],
        Float32[], Float32[],
    )
    return Float64(result[1])
end

"""
    delta(o::Option, vol::Float64, price_underlying::Float64, calc_date::Date)::Float64
"""
function delta(o::Option, vol::Float64, price_underlying::Float64, calc_date::Date)::Float64
    result = get_v_delta_fd(
        Float32[price_underlying],
        Float32[o.strike],
        Float32[_tenor(o.expiry, calc_date)],
        Float32[vol],
        UInt8[o.right == option_right_call ? 1 : 0],
        Float32[], Float32[],
        Float32[], Float32[],
    )
    return Float64(result[1])
end

"""
    vega(o::Option, vol::Float64, price_underlying::Float64, calc_date::Date)::Float64
"""
function vega(o::Option, vol::Float64, price_underlying::Float64, calc_date::Date)::Float64
    result = get_v_vega_fd(
        Float32[price_underlying],
        Float32[o.strike],
        Float32[_tenor(o.expiry, calc_date)],
        Float32[vol],
        UInt8[o.right == option_right_call ? 1 : 0],
        Float32[], Float32[],
        Float32[], Float32[],
    )
    return Float64(result[1])
end

"""
    greeks(o::Option, params::GreekParameters) -> Greeks
"""
function greeks(o::Option, params::GreekParameters)::Greeks
    try
        d = delta(o, params.volatility, params.price_underlying, params.calculation_date)
        v = vega(o,  params.volatility, params.price_underlying, params.calculation_date)
        return Greeks(d, 0.0, v, 0.0)
    catch e
        @warn "greeks() failed: $e"
        return Greeks(0.0, 0.0, 0.0, 0.0)
    end
end

function is_otm(o::Option, price_underlying::Float64)::Bool
    return intrinsic_value(o, price_underlying) == 0.0
end

function extrinsic_value(o::Option, vol::Float64, price_underlying::Float64, calc_date::Date)::Float64
    return npv(o, vol, price_underlying, calc_date) - intrinsic_value(o, price_underlying)
end

function pv(k::Float64, t::Float64, r::Float64)::Float64
    return k * exp(-r * t)
end

# ── String representation & hashing ───────────────────────────────────────────

    Base.repr(o::Option) = ib_symbol(o)
    Base.string(o::Option) = repr(o)
    Base.hash(o::Option) = hash(repr(o))
Base.:(==)(a::Option, b::Option) = repr(a) == repr(b)
Base.:<(a::Option, b::Option)  = repr(a) < repr(b)
Base.:<=(a::Option, b::Option) = repr(a) <= repr(b)
Base.:>=(a::Option, b::Option) = repr(a) >= repr(b)
Base.:>(a::Option, b::Option)  = repr(a) > repr(b)

# ── Serialization ──────────────────────────────────────────────────────────────

function getstate(o::Option)::Dict{String, Any}
    return Dict(
        "symbol"           => o.symbol,
        "underlying"       => o.underlying_symbol,
        "expiry"           => string(o.expiry),
        "strike"           => o.strike,
        "right"            => string(o.right),
        "option_style"     => o.option_style,
        "multiplier"       => o.multiplier,
        "calculation_date" => string(o.calculation_date),
        "price_underlying" => o.price_underlying,
        "volatility"       => o.volatility,
    )
end

function setstate!(o::Option, state::Dict{String, Any})
    o.symbol            = state["symbol"]
    o.underlying_symbol = state["underlying"]
    o.expiry            = Date(state["expiry"])
    o.strike            = state["strike"]
    o.right             = option_right_from_string(state["right"])
    o.option_style      = state["option_style"]
    o.multiplier        = state["multiplier"]
    o.calculation_date  = Date(state["calculation_date"])
    o.price_underlying  = state["price_underlying"]
    o.volatility        = state["volatility"]
    o.equity            = Equity(o.underlying_symbol)
    return o
end

# ── Internal helpers ───────────────────────────────────────────────────────────

"""
    _tenor(expiry::Date, calc_date::Date)::Float32

Compute time-to-expiry in years (Actual/365).
"""
function _tenor(expiry::Date, calc_date::Date)::Float32
    return Float32(max((expiry - calc_date).value, 0) / 365.0)
end
