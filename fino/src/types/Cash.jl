# types/Cash.jl

using Dates

"""
    Cash <: Security

Represents a cash position (currency).

Fields:
- `symbol::String` — currency code (stored uppercase), e.g. "USD"
- `multiplier::Int` — contract multiplier (default 1)
"""
    mutable struct Cash <: Security
    symbol::String
    multiplier::Int

    function Cash(symbol::String; multiplier::Int = 1)
        new(uppercase(symbol), multiplier)
    end
end

# ── Implement Security interface ───────────────────────────────────────────────

function underlying_symbol(cash::Cash)::String
    return cash.symbol
end

function csv_name(cash::Cash, tick_type, resolution, dt::Union{Date, Nothing} = nothing)::String
    throw(NotImplementedError("csv_name not implemented for Cash"))
end

function zip_name(cash::Cash, tick_type, resolution, dt::Union{Date, Nothing} = nothing)::String
    throw(NotImplementedError("zip_name not implemented for Cash"))
end

function nlv(cash::Cash, market_data::Dict, q::Float64 = 1.0, scenario::String = "mid")::Float64
    return cash.multiplier * q
end

# ── Greek approximations (cash returns constant values) ────────────────────────

"""
    delta(cash::Cash, args...; kwargs...)

Cash has delta = 1 (moves dollar-for-dollar).
"""
function delta(cash::Cash, args...; kwargs...)::Float64
    return 1.0
end

"""
    iv(cash::Cash, args...; kwargs...)

Cash has no implied volatility (return 0).
"""
function iv(cash::Cash, args...; kwargs...)::Float64
    return 0.0
end

"""
    npv(cash::Cash, args...; kwargs...)

Net present value for cash is 1.0 (normalized).
"""
function npv(cash::Cash, args...; kwargs...)::Float64
    return 1.0
end

# ── String representation & hashing ────────────────────────────────────────────

    Base.repr(cash::Cash) = uppercase(cash.symbol)

    Base.string(cash::Cash) = repr(cash)

    Base.hash(cash::Cash) = hash(repr(cash))

# ── Factory method ────────────────────────────────────────────────────────────

"""
    cash_from_filename(filename::String)::Cash

Create a Cash from a filename (e.g., "USD_quote.csv").
Extracts symbol from the filename.
"""
function cash_from_filename(filename::String)::Cash
    symbol = split(filename, "_")[1]
    return Cash(symbol)
end