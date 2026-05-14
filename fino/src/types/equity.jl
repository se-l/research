# types/Equity.jl

using Dates

"""
    Equity <: Security

Represents an equity (stock) security.

Fields:
- `symbol::String` — ticker symbol
- `multiplier::Int` — contract multiplier (default 1)
"""
    mutable struct Equity <: Security
    symbol::String
    multiplier::Int

    function Equity(symbol::String; multiplier::Int = 1)
        new(symbol, multiplier)
    end
end

# ── Implement Security interface ───────────────────────────────────────────────

function underlying_symbol(eq::Equity)::String
    return eq.symbol
end

function csv_name(eq::Equity, tick_type, resolution, dt::Union{Date, Nothing} = nothing)::String
    # For minute, second, tick resolutions: include date
    if (resolution == resolution_minute || resolution == resolution_second || resolution == resolution_tick) && dt !== nothing
        return "$(Dates.format(dt, "yyyymmdd"))_$(lowercase(eq.symbol))_$(string(resolution))_$(string(tick_type)).csv"
    else
        return "$(lowercase(eq.symbol)).csv"
    end
end

function zip_name(eq::Equity, tick_type, resolution, dt::Union{Date, Nothing} = nothing)::String
    # For daily and hour: no date in filename
    if resolution == resolution_daily || resolution == resolution_hour
        return "$(lowercase(eq.symbol)).zip"
    else
        if dt === nothing
            throw(ArgumentError("dt must be provided for zip_name with resolution $(resolution)"))
        end
        return "$(Dates.format(dt, "yyyymmdd"))_$(string(tick_type)).zip"
    end
end

function nlv(eq::Equity, market_data::Dict, q::Float64 = 1.0, scenario::String = "mid")::Float64
    snap = market_data[eq]
    return snap.spot * eq.multiplier * q
end

# ── Greek approximations (equities return constant values) ──────────────────────

"""
    delta(eq::Equity, args...)

Equities have delta = 1 (move dollar-for-dollar with price).
"""
function delta(eq::Equity, args...; kwargs...)::Float64
    return 1.0
end

"""
    iv(eq::Equity, args...)

Equities have no implied volatility (return 0).
"""
function iv(eq::Equity, args...; kwargs...)::Float64
    return 0.0
end

"""
    npv(eq::Equity, args...)

Net present value for equity is spot price (simplified).
"""
function npv(eq::Equity, args...; kwargs...)::Float64
    return 1.0
end

# ── String representation & hashing ────────────────────────────────────────────

    Base.repr(eq::Equity) = uppercase(eq.symbol)

    Base.string(eq::Equity) = repr(eq)

    Base.hash(eq::Equity) = hash(repr(eq))

# ── Serialization helpers (for pickle-like functionality) ────────────────────────

"""
    getstate(eq::Equity)

Return a serializable dict representation.
"""
function getstate(eq::Equity)::Dict{String, Any}
    return Dict("symbol" => repr(eq), "multiplier" => eq.multiplier)
end

"""
    setstate!(eq::Equity, state::Dict)

Restore state from a dict (inverse of getstate).
"""
function setstate!(eq::Equity, state::Dict{String, Any})
    eq.symbol = state["symbol"]
    eq.multiplier = state["multiplier"]
    return eq
end

# ── Factory method ─────────────────────────────────────────────────────────────

"""
    Equity(filename::String)

Create an Equity from a filename like "hpe_quote.csv".
Extracts symbol from the filename.
"""
function equity_from_filename(filename::String)::Equity
    # Simple extraction: assumes symbol is the part before any extension/suffix
    symbol = split(filename, "_")[1]
    return Equity(symbol)
end
