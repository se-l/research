# types/Security.jl

using Dates

# ── Abstract base type ─────────────────────────────────────────────────────────
"""
    Security

Abstract base type for securities (equities, options, etc.).
Subtypes must implement: underlying_symbol, csv_name, zip_name, nlv
"""
abstract type Security end

"""
    underlying_symbol(sec::Security)

Return the underlying symbol of the security.
Must be implemented by subtypes.
"""
function underlying_symbol(sec::Security)
    error("underlying_symbol not implemented for $(typeof(sec))")
end

"""
    csv_name(sec::Security, tick_type, resolution::Resolution, dt::Date = nothing)

Return CSV filename for the security at given tick type, resolution, and date.
Must be implemented by subtypes.
"""
function csv_name(sec::Security, tick_type, resolution, dt::Date = nothing)
    error("csv_name not implemented for $(typeof(sec))")
end

"""
    zip_name(sec::Security, tick_type, resolution::Resolution, dt::Date = nothing)

Return ZIP filename for the security at given tick type, resolution, and date.
Must be implemented by subtypes.
"""
function zip_name(sec::Security, tick_type, resolution, dt::Date = nothing)
    error("zip_name not implemented for $(typeof(sec))")
end

"""
    nlv(sec::Security, market_data::Dict, q::Float64 = 1.0, scenario::String = "mid")

Calculate net liquidation value given market data, quantity, and scenario.
Must be implemented by subtypes.
"""
function nlv(sec::Security, market_data::Dict, q::Float64 = 1.0, scenario::String = "mid")
    error("nlv not implemented for $(typeof(sec))")
end

# ── Concrete implementations of comparison & hashing ────────────────────────────
Base.repr(sec::Security) = sec.symbol

Base.hash(sec::Security) = hash(repr(sec))

Base.:<(sec1::Security, sec2::Security) = repr(sec1) < repr(sec2)
Base.:<=(sec1::Security, sec2::Security) = repr(sec1) <= repr(sec2)
Base.:>=(sec1::Security, sec2::Security) = repr(sec1) >= repr(sec2)
Base.:>(sec1::Security, sec2::Security) = repr(sec1) > repr(sec2)

Base.string(sec::Security) = repr(sec)

Base.:(==)(sec1::Security, sec2::Security) = repr(sec1) == repr(sec2)


# ── SecurityDataSnap ───────────────────────────────────────────────────────────
"""
    SecurityDataSnap

Snapshot of a security's market data at a point in time.

Fields:
- `security::Security` — the security
- `ts::DateTime` — timestamp
- `spot::Float64` — spot price
- `bid::Union{Float64, Nothing}` — bid price (optional)
- `ask::Union{Float64, Nothing}` — ask price (optional)
"""
struct SecurityDataSnap
    security::Security
    ts::DateTime
    spot::Float64
    bid::Union{Float64, Nothing}
    ask::Union{Float64, Nothing}

    function SecurityDataSnap(
        security::Security,
        ts::DateTime,
        spot::Float64;
        bid::Union{Float64, Nothing} = nothing,
        ask::Union{Float64, Nothing} = nothing,
    )
        new(security, ts, spot, bid, ask)
    end
end

Base.hash(snap::SecurityDataSnap) = hash((snap.security.symbol, snap.ts))
