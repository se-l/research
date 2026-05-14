# types/Portfolio.jl

using DataFrames
using Dates
using .Fino: Holding

"""
    Portfolio

A collection of security holdings (implements dict-like interface).

Fields:
- `holdings::Dict{Security, Float64}` — security => quantity mapping
"""
mutable struct Portfolio
    holdings::Dict{Security, Float64}

    function Portfolio(holdings::Union{Dict{Security, Float64}, Nothing} = nothing)
        new(holdings === nothing ? Dict{Security, Float64}() : holdings)
    end
end

# ── Factory methods ────────────────────────────────────────────────────────────

"""
    Portfolio(holdings::Set{Holding})

Create a Portfolio from a set of Holdings.
"""
function Portfolio(holdings::Vector{Holding})
    pf_dict = Dict{Security, Float64}()
    for h in holdings
        pf_dict[h.symbol] = h.quantity
    end
    return Portfolio(pf_dict)
end

# ── Properties ─────────────────────────────────────────────────────────────────

"""
    underlying(pf::Portfolio)::Union{String, Nothing}

Return the underlying symbol if portfolio is non-empty.
"""
function underlying(pf::Portfolio)::Union{String, Nothing}
    if length(pf.holdings) > 0
        first_security = first(keys(pf.holdings))
        return uppercase(underlying_symbol(first_security))
    end
    return nothing
end

# ── Portfolio operations ───────────────────────────────────────────────────────

"""
    add_holding!(pf::Portfolio, security::Security, quantity::Float64)

Add quantity to a security in the portfolio.
"""
function add_holding!(pf::Portfolio, security::Security, quantity::Float64)::Portfolio
    if !haskey(pf.holdings, security)
        pf.holdings[security] = 0.0
    end
    pf.holdings[security] += quantity
    return pf
end

"""
    remove_security!(pf::Portfolio, security::Security)

Remove a security from the portfolio.
"""
function remove_security!(pf::Portfolio, security::Security)
    delete!(pf.holdings, security)
    return pf
end

"""
    delta_total(pf::Portfolio, calc_date::Date, s::Float64, df::DataFrame; iv_col::String = "mid_iv")::Float64

Calculate total delta of portfolio across all positions.
"""
function delta_total(pf::Portfolio, calc_date::Date, s::Float64, df::DataFrame; iv_col::String = "mid_iv")::Float64
    _delta_total = 0.0

    for (sec, q) in pf.holdings
        if typeof(sec) <: Equity
            _delta_total += q
        elseif typeof(sec) <: Option
            try
                # val_from_df(df, sec.expiry, sec.strike, sec.right, iv_col)
                # Assuming val_from_df is a helper function
                iv_val = val_from_df(df, sec.expiry, sec.optionContract.strike, sec.right, iv_col)

                if isnan(iv_val) || iv_val == 0.0
                    price = val_from_df(df, sec.expiry, sec.optionContract.strike, sec.right, "mid_price")
                    spot = val_from_df(df, sec.expiry, sec.optionContract.strike, sec.right, "spot")
                    iv_val = iv(sec, price, spot, calc_date)
                end
            catch e
                # Missing price data
                @warn "No $iv_col for $sec on $calc_date. Setting IV to 0.2. To be improved."
                iv_val = 0.2
            end
            _delta_total += delta(sec, iv_val, s, calc_date) * q * sec.multiplier
        elseif typeof(sec) <: Cash
            continue
        else
            throw(ArgumentError("Invalid security type: $(typeof(sec))"))
        end
    end

    return _delta_total
end

"""
    get_holdings(pf::Portfolio)::Vector{Holding}

Return all holdings as a vector of Holding objects.
"""
function get_holdings(pf::Portfolio)::Vector{Holding}
    return [Holding(sec, q) for (sec, q) in pf.holdings]
end

"""
    nlv(pf::Portfolio, market_data::Dict{Security, SecurityDataSnap}, scenario::String = "mid")::Float64

Calculate net liquidation value of portfolio.
"""
function nlv(pf::Portfolio, market_data::Dict{Security, SecurityDataSnap}, scenario::String = "mid")::Float64
    return sum(nlv(security, market_data, quantity, scenario) for (security, quantity) in pf.holdings)
end

"""
    print_backtest_holdings(pf::Portfolio)

Print holdings in CSV format (Symbol, Quantity, FillPrice).
"""
function print_backtest_holdings(pf::Portfolio)
    println("Symbol,Quantity,FillPrice")
    for (sec, q) in pf.holdings
        println("$(string(sec)),$q,0")
    end
end

# ── Dict-like interface ────────────────────────────────────────────────────────

    Base.keys(pf::Portfolio) = keys(pf.holdings)

function securities(pf::Portfolio)
    return keys(pf.holdings)
end

    Base.iterate(pf::Portfolio) = iterate(get_holdings(pf))
    Base.iterate(pf::Portfolio, state) = iterate(get_holdings(pf), state)

    Base.getindex(pf::Portfolio, key::Security) = get(pf.holdings, key, 0.0)

    Base.get(pf::Portfolio, key::Security, default) = get(pf.holdings, key, default)

    Base.length(pf::Portfolio) = length(pf.holdings)

    Base.repr(pf::Portfolio) = join(["$(string(o)): $q" for (o, q) in pf.holdings], ", ")

    Base.string(pf::Portfolio) = repr(pf)

"""
    copy(pf::Portfolio)

Create a shallow copy of the portfolio.
"""
    Base.copy(pf::Portfolio) = Portfolio(copy(pf.holdings))

    Base.isempty(pf::Portfolio) = isempty(pf.holdings)

# ── Serialization ──────────────────────────────────────────────────────────────

"""
    getstate(pf::Portfolio)

Return a serializable dict representation.
"""
function getstate(pf::Portfolio)::Dict{String, Any}
    return Dict("holdings" => pf.holdings)
end

"""
    setstate!(pf::Portfolio, state::Dict)

Restore portfolio from serialized state.
"""
function setstate!(pf::Portfolio, state::Dict{String, Any})
    pf.holdings = state["holdings"]
    return pf
end
