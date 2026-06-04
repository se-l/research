module YieldCurve

#=
Description: Zero curve construction from US Treasury bill rates and par yields.
             Mirrors yield_curve.py — bootstraps par yields to zero rates,
             merges with bill rates, and caches results.
Author: seb
Date: 4/12/2026
=#

export ZeroCurveData, get_zero_curve, get_last_zero_curve,
       store_calibrated_rates, load_calibrated_rates, has_calibrated_curve

using Dates
using CSV
using DataFrames
using ..Paths

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

struct ZeroCurveData
    times ::Vector{Float32}
    rates ::Vector{Float32}
end

ZeroCurveData() = ZeroCurveData(Float32[], Float32[])

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

function _ir_root(market::String)::String
    joinpath(PATH_DATA_INTEREST_RATE, market)
end

function _bill_rates_path(market::String, year::Int)::String
    joinpath(_ir_root(market), "daily-treasury-bill-rates", "$year.csv")
end

function _par_yield_path(market::String, year::Int)::String
    joinpath(_ir_root(market), "daily-treasury-par-yield-curve-rates", "$year.csv")
end

function _calibrated_path(ticker::String, calculation_date::Date, market::String)::String
    joinpath(_ir_root(market), "daily",
             "$(uppercase(ticker))-$(Dates.format(calculation_date, "yyyymmdd")).csv")
end

# ─────────────────────────────────────────────────────────────────────────────
# Module-level state  (mirrors Python singleton fields)
# ─────────────────────────────────────────────────────────────────────────────

# Date → Dict{Float32, Float32}  (tenor → rate)
const _bill_rates  = Dict{Date, Dict{Float32, Float32}}()
const _par_yields  = Dict{Date, Dict{Float32, Float32}}()
const _zero_curves = Dict{Date, Dict{Float32, Float32}}()
const _initialized = Set{Tuple{Date, String}}()   # (date, market)

# LRU-style caches for the two public entry points
const _zero_curve_cache      = Dict{Tuple{Date, String, String}, ZeroCurveData}()
const _last_zero_curve_cache = Dict{Tuple{Date, String, String, Int}, ZeroCurveData}()

# ─────────────────────────────────────────────────────────────────────────────
# CSV loaders
# ─────────────────────────────────────────────────────────────────────────────

function _parse_date_mdy(s::AbstractString)::Union{Date, Nothing}
    try
        return Date(s, "m/d/yyyy")
    catch
        return nothing
    end
end

function _load_bill_rates!(market::String, dt::Date)
    path = _bill_rates_path(market, year(dt))
    isfile(path) || return

    # CE columns are at 1-based indices 3,5,7,9,11,13,15 (0-based 2,4,6,8,10,12,14)
    ce_cols   = [3, 5, 7, 9, 11, 13, 15]
    tenors_f32 = Float32[4/52, 6/52, 8/52, 13/52, 17/52, 26/52, 52/52]

    open(path, "r") do f
        readline(f)   # skip header
        for line in eachline(f)
            parts = split(line, ',')
            isempty(parts[1]) && continue
            row_date = _parse_date_mdy(strip(parts[1]))
            row_date === nothing && continue

            d = get!(() -> Dict{Float32, Float32}(), _bill_rates, row_date)
            for (i, col) in enumerate(ce_cols)
                col > length(parts) && continue
                raw = strip(parts[col])
                isempty(raw) && continue
                d[tenors_f32[i]] = Float32(parse(Float64, raw) / 100.0)
            end
        end
    end
end

function _load_par_yields!(market::String, dt::Date)
    path = _par_yield_path(market, year(dt))
    isfile(path) || return

    tenors_f32 = Float32[1/12, 1.5/12, 2/12, 3/12, 4/12, 6/12,
                          1, 2, 3, 5, 7, 10, 20, 30]

    open(path, "r") do f
        readline(f)   # skip header
        for line in eachline(f)
            parts = split(line, ',')
            isempty(parts[1]) && continue
            row_date = _parse_date_mdy(strip(parts[1]))
            row_date === nothing && continue

            par_pairs = Tuple{Float32, Float32}[]
            for (i, t) in enumerate(tenors_f32)
                col = i + 1
                col > length(parts) && continue
                raw = strip(parts[col])
                isempty(raw) && continue
                push!(par_pairs, (t, Float32(parse(Float64, raw) / 100.0)))
            end

            isempty(par_pairs) && continue
            bootstrapped = _bootstrap(par_pairs)

            d = get!(() -> Dict{Float32, Float32}(), _par_yields, row_date)
            merge!(d, bootstrapped)
        end
    end
end

function _initialize!(market::String, dt::Date)
    key = (dt, market)
    key in _initialized && return
    push!(_initialized, key)
    _load_bill_rates!(market, dt)
    _load_par_yields!(market, dt)
end

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap par → zero  (semi-annual Treasury convention)
# ─────────────────────────────────────────────────────────────────────────────

function _interpolate_zero(zero_rates::Dict{Float32, Float32}, t::Float32)::Float32
    times = sort(collect(keys(zero_rates)))
    isempty(times) && return 0f0
    t <= times[1]   && return zero_rates[times[1]]
    t >= times[end] && return zero_rates[times[end]]
    for i in 1:length(times)-1
        if times[i] <= t <= times[i+1]
            t1, t2 = times[i], times[i+1]
            r1, r2 = zero_rates[t1], zero_rates[t2]
            return r1 + (r2 - r1) * (t - t1) / (t2 - t1)
        end
    end
    return zero_rates[times[end]]
end

function _bootstrap(par_pairs::Vector{Tuple{Float32, Float32}})::Dict{Float32, Float32}
    zero_rates = Dict{Float32, Float32}()
    sort!(par_pairs, by=first)

    for (t, c) in par_pairs
        if t <= 1f0
            zero_rates[t] = c
        else
            coupon       = c / 2f0
            pv_coupons   = 0f0
            num_payments = round(Int, t * 2)

            for i in 1:num_payments-1
                payment_time = Float32(i) * 0.5f0
                z_i = _interpolate_zero(zero_rates, payment_time)
                pv_coupons += coupon / (1f0 + z_i / 2f0) ^ (2f0 * payment_time)
            end

            remaining = 1f0 - pv_coupons
            if remaining <= 0f0
                zero_rates[t] = c
                continue
            end

            z_n = 2f0 * (((1f0 + coupon) / remaining) ^ (1f0 / (2f0 * t)) - 1f0)
            zero_rates[t] = z_n
        end
    end

    return zero_rates
end

# ─────────────────────────────────────────────────────────────────────────────
# Calibrated curve persistence
# ─────────────────────────────────────────────────────────────────────────────

function has_calibrated_curve(calculation_date::Date, ticker::String, market::String="usa")::Bool
    isfile(_calibrated_path(ticker, calculation_date, market))
end

function store_calibrated_rates(ticker::String, calculation_date::Date,
                                curve::ZeroCurveData, market::String="usa")
    path = _calibrated_path(ticker, calculation_date, market)
    mkpath(dirname(path))
    df = DataFrame(Tenor=Float64.(curve.times), Rate=Float64.(curve.rates))
    CSV.write(path, df)
end

function load_calibrated_rates(ticker::String, calculation_date::Date,
                               market::String="usa")::ZeroCurveData
    path = _calibrated_path(ticker, calculation_date, market)
    isfile(path) || error("load_calibrated_rates(): $path does not exist.")
    df = CSV.read(path, DataFrame)
    ZeroCurveData(Float32.(df.Tenor), Float32.(df.Rate))
end

# ─────────────────────────────────────────────────────────────────────────────
# Core curve construction
# ─────────────────────────────────────────────────────────────────────────────

function _get_latest_available_curve(from_date::Date, market::String,
                                     max_days::Int=180)::ZeroCurveData
    for offset in 1:max_days
        candidate = from_date - Day(offset)
        _initialize!(market, candidate)
        bills = get(_bill_rates, candidate, Dict{Float32, Float32}())
        pars  = get(_par_yields,  candidate, Dict{Float32, Float32}())
        if !isempty(bills) || !isempty(pars)
            @info "get_zero_curve(): rates unavailable for $from_date, using $candidate instead"
            combined = Dict{Float32, Float32}(bills)
            bill_max = isempty(bills) ? 0f0 : maximum(keys(bills))
            for (t, r) in pars
                t > bill_max && (combined[t] = r)
            end
            sorted_tenors = sort(collect(keys(combined)))
            return ZeroCurveData(sorted_tenors, [combined[t] for t in sorted_tenors])
        end
    end
    return ZeroCurveData()
end

"""
    get_zero_curve(calculation_date, ticker="", market="usa") -> ZeroCurveData

Returns the zero curve for `calculation_date`.  If a calibrated curve for
`ticker` exists on disk it is loaded directly; otherwise the curve is built
from bill rates + bootstrapped par yields.
"""
function get_zero_curve(calculation_date::Date,
                        ticker::String="",
                        market::String="usa")::ZeroCurveData
    cache_key = (calculation_date, ticker, market)
    haskey(_zero_curve_cache, cache_key) && return _zero_curve_cache[cache_key]

    # Prefer calibrated curve when ticker is provided
    if !isempty(ticker) && has_calibrated_curve(calculation_date, ticker, market)
        result = load_calibrated_rates(ticker, calculation_date, market)
        _zero_curve_cache[cache_key] = result
        return result
    end

    _initialize!(market, calculation_date)

    if !haskey(_zero_curves, calculation_date)
        bills = get(_bill_rates, calculation_date, Dict{Float32, Float32}())
        pars  = get(_par_yields,  calculation_date, Dict{Float32, Float32}())

        if isempty(bills) && isempty(pars)
            # Rates not yet published for this date — fall back to the most
            # recent date for which we do have data (up to 14 calendar days back).
            fallback_result = _get_latest_available_curve(calculation_date, market)
            _zero_curve_cache[cache_key] = fallback_result
            return fallback_result
        end

        combined = Dict{Float32, Float32}(bills)

        bill_max = isempty(bills) ? 0f0 : maximum(keys(bills))
        for (t, r) in pars
            t > bill_max && (combined[t] = r)
        end

        sorted_tenors = sort(collect(keys(combined)))
        _zero_curves[calculation_date] = Dict(t => combined[t] for t in sorted_tenors)
    end

    curve = _zero_curves[calculation_date]
    sorted_tenors = sort(collect(keys(curve)))
    result = ZeroCurveData(sorted_tenors, [curve[t] for t in sorted_tenors])

    _zero_curve_cache[cache_key] = result
    return result
end

"""
    get_last_zero_curve(calculation_date, ticker="", delta_days=-14, market="usa") -> ZeroCurveData

Searches backwards from `calculation_date + delta_days` to `calculation_date`
for the most recent calibrated curve for `ticker`. Falls back to the generic
zero curve if none is found.
"""
function get_last_zero_curve(calculation_date::Date,
                             ticker::String="",
                             delta_days::Int=-14,
                             market::String="usa")::ZeroCurveData
    cache_key = (calculation_date, ticker, market, delta_days)
    haskey(_last_zero_curve_cache, cache_key) && return _last_zero_curve_cache[cache_key]

    # Walk backwards over the window [calculation_date + delta_days .. calculation_date]
    start_date = calculation_date + Day(delta_days)   # delta_days is negative → earlier date
    for offset in 0:-sign(delta_days):delta_days
        candidate = calculation_date + Day(offset)
        candidate < start_date && break
        if !isempty(ticker) && has_calibrated_curve(candidate, ticker, market)
            @info "get_last_zero_curve(): using calibrated curve" calculation_date candidate ticker
            result = get_zero_curve(candidate, ticker, market)
            _last_zero_curve_cache[cache_key] = result
            return result
        end
    end

    # Fallback: generic treasury zero curve for calculation_date
    result = get_zero_curve(calculation_date, "", market)
    _last_zero_curve_cache[cache_key] = result
    return result
end

function get_last_zero_curve(calculation_date::Date, ticker, delta_days::Int=-14, market::String="usa")::ZeroCurveData
    get_last_zero_curve(calculation_date, uppercase(ticker.symbol), delta_days, market)
end

end # module YieldCurve