using Dates
using DataFrames  # For pd.Timestamp compatibility
using JSON

# ============================================================================
# Module-level caches (equivalent to @lru_cache(maxsize=1))
# ============================================================================

const _market_hours_cache = Ref{Union{Nothing, Dict}}(nothing)
const _holidays_cache = Ref{Union{Nothing, Vector{Date}}}(nothing)
const _early_closes_cache = Ref{Union{Nothing, Vector{DateTime}}}(nothing)

# ============================================================================
# Market hours functions
# ============================================================================

"""Load market hours database from JSON file (cached)."""
function load_market_hours()::Dict
    if _market_hours_cache[] === nothing
        open(joinpath(Paths.PATH_MARKET_HOURS, "market-hours-database.json"), "r") do f
            _market_hours_cache[] = JSON.parse(read(f, String))
        end
    end
    return _market_hours_cache[]::Dict
end

"""Get market hours entry for a given market (cached)."""
function get_market_hours(market::String="Equity-usa-[*]")::Dict
    return load_market_hours()["entries"][market]
end

"""Get list of holiday dates (cached)."""
function get_market_hours_holidays(market::String="Equity-usa-[*]")::Vector{Date}
    if _holidays_cache[] === nothing
        entries = get_market_hours(market)
        _holidays_cache[] = [Date(d, "mm/dd/yyyy") for d in entries["holidays"]]
    end
    return _holidays_cache[]::Vector{Date}
end

"""Get holidays as a Date array (cached). In Julia, Vector{Date} replaces numpy datetime64[D]."""
function get_market_hours_holidays_arr(market::String="Equity-usa-[*]")::Vector{Date}
    return get_market_hours_holidays(market)
end

"""Get list of early close datetimes (cached)."""
function get_market_hours_early_closes(market::String="Equity-usa-[*]")::Vector{DateTime}
    if _early_closes_cache[] === nothing
        entries = get_market_hours(market)
        _early_closes_cache[] = [DateTime(k * " " * v, "mm/dd/yyyy HH:MM:SS") for (k, v) in entries["earlyCloses"]]
    end
    return _early_closes_cache[]::Vector{DateTime}
end

# ============================================================================
# Trade days
# ============================================================================

"""Return all trading days (non-weekend, non-holiday) between two dates, inclusive."""
function trade_days_between_dates(dt1::Date, dt2::Date; market::String="Equity-usa-[*]")::Vector{Date}
    holidays = Set(get_market_hours_holidays(market))
    d_start = min(dt1, dt2)
    d_end = max(dt1, dt2)
    return [dt for dt in d_start:Day(1):d_end if dayofweek(dt) <= 5 && dt ∉ holidays]
end

"""
    add_trade_days(dt::Date, n::Int; market::String="Equity-usa-[*]") -> Date

Add n trading days to date dt.
"""
function add_trade_days(dt::Date, n::Int; market::String="Equity-usa-[*]")::Date
    holidays = Set(get_market_hours_holidays(market))
    current_dt = dt
    remaining = abs(n)
    step = n >= 0 ? Day(1) : Day(-1)
    
    while remaining > 0
        current_dt += step
        if dayofweek(current_dt) <= 5 && current_dt ∉ holidays
            remaining -= 1
        end
    end
    return current_dt
end

# ============================================================================
# Memoization cache for get_tenor
# ============================================================================

# Simple memoization dict (Julia equivalent of @lru_cache)
const _TENOR_CACHE = Dict{Tuple{Date, DateTime}, Float64}()

# ============================================================================
# Date to datetime conversion
# ============================================================================

"""
    date_to_datetime(dt::Date, offset::Dates.HourMinute) -> DateTime

Convert a Date to DateTime with optional time offset.
Default offset is midnight (00:00).
"""
function date_to_datetime(dt::Date, offset::Dates.CompoundPeriod=Hour(0) + Minute(0))::DateTime
    return DateTime(dt) + offset
end

"""
    date_to_datetime(dt::DateTime, offset::Dates.HourMinute=HourMinute(0, 0)) -> DateTime

If already DateTime, return as-is with offset.
"""
function date_to_datetime(dt::DateTime, offset::Dates.CompoundPeriod=Hour(0) + Minute(0))::DateTime
    return dt + offset
end

# ============================================================================
# Date to start of day
# ============================================================================

"""
    date_to_sod(dt) -> DateTime

Convert date to start of day (9:30 AM).
Presumes market open time of 9:30.
"""
function date_to_sod(dt)::DateTime
    if dt isa DateTime
        return dt
    end
    return date_to_datetime(dt, Hour(9) + Minute(30))
end

# ============================================================================
# Date to end of day
# ============================================================================

function date_to_eod(dt)::DateTime
    if dt isa DateTime
        return dt
    end
    return date_to_datetime(dt, Hour(17) + Minute(30))
end

# ============================================================================
# Get tenor (years to expiration from calculation date)
# ============================================================================

"""
    get_tenor(dt::Date, calculation_dt) -> Float64

Calculate time to maturity in years.
Presumes able to send exercise notice at about 17:30 on exercise day.
If calculation_dt is a date only, presume it's 9:30 market open time.

# Arguments
- `dt`: expiry date
- `calculation_dt`: calculation date (can be Date, DateTime, or QuantLib Date)

# Returns
- Time to maturity in years (fraction of year)
"""
function get_tenor(dt::Union{Date, DateTime}, calculation_dt)::Float64
    # Check cache
    cache_key = (dt, calculation_dt)
    if haskey(_TENOR_CACHE, cache_key)
        return _TENOR_CACHE[cache_key]
    end

    # Handle calculation_dt type
    start_dt = if calculation_dt isa DateTime
        calculation_dt
    elseif calculation_dt isa Date
        DateTime(calculation_dt)
    else
        error("Unsupported type - calculation_dt: $(typeof(calculation_dt))")
    end
    start = date_to_sod(start_dt)

    # Handle dt (expiry) type
    end_dt = if dt isa Date
        DateTime(dt)
    elseif dt isa DateTime
        dt
    else
        error("Unsupported type tenor date: $(typeof(dt))")
    end

    # Add 17:30 (5:30 PM) to the end date
    end_with_time = end_dt + Hour(17) + Minute(30)

    # Calculate fractional years
    seconds = (end_with_time - start).value / 1000  # convert milliseconds to seconds
    days_fraction = seconds / (60^2 * 24)
    years = days_fraction / 365.0

    # Cache result
    _TENOR_CACHE[cache_key] = years

    return years
end

"""
    get_v_tenor(dt::Vector{DateTime}, calculation_dt::Union{DateTime, Vector{DateTime}}) -> Vector{Float64}

Compute time-to-expiry in years (Actual/365) for each element of `dt`
relative to `calculation_dt`.
"""
function get_v_tenor(dt::Vector{DateTime}, calculation_dt::DateTime)::Vector{Float64}
    isempty(dt) && return Float64[]
    return [Dates.value(Millisecond(d - calculation_dt)) / 1000.0 / (365 * 24 * 3600) for d in dt]
end

function get_v_tenor(dt::Vector{DateTime}, calculation_dt::Vector{DateTime})::Vector{Float64}
    isempty(dt) && return Float64[]
    return [Dates.value(Millisecond(dt[i] - calculation_dt[i])) / 1000.0 / (365 * 24 * 3600) for i in eachindex(dt)]
end

# ============================================================================
# Clear cache function
# ============================================================================

"""
    clear_tenor_cache()

Clear the tenor memoization cache.
"""
function clear_tenor_cache()
    empty!(_TENOR_CACHE)
end

# ============================================================================
# Export public API
# ============================================================================

export get_tenor,
       date_to_sod,
       date_to_datetime,
       add_trade_days,
       clear_tenor_cache