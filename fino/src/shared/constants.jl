using Dates
using JSON
using Memoization
using TimeZones

const FILE_ROOT = Paths.PATH_DATA

# ── Lazy-loaded globals (no I/O on include) ────────────────────────────────────
let
    global _dividend_yield_loaded = false
    global DividendYield = Dict{String, Any}()
    global function get_dividend_yield()::Dict{String, Any}
        if !_dividend_yield_loaded
            try
                open(Paths.PATH_DIVIDEND_YIELDS, "r") do fh
                    global DividendYield = JSON.parse(read(fh, String))
                end
            catch e
                error("No DividendYield file found: $e")
                global DividendYield = Dict{String, Any}()
            end
            global _dividend_yield_loaded = true
        end
        return DividendYield
    end
end

let
    global _earnings_announcements_loaded = false
    global earnings_announcements = Vector{Dict{String, Any}}()
    global function get_earnings_announcements()::Vector{Dict{String, Any}}
        if !_earnings_announcements_loaded
            try
                open(Paths.PATH_EARNINGS, "r") do fh
                    global earnings_announcements = JSON.parse(read(fh, String))
                end
            catch e
                error("No EarningsAnnouncement file found: $e")
                global earnings_announcements = Vector{Dict{String, Any}}()
            end
            global _earnings_announcements_loaded = true
        end
        return earnings_announcements
    end
end

const DISCOUNT_RATE_MARKET = get(DividendYield, "DiscountRateMarket", 0.0435)  # https://www.bloomberg.com/markets/rates-bonds/government-bonds/us

@memoize function EarningsPreSessionDates(sym::String)
    """Returned dates are the last open market trading days before an earnings announcement.
    So if the release is in the morning before market open, then this return T-1 business date"""
    ea = get_earnings_announcements()
    dates = [Dates.Date(x["Date"]) for x in ea if uppercase(x["Symbol"]) == uppercase(sym)]
    sort!(dates)
    return dates
end

"""
    next_release_date(sym, dt) -> Union{Date, Nothing}

Returns the next earnings release date on or after `dt`, or `nothing` if none found.
"""
function next_release_date(sym::String, dt::Date)::Union{Date, Nothing}
    release_dates = EarningsPreSessionDates(sym)
    return something(findfirst(d -> d >= dt, release_dates) |> i -> i === nothing ? nothing : release_dates[i], nothing)
end

function _et_business_date()::Date
    et_now = now(TimeZone("America/New_York"))
    if hour(et_now) < 9 || (hour(et_now) == 9 && minute(et_now) < 30)
        return Date(et_now) - Day(1)
    end
    return Date(et_now)
end

"""
    get_configs(v_ticker::Vector{String}; n_days_lookback=-1, n_days_lookahead=2, min_release_date=nothing, takes=nothing) -> Vector{RawDataConfig}

Sorted by release date. Easy to continue where left off.
"""
function get_configs(v_ticker::Vector{String}; n_days_lookback::Int=-1, n_days_lookahead::Int=2, min_release_date::Union{Date, Nothing}=nothing, takes::Union{AbstractRange, Vector{Int}, Nothing}=nothing)::Vector{RawDataConfig}
    ea_configs = []
    iterations = takes === nothing ? (-30:-1) : takes
    
    for take in iterations
        for ticker in v_ticker
            dates = EarningsPreSessionDates(ticker)
            # Python's try-except IndexError
            if take < 0
                idx = length(dates) + take + 1
            else
                idx = take + 1
            end
            
            if idx < 1 || idx > length(dates)
                continue
            end
            
            ea_date = dates[idx]
            
            if min_release_date !== nothing && min_release_date > ea_date
                continue
            end
            
            start_dt = add_trade_days(ea_date, n_days_lookback)
            stop_dt = add_trade_days(ea_date, n_days_lookahead)
            stop_dt = min(stop_dt, _et_business_date())
            push!(ea_configs, (ea_date, RawDataConfig(start_dt, stop_dt, ticker)))
        end
    end
    # Sort by ea_date and return the RawDataConfig objects
    sort!(ea_configs, by = x -> x[1])
    return [c[2] for c in ea_configs]
end

export EarningsPreSessionDates, next_release_date, get_configs

model_nm_earnings_iv_drop_regressor = get(ENV, "model_nm_earnings_iv_drop_regressor", "f_20260404-144024")
TZ_USEASTERN = "US/Eastern"
TZ_UTC = "UTC"
TZ_HK = "Asia/Hong_Kong"
dt_fmt_ymd = "%Y%m%d"
dt_fmt_iso = "%Y-%m-%d"
dt_fmt_ymdhms = "%Y%m%d-%H%M%S"
dt_fmt_eastern = "%Y%m%d %H:%M:%S US/Eastern"  # 20031126 15:59:00 US/Eastern
dt_fmt_ib_bar = "%Y%m%d-%H:%M:%S"
dt_fmt_pb = "%Y-%m-%dT%H:%M:%S"
USA = "usa"
SECOND = "second"
MINUTE = "minute"
HOUR = "hour"