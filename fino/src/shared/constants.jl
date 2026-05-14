using Dates
using JSON  # For standard JSON; if JSON5 needed, use JSON3.jl instead
using Memoization  # ] add Memoization; replaces functools.lru_cache

const FILE_ROOT = Paths.PATH_DATA  # Assuming Paths is defined elsewhere; adjust as needed

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