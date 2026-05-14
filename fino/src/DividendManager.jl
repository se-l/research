module DividendManager

using Dates
using DataFrames
using HTTP
using JSON
using CSV
using ..Paths

export Dividend, get_dividends, fetch_dividends, load_from_disk, get_dividend_amount_times

const NO_DIVIDENDS = Set(["PANW", "ONON", "XPO", "PATH", "CDNS", "BIDU", "CBRE",
    "CRWD", "DLTR", "DOCU", "KMX", "LI", "MDB", "MRNA",
    "NEOG", "NFLX", "ON", "PDD", "PLTR", "SNOW", "WDAY", "ZK"])

mutable struct Dividend
    ticker::String
    ex_date::Date
    amount::Float32
    predicted::Bool

    Dividend(ticker::String, ex_date::Date, amount::Float32; predicted::Bool=false) =
        new(ticker, ex_date, amount, predicted)
end

function dividends_dir()::String
    return joinpath(PATH_DATA, "equity", "usa", "dividends")
end

function dividends_csv_path(ticker::String)::String
    return joinpath(dividends_dir(), "$(uppercase(ticker)).csv")
end

function to_yyyymmdd(d::Date)::String
    return Dates.format(d, "yyyymmdd")
end

function from_yyyymmdd(s::String)::Date
    return Date(s, "yyyymmdd")
end

function write_dividends_csv(ticker::String, dividends::Vector{Dividend})::Nothing
    mkpath(dividends_dir())

    df = DataFrame(
        ExDate = [to_yyyymmdd(d.ex_date) for d in sort(dividends, by=x->x.ex_date)],
        Amount = [d.amount for d in sort(dividends, by=x->x.ex_date)],
        Ccy = ["USD" for _ in dividends]
    )

    csv_path = dividends_csv_path(ticker)
    CSV.write(csv_path, df)
end

function read_dividends_csv(ticker::String)::Union{Vector{Dividend}, Nothing}
    path = dividends_csv_path(ticker)

    if !isfile(path)
        return nothing
    end

    try
        df = CSV.read(path, DataFrame; types=Dict(:ExDate => String))

        if nrow(df) == 0
            return Dividend[]
        end

        out = Dividend[]
        for row in eachrow(df)
            try
                exd = from_yyyymmdd(String(row.ExDate))
                amt = Float32(row.Amount)
                push!(out, Dividend(uppercase(ticker), exd, amt))
            catch
                continue
            end
        end
        return out
    catch
        return nothing
    end
end

function load_from_disk(ticker::String, start::Union{String, Date}, end_date::Union{String, Date})::Union{Vector{Dividend}, Nothing}
    # Normalize dates
    if start isa String
        start = Date(start, "yyyy-mm-dd")
    end
    if end_date isa String
        end_date = Date(end_date, "yyyy-mm-dd")
    end

    data = read_dividends_csv(ticker)
    if data === nothing
        return nothing
    end

    return [d for d in data if start <= d.ex_date <= end_date]
end

function infer_frequency_months(ex_dates::Vector{Date})::Int
    if length(ex_dates) < 2
        return 3
    end

    ex_dates_sorted = sort(ex_dates)
    deltas = [(ex_dates_sorted[i] - ex_dates_sorted[i-1]).value for i in 2:length(ex_dates_sorted)]

    if isempty(deltas)
        return 3
    end

    median_days = sort(deltas)[div(length(deltas), 2) + 1]

    candidates = Dict(1 => 30, 2 => 60, 3 => 91, 4 => 121, 6 => 182, 12 => 365)

    months = argmin([abs(candidates[m] - median_days) for m in keys(candidates)])
    return collect(keys(candidates))[months]
end

function add_months(d::Date, months::Int)::Date
    y = year(d) + div(month(d) - 1 + months, 12)
    m = mod(month(d) - 1 + months, 12) + 1

    if m == 12
        next_y, next_m = y + 1, 1
    else
        next_y, next_m = y, m + 1
    end

    first_next_month = Date(next_y, next_m, 1)
    last_day_prev = first_next_month - Day(1)
    _day = min(day(d), day(last_day_prev))

    return Date(y, m, _day)
end

function forecast_dividends(ticker::String, history::Vector{Dividend}, until::Date)::Vector{Dividend}
    if isempty(history)
        return Dividend[]
    end

    td = today()
    hist = [d for d in history if d.ex_date <= td]

    if isempty(hist)
        return Dividend[]
    end

    hist_sorted = sort(hist, by=x->x.ex_date)
    last = hist_sorted[end]

    recent_history = hist_sorted[max(1, length(hist_sorted)-11):end]
    months = infer_frequency_months([d.ex_date for d in recent_history])

    forecast = Dividend[]
    next_date = add_months(last.ex_date, months)

    horizon = min(until, td + Day(round(Int, 365.25 * 2)))

    while next_date <= horizon
        years = (next_date - last.ex_date).value / 365.25
        amt = last.amount * ((1.0 + 0.05) ^ years)
        push!(forecast, Dividend(uppercase(ticker), next_date, Float32(amt); predicted=true))
        next_date = add_months(next_date, months)
    end

    return forecast
end

function get_dividends(ticker::String, start::Union{String, Date, DateTime}, end_date::Union{String, Date, Nothing}=nothing)::Vector{Dividend}
    if uppercase(ticker) in NO_DIVIDENDS
        return Dividend[]
    end

    if end_date === nothing
        if start isa String
            start_date = Date(start, "yyyy-mm-dd")
        else
            start_date = start isa DateTime ? Date(start) : start
        end
        end_date = start_date + Year(3)
    end

    # Normalize dates
    if start isa String
        start_date = Date(start, "yyyy-mm-dd")
    elseif start isa DateTime
        start_date = Date(start)
    else
        start_date = start
    end

    if end_date isa String
        end_date = Date(end_date, "yyyy-mm-dd")
    elseif end_date isa DateTime
        end_date = Date(end_date)
    end

    # Try load from disk
    loaded = load_from_disk(ticker, start_date, end_date)
    if loaded === nothing
        # Fetch from Yahoo (simplified - would need yfinance equivalent)
        history = Dividend[]
        write_dividends_csv(ticker, history)
    else
        history = read_dividends_csv(ticker)
        if history === nothing
            history = Dividend[]
        end
    end

    td = today()
    in_range_hist = [d for d in history if start_date <= d.ex_date <= min(end_date, td)]
    forecasts = forecast_dividends(ticker, history, end_date)
    in_range_forecasts = [d for d in forecasts if start_date <= d.ex_date <= end_date]

    combined = sort(vcat(in_range_hist, in_range_forecasts), by=x->x.ex_date)
    return combined
end

function get_dividends(ticker, start::Union{String, Date, DateTime}, end_date::Union{String, Date, Nothing}=nothing)::Vector{Dividend}
    get_dividends(uppercase(ticker.symbol), start, end_date)
end

function fetch_dividends(
ticker::String,
start_date::Union{String, Date},
end_date::Union{String, Date, Nothing}=nothing;
source::String="yahoo"
)::Vector{Dividend}
    """
    Fetch dividend data for a given ticker symbol and date range.

    Parameters:
    -----------
    ticker : String
        The ticker symbol of the stock (e.g., 'AAPL', 'MSFT')

    start_date : String or Date
        The start date for dividend data retrieval
        String format should be 'YYYY-MM-DD'

    end_date : String, Date, or Nothing, optional
        The end date for dividend data retrieval
        String format should be 'YYYY-MM-DD'
        If nothing, defaults to current date

    source : String, optional
        The data source to use. Currently supports:
        - 'yahoo': Yahoo Finance (default)

    Returns:
    --------
    Vector{Dividend}
        Vector containing dividend information

    Throws:
    -------
    ErrorException
        If an unsupported source is specified
    """

    # Standardize dates
    if start_date isa String
        start_date = Date(start_date, "yyyy-mm-dd")
    end

    if end_date isa String
        end_date = Date(end_date, "yyyy-mm-dd")
    elseif end_date === nothing
        end_date = today()
    end

    # Validate source
    if lowercase(source) != "yahoo"
        throw(ErrorException("Source '$source' not supported. Currently only 'yahoo' is supported."))
    end

    # Note: Actual Yahoo Finance integration would require a Julia package like YFinance.jl
    # This is a placeholder structure
    return get_dividends(ticker, start_date, end_date)
end

# ─────────────────────────────────────────────────────────────────────────────
# Cached dividend amount + time-to-ex-date lookup
# Key: (uppercase(ticker), calculation_date)
# ─────────────────────────────────────────────────────────────────────────────

const _DIV_CACHE = Dict{Tuple{String, Date}, Tuple{Vector{Float32}, Vector{Float32}}}()

function get_dividend_amount_times(ticker::String, calculation_date::Date)::Tuple{Vector{Float32}, Vector{Float32}}
    key = (uppercase(ticker), calculation_date)

    if haskey(_DIV_CACHE, key)
        return _DIV_CACHE[key]
    end

    dividends = get_dividends(uppercase(ticker), calculation_date, Date(2030, 1, 1))

    amounts = [d.amount for d in dividends]
    times   = [((d.ex_date - calculation_date).value + 1) / 365.0 for d in dividends]

    result = (amounts, times)
    _DIV_CACHE[key] = result
    return result
end

function get_dividend_amount_times(ticker, calculation_date::Date)::Tuple{Vector{Float32}, Vector{Float32}}
    get_dividend_amount_times(uppercase(ticker.symbol), calculation_date)
end

end # module DividendManager