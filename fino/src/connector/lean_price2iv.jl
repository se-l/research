# store_iv.jl
# Julia Script

#=

Description: You are given 3 files gouped by date, eg, 20250605_quote_american.zip, 20250605_trade_american.zip and 20250605_quote.zip.
All prices in basis points: 1/10_000
All times are ms since midnight
Containing the following schemas:

Equity Trade / 20250605_quote.zip.
msSinceMidnight|PriceBid|SizeBid|PriceAsk|SizeAsk|NotRelevant|NotRelevant|NotRelevant
14340016.3	0	0	0	0			0

Option Trade 20250605_trade_american.zip
msSinceMidnight|Price|Size|NotRelevant|NotRelevant|NotRelevant
34219926	147800	13			0


Option Quote 20250605_quote_american.zip
msSinceMidnight|PriceBid|SizeBid|PriceAsk|SizeAsk|NotRelevant|NotRelevant
34201251.65	240500	1	323500	1		0

We need to convert the option bid ask prices to bid ask implied volatilities using batch GPU calculators.

Target schema output is:
msSinceMidnight|UnderlyingPrice.Close|PriceBid?.Close ?? 0|IvBid?.Close ?? 0|PriceAsk?.Close|IvAsk?.Close";

Please devise a scripts that in sequence:
1. Read in a quote file above from
D:\trade\data\option\usa\second
and
D:\trade\data\equity\usa\second
1. Creates input vectors needed by get_v_iv_fd
2. Important to recognize that spots are mid price from equity. and these can never front run option quotes, need to fill forwards equtity spots
3. metrics such as strike, right, tenor are parsed from the .csv name of the zip archive.
20260109_dal_second_quote_american_call_200000_20260116.csv
so that's trade date, ticker, right  strike in basis points and last, tenor
4. Calculation datetime is the zip dates + given timestamp
5. Rates come from YieldCurve.get_last_zero_curve(calculation_date::Date,
                             ticker::String="",
                             delta_days::Int=-14,
                             market::String="usa")
5. Dividends come from get_dividend_amount_times
6. Each zip entry name is
    string entryName = $"{date:yyyyMMdd}_{underlying.Value}_{resolution}_{tickType.TickTypeToLower()}_american_{symbol.ID.OptionRight}_{Math.Round(symbol.ID.StrikePrice * 10000m)}_{symbol.ID.Date:yyyyMMdd}.csv".ToLowerInvariant();
6. THe zip file name stored next to where quote.zip was from should be names:
 $"{optionPath}_{date.Year}_{tickTypeString}_{symbol.ID.OptionStyle.OptionStyleToLower()}.zip";

Author: seb
Date: 4/13/2026
=#

# For a given date, reads equity + option quote zips, aligns spots (no look-ahead),
# fetches rates & dividends, then batch-computes bid/ask IVs via get_v_iv_fd.

using ZipFile, Dates, DataFrames, CSV

using ..PricingEngine
using ..DividendManager
using ..YieldCurve
using ..Paths

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

const OPTION_SECOND_DIR = raw"D:\trade\data\option\usa\second"
const EQUITY_SECOND_DIR = raw"D:\trade\data\equity\usa\second"
const OUTPUT_DIR      = raw"D:\trade\data\option\usa\second"

const BP = 10_000f0   # basis-point divisor

# Number of significant digits to round tenors to (reduces GPU call size)
const TENOR_SIG_DIGITS = 3

# ─────────────────────────────────────────────────────────────────────────────
# Tick type enum
# ─────────────────────────────────────────────────────────────────────────────

@enum TickType begin
    TickQuote
    TickTrade
end

tick_type_str(t::TickType) = t == TickQuote ? "quote" : "trade"
tick_type_output_str(t::TickType) = t == TickQuote ? "iv_quote" : "iv_trade"

# ─────────────────────────────────────────────────────────────────────────────
# Filename parsing
# ─────────────────────────────────────────────────────────────────────────────

struct OptionMeta
    trade_date ::Date
    ticker     ::String
    is_call    ::Bool       # true = call, false = put
    strike     ::Float32    # in real price units
    expiry     ::Date
end

"""
Parse entry name like:
  20260109_dal_second_quote_american_call_200000_20260116.csv
"""
function parse_entry_name(name::String)::OptionMeta
    base   = replace(name, r"\.csv$"i => "")
    parts  = split(base, '_')
    # parts: [date, ticker, resolution, ticktype, american, right, strike_bp, expiry]
    trade_date = Date(parts[1], dateformat"yyyymmdd")
    ticker     = uppercase(parts[2])
    is_call    = lowercase(parts[6]) == "call"
    strike     = parse(Float32, parts[7]) / BP
    expiry     = Date(parts[8], dateformat"yyyymmdd")
    OptionMeta(trade_date, ticker, is_call, strike, expiry)
end

# ─────────────────────────────────────────────────────────────────────────────
# Zip path helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
option quote zip:  {ticker_dir}/{date_str}_quote_american.zip
equity quote zip:  {ticker_dir}/{date_str}_quote.zip
"""
function option_zip(ticker::String, tick_type::TickType, date::Date)::String
    date_str = Dates.format(date, "yyyymmdd")
    dir      = joinpath(OPTION_SECOND_DIR, lowercase(ticker))
    joinpath(dir, "$(date_str)_$(tick_type_str(tick_type))_american.zip")
end

function equity_quote_zip(ticker::String, date::Date)::String
    date_str = Dates.format(date, "yyyymmdd")
    dir      = joinpath(EQUITY_SECOND_DIR, lowercase(ticker))
    joinpath(dir, "$(date_str)_quote.zip")
end

function output_zip_path(ticker::String, tick_type::TickType, date::Date)::String
    date_str = Dates.format(date, "yyyymmdd")
    dir      = joinpath(OUTPUT_DIR, lowercase(ticker))
    mkpath(dir)
    joinpath(dir, "$(date_str)_$(tick_type_output_str(tick_type))_american.zip")
end

# ─────────────────────────────────────────────────────────────────────────────
# Reading helpers  (pipe-delimited, first row = header)
# ─────────────────────────────────────────────────────────────────────────────

function read_csv_raw(io::IO, col_names::Vector{Symbol})::DataFrame
    df = CSV.read(io, DataFrame;
        delim         = ',',
        header        = false,
        missingstring = "",
        types         = Float64)
    rename!(df, col_names[1:ncol(df)])
    df
end

"""
Second-resolution equity quote columns:
ms | bidOpen | bidHigh | bidLow | bidClose | bidSize | askOpen | askHigh | askLow | askClose | askSize
Only bidClose and askClose are used to compute mid spot.
"""
function read_equity_quotes(zip_path::String)::DataFrame
    cols = [:ms, :bidOpen, :bidHigh, :bidLow, :bidClose, :bidSize,
                 :askOpen, :askHigh, :askLow, :askClose, :askSize]
    zr  = ZipFile.Reader(zip_path)
    raw = read(zr.files[1])
    close(zr)
    isempty(raw) && return DataFrame(ms=Float64[], mid=Float32[])
    df  = read_csv_raw(IOBuffer(raw), cols)
    isempty(df) && return DataFrame(ms=Float64[], mid=Float32[])
    select!(df, :ms, :bidClose, :askClose)
    df[!, :mid] = ((df.bidClose .+ df.askClose) ./ 2f0) ./ BP
    select!(df, :ms, :mid)
    df
end

"""
Second-resolution option quote columns:
ms | bidOpen | bidHigh | bidLow | bidClose | bidSize | askOpen | askHigh | askLow | askClose | askSize
Only bidClose and askClose are used.
"""
function read_option_quotes(io::IO)::DataFrame
    cols = [:ms, :bidOpen, :bidHigh, :bidLow, :bidClose, :bidSize,
                 :askOpen, :askHigh, :askLow, :askClose, :askSize]
    raw = read(io)
    isempty(raw) && return DataFrame(ms=Float64[], bidClose=Float32[], askClose=Float32[])
    df = read_csv_raw(IOBuffer(raw), cols)
    isempty(df) && return DataFrame(ms=Float64[], bidClose=Float32[], askClose=Float32[])
    select!(df, :ms, :bidClose, :askClose)
    df[!, :bidClose] = df.bidClose ./ BP
    df[!, :askClose] = df.askClose ./ BP
    df
end

"""
Second-resolution option trade columns:
ms | open | high | low | close | size
Only close is used.
"""
function read_option_trades(io::IO)::DataFrame
    cols = [:ms, :open, :high, :low, :close, :size]
    raw  = read(io)
    isempty(raw) && return DataFrame(ms=Float64[], close=Float32[])
    df   = read_csv_raw(IOBuffer(raw), cols)
    isempty(df) && return DataFrame(ms=Float64[], close=Float32[])
    select!(df, :ms, :close)
    df[!, :close] = df.close ./ BP
    df
end

# ─────────────────────────────────────────────────────────────────────────────
# Forward-fill equity spot (no look-ahead)
#
# For each option quote timestamp, use the LAST equity mid that is <= that ms.
# ─────────────────────────────────────────────────────────────────────────────

function align_spots(opt_ms::Vector{Float64}, eq::DataFrame)::Vector{Float32}
    eq_ms  = eq.ms
    eq_mid = eq.mid
    n      = length(opt_ms)
    spots  = Vector{Float32}(undef, n)

    eq_idx = 1
    for i in 1:n
        t = opt_ms[i]
        # advance equity pointer while next equity tick is still <= t
        while eq_idx < length(eq_ms) && eq_ms[eq_idx + 1] <= t
            eq_idx += 1
        end
        spots[i] = eq_ms[eq_idx] <= t ? Float32(eq_mid[eq_idx]) : NaN32
    end
    spots
end

# ─────────────────────────────────────────────────────────────────────────────
# Tenor helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Round to N significant digits to collapse near-duplicate tenor values."""
function round_sig(x::Float32, sig::Int)::Float32
    x == 0f0 && return 0f0
    d = ceil(Int, log10(abs(x)))
    factor = 10f0^(sig - d)
    round(x * factor) / factor
end

function tenors_years(date::Date, expiry::Date, ms::Vector{Float64})::Vector{Float32}
    expiry_dt = DateTime(expiry) + Minute(round(Int, 17.5 * 60))
    map(ms) do m
        calc_dt = DateTime(date) + Millisecond(round(Int64, m))
        raw     = Float32(max(0.0, (expiry_dt - calc_dt).value / (1000.0 * 86400 * 365)))
        round_sig(raw, TENOR_SIG_DIGITS)
    end
end

function ms_to_date(date::Date, ms::Float64)::DateTime
    DateTime(date) + Millisecond(round(Int64, ms))
end

#function tenors_years(date::Date, expiry::Date, ms::Vector{Float64})::Vector{Float32}
#    expiry_dt = DateTime(expiry) + Minute(round(Int, 17.5 * 60))
#    map(ms) do m
#        calc_dt = ms_to_date(date, m)
#        Float32(max(0.0, (expiry_dt - calc_dt).value / (1000.0 * 86400 * 365)))
#    end
#end
# ─────────────────────────────────────────────────────────────────────────────
# Collect all entry data from the zip (no GPU call yet)
# Returns a vector of NamedTuples, one per CSV entry
# ─────────────────────────────────────────────────────────────────────────────

struct EntryData
    name     ::String
    meta     ::OptionMeta
    ms_vec   ::Vector{Float64}
    spots    ::Vector{Float32}
    bid      ::Vector{Float32}  # bidClose for quotes; tradeClose for trades (ask stays empty)
    ask      ::Vector{Float32}  # askClose for quotes; empty for trades
    tenors   ::Vector{Float32}
    is_trade ::Bool             # true = single-price trade entry
end

function collect_entries(
    opt_zip  ::String,
    tick_type::TickType,
    date     ::Date,
    eq_df    ::DataFrame
)::Vector{EntryData}

    entries = EntryData[]
    zr = ZipFile.Reader(opt_zip)
    try
        for entry in zr.files
            endswith(entry.name, ".csv") || continue
            @info "  reading entry: $(entry.name)"

            meta = parse_entry_name(entry.name)
            raw  = read(entry)

            if tick_type == TickQuote
                opt_df = read_option_quotes(IOBuffer(raw))
                isempty(opt_df) && continue
                ms_vec = Float64.(opt_df.ms)
                spots  = align_spots(ms_vec, eq_df)
                tenors = tenors_years(date, meta.expiry, ms_vec)
                push!(entries, EntryData(
                    entry.name, meta, ms_vec, spots,
                    Float32.(opt_df.bidClose),
                    Float32.(opt_df.askClose),
                    tenors, false))
            else  # TickTrade
                opt_df = read_option_trades(IOBuffer(raw))
                isempty(opt_df) && continue
                ms_vec = Float64.(opt_df.ms)
                spots  = align_spots(ms_vec, eq_df)
                tenors = tenors_years(date, meta.expiry, ms_vec)
                push!(entries, EntryData(
                    entry.name, meta, ms_vec, spots,
                    Float32.(opt_df.close),
                    Float32[],          # no ask side for trades
                    tenors, true))
            end
        end
    finally
        close(zr)
    end
    entries
end

# ─────────────────────────────────────────────────────────────────────────────
# Build a single deduplicated GPU input batch across ALL entries × both sides,
# run ONE get_v_iv_fd call, scatter results back.
#
# A pricing key is: (price, spot, strike, tenor, right)
# All five must match for a result to be reused.
# ─────────────────────────────────────────────────────────────────────────────

"""
Encodes one row that needs IV pricing.
"""
struct PricingKey
    price  ::Float32
    spot   ::Float32
    strike ::Float32
    tenor  ::Float32
    right  ::UInt8      # 1 = call, 0 = put
end

function build_and_run_gpu(
    entries     ::Vector{EntryData},
    rates_times ::Vector{Float32},
    rates_curve ::Vector{Float32},
    div_amounts ::Vector{Float32},
    div_times   ::Vector{Float32}
)
    index_map = Dict{PricingKey, Int}()
    u_price  = Float32[]
    u_spot   = Float32[]
    u_strike = Float32[]
    u_tenor  = Float32[]
    u_right  = UInt8[]

    bid_indices = [zeros(Int, length(e.ms_vec)) for e in entries]
    ask_indices = [zeros(Int, length(e.ms_vec)) for e in entries]

    for (ei, e) in enumerate(entries)
        right = e.meta.is_call ? UInt8(1) : UInt8(0)
        n     = length(e.ms_vec)

        # sides to process: quotes have bid+ask; trades have only bid slot (close price)
        sides = e.is_trade ?
            ((e.bid, bid_indices[ei]),) :
            ((e.bid, bid_indices[ei]), (e.ask, ask_indices[ei]))

        for i in 1:n
            spot  = e.spots[i]
            tenor = e.tenors[i]
            valid_base = !isnan(spot) && tenor > 0f0

            for (side_prices, side_indices) in sides
                p = side_prices[i]
                if valid_base && p > 0f0
                    key = PricingKey(p, spot, e.meta.strike, tenor, right)
                    idx = get!(index_map, key) do
                        push!(u_price,  p)
                        push!(u_spot,   spot)
                        push!(u_strike, e.meta.strike)
                        push!(u_tenor,  tenor)
                        push!(u_right,  right)
                        length(u_price)
                    end
                    side_indices[i] = idx
                end
            end
        end
    end

    n_uniq = length(u_price)
    @info "GPU batch: $n_uniq unique pricing inputs (deduplicated across all entries)"

    iv_all = if n_uniq > 0
        get_v_iv_fd(
            u_price, u_spot, u_strike, u_tenor, u_right,
            rates_curve, rates_times,
            div_amounts, div_times)
    else
        Float32[]
    end

    return bid_indices, ask_indices, iv_all
end

# ─────────────────────────────────────────────────────────────────────────────
# Assemble per-entry result DataFrames from scattered IV results
# ─────────────────────────────────────────────────────────────────────────────

function assemble_result(
    e        ::EntryData,
    bid_idxs ::Vector{Int},
    ask_idxs ::Vector{Int},
    iv_all   ::Vector{Float32}
)::DataFrame

    n        = length(e.ms_vec)
    ms_out   = round.(Int64, e.ms_vec)
    spot_out = round.(Int64, e.spots .* BP)

    if e.is_trade
        # output: ms, underlyingPrice, tradedPrice, IV
        price_out = round.(Int64, e.bid .* BP)
        iv_out    = Vector{Union{Int64,Missing}}(missing, n)
        for i in 1:n
            if bid_idxs[i] > 0
                v = iv_all[bid_idxs[i]]
                isnan(v) || (iv_out[i] = round(Int64, v * BP))
            end
        end
        return DataFrame(
            ms              = ms_out,
            underlying_price = spot_out,
            traded_price    = price_out,
            iv              = iv_out)
    else
        # output: ms, spot, price_bid, iv_bid, price_ask, iv_ask
        price_bid_out = round.(Int64, e.bid .* BP)
        price_ask_out = round.(Int64, e.ask .* BP)
        iv_bid_out    = Vector{Union{Int64,Missing}}(missing, n)
        iv_ask_out    = Vector{Union{Int64,Missing}}(missing, n)
        for i in 1:n
            if bid_idxs[i] > 0
                v = iv_all[bid_idxs[i]]
                isnan(v) || (iv_bid_out[i] = round(Int64, v * BP))
            end
            if ask_idxs[i] > 0
                v = iv_all[ask_idxs[i]]
                isnan(v) || (iv_ask_out[i] = round(Int64, v * BP))
            end
        end
        return DataFrame(
            ms        = ms_out,
            spot      = spot_out,
            price_bid = price_bid_out,
            iv_bid    = iv_bid_out,
            price_ask = price_ask_out,
            iv_ask    = iv_ask_out)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Top-level: process one ticker × one date
# ─────────────────────────────────────────────────────────────────────────────

function process_ticker_date(ticker::String, tick_type::TickType, date::Date)
    opt_zip = option_zip(ticker, tick_type, date)
    eq_zip  = equity_quote_zip(ticker, date)   # always quote for spot

    isfile(opt_zip) || (println("Missing option zip: $opt_zip"); return)
    isfile(eq_zip)  || (println("Missing equity zip: $eq_zip");  return)

    @info "Processing $ticker $(tick_type_str(tick_type)) $date"

    zc          = get_last_zero_curve(date, ticker, -14, "usa")
    rates_times = Float32.(zc.times)
    rates_curve = Float32.(zc.rates)

    div_amounts, div_times = get_dividend_amount_times(ticker, date)
    div_amounts = Float32.(div_amounts)
    div_times   = Float32.(div_times)

    eq_df = read_equity_quotes(eq_zip)
    sort!(eq_df, :ms)

    entries = collect_entries(opt_zip, tick_type, date, eq_df)
    isempty(entries) && (@info "No entries found in $opt_zip"; return)

    bid_indices, ask_indices, iv_all =
        build_and_run_gpu(entries, rates_times, rates_curve, div_amounts, div_times)

    out_zip = output_zip_path(ticker, tick_type, date)
    zw = ZipFile.Writer(out_zip)
    try
        for (ei, e) in enumerate(entries)
            result   = assemble_result(e, bid_indices[ei], ask_indices[ei], iv_all)
            out_name = replace(e.name, r"_(quote|trade)"i => "")
            f = ZipFile.addfile(zw, out_name)
            isempty(result) ? write(f, "") :
                CSV.write(f, result; delim=',', writeheader=false, missingstring="")
        end
    finally
        close(zw)
    end

    @info "Written: $out_zip"
end

# ─────────────────────────────────────────────────────────────────────────────
# Batch: process all dates for a ticker found in its option directory
# ─────────────────────────────────────────────────────────────────────────────

"""
    process_ticker(ticker; after=Date(0))

Scans the option second directory for `ticker` and calls `process_ticker_date`
for every date found in both `*_quote_american.zip` and `*_trade_american.zip`
files. Only dates strictly after `after` are processed.
"""
function process_ticker(ticker::String; after::Date = Date(0))
    dir = joinpath(OPTION_SECOND_DIR, lowercase(ticker))
    isdir(dir) || (println("Missing ticker directory: $dir"); return)

    # Collect all dates that have at least one relevant zip
    date_types = Dict{Date, Set{TickType}}()

    for fname in readdir(dir)
        m = match(r"^(\d{8})_(quote|trade)_american\.zip$", fname)
        m === nothing && continue

        date = tryparse(Date, m[1], dateformat"yyyymmdd")
        date === nothing && continue
        date > after || continue

        tick_type = m[2] == "quote" ? TickQuote : TickTrade
        push!(get!(date_types, date, Set{TickType}()), tick_type)
    end

    isempty(date_types) && (@info "No files found after $after for $ticker"; return)

    for date in sort(collect(keys(date_types)))
        for tick_type in (TickQuote, TickTrade)   # deterministic order
            tick_type ∈ date_types[date] || continue
            try
                process_ticker_date(ticker, tick_type, date)
            catch e
                @error "Failed $ticker $(tick_type_str(tick_type)) $date" exception=e
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

function run(; ticker::String, tick_type::TickType, date::Date)
    process_ticker_date(ticker, tick_type, date)
end
process_ticker("FDX", after=Date(2023, 12, 14))
#run(ticker="FDX", tick_type=TickQuote, date=Date(2025, 12, 18))
#run(ticker="FDX", tick_type=TickTrade, date=Date(2025, 12, 18))