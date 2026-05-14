using Dates

struct RawDataConfig
    start::Date
    stop::Date      # 'end' is a reserved word in Julia
    tickers::String
end