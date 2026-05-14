# types/SymDate.jl

using Dates

"""
    SymDate

A pair of symbol and date for grouping by earnings release dates.

Fields:
- `symbol::String` — ticker symbol (stored uppercase)
- `date::Date` — the date
"""
struct SymDate
    symbol::String
    date::Date

    function SymDate(symbol::String, date::Date)
        new(uppercase(symbol), date)
    end
end

# ── Comparison and hashing ────────────────────────────────────────────────────

Base.:(==)(sd1::SymDate, sd2::SymDate) = (sd1.symbol == uppercase(sd2.symbol)) && (sd1.date == sd2.date)

    Base.hash(sd::SymDate) = hash((sd.symbol, sd.date))

# ── String representation ──────────────────────────────────────────────────────

    Base.repr(sd::SymDate) = "$(sd.symbol) - $(Dates.format(sd.date, "yyyy-mm-dd"))"

    Base.string(sd::SymDate) = repr(sd)