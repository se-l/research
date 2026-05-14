using Dates
using Random

using .DividendManager: Dividend

struct CalibrationItem
    mny_fwd_ln::Vector{Float64}
    strike::Vector{Float64}
    calculation_date::Date
    tenor_dt::Date
    v_is_call::Vector{UInt8}
    iv::Vector{Float64}
    price::Vector{Float64}
    spot::Vector{Float64}
    dividends::Vector{Dividend}
    weights::Union{Vector{Float64}, Nothing}
    vega::Union{Vector{Float64}, Nothing}
    ts::Union{Vector{DateTime}, Nothing}
    bid_price::Vector{Float64}
    ask_price::Vector{Float64}
    dividend_yield::Union{Float64, Nothing}  # not needed anymore for American options
end

function CalibrationItem(;
    mny_fwd_ln::Vector,
    strike::Vector,
    calculation_date::Date,
    tenor_dt::Date,
    v_is_call::Vector{UInt8},
    iv::Vector,
    price::Vector,
    spot::Vector,
    dividends::Vector{Dividend}=Dividend[],
    weights::Union{Vector, Nothing}=nothing,
    vega::Union{Vector, Nothing}=nothing,
    ts::Union{Vector{DateTime}, Nothing}=nothing,
    bid_price::Union{Vector, Nothing}=nothing,
    ask_price::Union{Vector, Nothing}=nothing,
    dividend_yield::Union{Float64, Nothing}=nothing
)
    n = length(price)
    bid_price_fixed = bid_price === nothing ? fill(NaN, n) : bid_price
    ask_price_fixed = ask_price === nothing ? fill(NaN, n) : ask_price
    
    return CalibrationItem(
        mny_fwd_ln, strike, calculation_date, tenor_dt, v_is_call, iv, price, spot,
        dividends, weights, vega, ts, bid_price_fixed, ask_price_fixed, dividend_yield
    )
end

function Base.getproperty(item::CalibrationItem, sym::Symbol)
    if sym === :tenor
        return get_tenor(item.tenor_dt, item.calculation_date)
    end
    return getfield(item, sym)
end

function Base.:(+)(a::CalibrationItem, b::CalibrationItem)
    if a.tenor_dt != b.tenor_dt
        throw(ArgumentError("Cannot add CalibrationItems with different tenor_dt"))
    end
    if a.calculation_date != b.calculation_date
        throw(ArgumentError("Cannot add CalibrationItems with different calculation_date"))
    end

    return CalibrationItem(
        mny_fwd_ln=vcat(a.mny_fwd_ln, b.mny_fwd_ln),
        strike=vcat(a.strike, b.strike),
        calculation_date=a.calculation_date,
        tenor_dt=a.tenor_dt,
        v_is_call=vcat(a.v_is_call, b.v_is_call),
        iv=vcat(a.iv, b.iv),
        price=vcat(a.price, b.price),
        spot=vcat(a.spot, b.spot),
        dividends=a.dividends,
        weights=a.weights === nothing || b.weights === nothing ? nothing : vcat(a.weights, b.weights),
        vega=a.vega === nothing || b.vega === nothing ? nothing : vcat(a.vega, b.vega),
        ts=a.ts === nothing || b.ts === nothing ? nothing : vcat(a.ts, b.ts),
        bid_price=vcat(a.bid_price, b.bid_price),
        ask_price=vcat(a.ask_price, b.ask_price),
        dividend_yield=a.dividend_yield
    )
end

function downsample(item::CalibrationItem, n::Int)
    n_actual = min(n, length(item.price))
    ix_sample = randperm(length(item.price))[1:n_actual]
    
    return CalibrationItem(
        mny_fwd_ln=item.mny_fwd_ln[ix_sample],
        strike=item.strike[ix_sample],
        calculation_date=item.calculation_date,
        tenor_dt=item.tenor_dt,
        v_is_call=item.v_is_call[ix_sample],
        iv=item.iv[ix_sample],
        price=item.price[ix_sample],
        spot=item.spot[ix_sample],
        dividends=item.dividends,
        weights=item.weights === nothing ? nothing : item.weights[ix_sample],
        vega=item.vega === nothing ? nothing : item.vega[ix_sample],
        ts=item.ts === nothing ? nothing : item.ts[ix_sample],
        bid_price=item.bid_price[ix_sample],
        ask_price=item.ask_price[ix_sample],
        dividend_yield=item.dividend_yield
    )
end

function df2calibration_items(df_in::DataFrame, calc_date::Date, iv_col_nm, price_col_nm, spot_col_nm, dividends::Vector{Dividend};
    weight_col_nm="vega_mid_price_iv", vega_col_nm=nothing, max_samples_by_tenor=nothing, seed=1234)
    
    # Filter out NaN values
    df = df_in[.!ismissing.(df_in[!, price_col_nm]) .& .!ismissing.(df_in[!, iv_col_nm]), :]
    
    if weight_col_nm !== nothing
        df = df[(.!ismissing.(df[!, weight_col_nm])) .& (df[!, weight_col_nm] .> 0), :]
    end
    
    Random.seed!(seed)
    calibration_items = CalibrationItem[]
    
    # Group by expiry
    for expiry in unique(df[!, :expiry])
        s_df = df[df[!, :expiry] .== expiry, :]
        
        if max_samples_by_tenor !== nothing
            n_samples = min(max_samples_by_tenor, nrow(s_df))
            ix_sample = randperm(nrow(s_df))[1:n_samples]
            df_sample = s_df[ix_sample, :]
        else
            df_sample = s_df
        end
        
        push!(calibration_items, CalibrationItem(
            mny_fwd_ln=Float64.(df_sample[!, :moneyness_fwd_ln]),
            strike=Float64.(df_sample[!, :strike]),
            calculation_date=calc_date,
            tenor_dt=expiry,
            v_is_call=Bool.(df_sample[!, :v_is_call]),
            iv=Float64.(df_sample[!, iv_col_nm]),
            price=Float64.(df_sample[!, price_col_nm]),
            spot=Float64.(df_sample[!, spot_col_nm]),
            dividend_yield=0.0,
            dividends=dividends,
            weights=weight_col_nm !== nothing ? Float64.(df_sample[!, weight_col_nm]) : nothing,
            vega=vega_col_nm !== nothing ? Float64.(df_sample[!, vega_col_nm]) : nothing,
            ts=DateTime.(df_sample[!, :ts]),
            bid_price="bid_close" in names(df_in) ? Float64.(df_sample[!, :bid_close]) : nothing,
            ask_price="ask_close" in names(df_in) ? Float64.(df_sample[!, :ask_close]) : nothing,
        ))
    end
    
    return calibration_items
end

function union_calibration_items(ci::Vector{CalibrationItem}...)
    dct_ci = Dict{Date, CalibrationItem}()
    
    for c in Iterators.flatten(ci)
        key = c.tenor_dt
        if !haskey(dct_ci, key)
            dct_ci[key] = c
        else
            dct_ci[key] = dct_ci[key] + c
        end
    end
    
    return sort(collect(values(dct_ci)), by=x -> x.tenor_dt)
end