module OrderBook

using DataFrames
using PyCall

import ..ffill, ..isna, ..bfill

mutable struct Book
    """
    - Multi dimensional frame with dimensions: time, side, level    ( potentially another to get size of each bid/ask within level)!
        created from a stream of ticks
    - Best bid ask frame.
    """
    df_quotes:: DataFrame
    level_distance
end

# @lru_cache()
# function level_distance(df_quotes):: Int
#     res = (df_quotes["price"].shift(1) - df_quotes["price"]).abs()
#     return res[res != 0].min()
# end


function create_order_book!(order_book)
    df = order_book.df_quotes
    # assert sorted(df.side.unique()) == [-1, 1], "Side is not fully determined. Infer from BBAB and price"
    # Count may only be availbe with Bitfinex at the moment. Enrich if necessary. Emptied levels, count 0, mean to be ignore for best bid determination, but
    # need information to accurately encode when levels where emptied and filled
    
    vec_count::Vector{Bool} = df.count .== 0
    # vec_count = df.count .== 0
    arg2::Vector = ifelse.(vec_count .== true, 0, df.side)
    # arg2 = ifelse.(vec_count .== true, 0, df.side)
    a, b = apply_best_bid_ask(df.price, arg2, df.size)
    df = hcat(df, DataFrame(best_bid=a))
    df = hcat(df, DataFrame(best_ask=b))

    df.best_bid = ffill(df.best_bid)
    df.best_ask = ffill(df.best_ask)

    df[df[!, :"best_bid"] .== 0, :"best_bid"] .= missing
    df[df[!, :"best_ask"] .== 99999, :"best_ask"] .= missing

    df.best_ask = bfill(df.best_ask)
    df.best_bid = bfill(df.best_bid)
    
    filter!(r -> !isna(r.price), df)
    filter!(r -> !isna(r.size), df)

    df = impute_missing_count!(df)

    # assert df.isna().sum().sum() == 0, "NANs at this step. why?"
    # assert (df.best_ask < df.best_bid).sum() == 0
    println("Null price rows: $(sum(isna(df.price)))")

    # Add Level
    ix_drop_no_level_land = findall((df.best_bid .< df.price) .& (df.price .< df.best_ask))
    # ix_drop_no_level_land = df.index[np.where((df.best_bid < df.price) & (df.price < df.best_ask))]
    if length(ix_drop_no_level_land) > 0
        println("Dropping $(length(ix_drop_no_level_land)) order book levels that are in between best ask and best bid.")
        df = df[Not(ix_drop_no_level_land), :]
    end

    # Assign side to emptied levels
    df[df.count .== 0, :"size"] .= 0
    # assert df["size"][df.count == 0].sum() == 0, "Size should be 0 whenever emtpy order book level / count is 0."

    # Might be better to drop these rather than reassigning. Could be due to data coming in wrong sequence.
    v_wrong_side_ask = (df.side .== -1) .& (df.price .< df.best_ask)
    v_wrong_side_bid = (df.side .== 1) .& (df.price .> df.best_bid)
    if sum(v_wrong_side_bid) + sum(v_wrong_side_ask) > 0
        println("Reassigning Inconsistent / Wrong side level BID: $(sum(v_wrong_side_bid)) - ASK: $(sum(v_wrong_side_ask)). Investigate if inconsistency count is high ")
        df[v_wrong_side_bid, :"side"] *= -1
        df[v_wrong_side_ask, :"side"] *= -1
    end

    # Filled Levels
    v_ask = df.side .== -1
    v_bid = df.side .== 1
    df = hcat(df, DataFrame(level=Array{Union{Missing, Int32}}(missing, size(df)[1])))
    df[v_ask, :"level"] = convert.(Int, round.((df[v_ask, :"price"] - df[v_ask, :"best_ask"]) / order_book.level_distance .+ 1))
    df[v_bid, :"level"] = convert.(Int, round.((df[v_bid, :"price"] - df[v_bid, :"best_bid"]) / order_book.level_distance .- 1))

    # assert loc(df)[ix_ask, "level"].min() >= 1
    # assert loc(df)[ix_bid, "level"].max() <= 0
    # assert df["level"].isna().sum() == 0

    df[!, :"level"] = abs.(df[!, "level"])
    order_book.df_quotes = df # = setindex!(df, ["timestamp", "side", "level"])
    return df
end

function impute_missing_count!(df::DataFrame)
    if sum(isna(df.count)) > 0
        ix_zero = isna(df.count) .& (abs.(df.size) .== 1)
        df[ix_zero, :"count"] .= 0

        ix_non_zero = isna(df.count) .& (abs.(df.size) .!= 1)
        df[ix_non_zero, :"count"] .= 1
        println("Imputed $(sum(ix_zero) + sum(ix_non_zero)) count values.")
        return df
    else
        return df
    end
end

# @property
# @lru_cache()  
# return min(int((order_book.df_quotes["price"] * order_book.level_from_price_pct / 100).max() / order_book.level_distance), 300)

function apply_best_bid_ask(price:: Vector, side:: Vector, size:: Vector)
    active_bids = Set()
    active_asks = Set()
    n = length(price)
    bbid = price[findfirst(side .== 1)]
    bask = price[findfirst(side .== -1)]
    best_bids = Array{Union{Missing, Float64}}(missing, n)
    best_asks = Array{Union{Missing, Float64}}(missing, n)
    for i = 1:n
        p = price[i]
        c = side[i]
        if c == 0
            if size[i] > 0
                if p in active_bids
                    delete!(active_bids, p)
                end
                if p > bbid
                    bbid = isempty(active_bids) ? 0 : maximum(active_bids)
                end

            elseif size[i] < 0
                if p in active_asks
                    delete!(active_asks, p)
                end
                if p < bask
                    bask = isempty(active_asks) ? 99999 : minimum(active_asks)
                end
            end
        else
            if size[i] > 0
                push!(active_bids, p)
                if p >= bbid
                    bbid = p
                end
            elseif size[i] < 0
                push!(active_asks, p)
                if p <= bask
                    bask = p
                end
            end
        end

        if bbid >= bask  # something went wrong. make it consistent
            if size[i] > 0  # is bid
                for n in filter((x)->x<=bbid, active_asks)
                    delete!(active_asks, n)
                end
                bask = isempty(active_asks) ? 99999 : minimum(active_asks)
            end
            if size[i] < 0  # is ask
                for n in filter((x)->x>=bask, active_bids)
                    delete!(active_bids, n)
                end
                bbid = isempty(active_bids) ? 0 : maximum(active_bids)
            end
        end

        best_bids[i] = bbid
        best_asks[i] = bask
        if i % 1000000 == 0
            println(i)
        end
    end
    return best_bids, best_asks
end

end