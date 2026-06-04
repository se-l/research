module SSVICalibrationHandler

using PythonCall
using ProtoBuf
using Nettle  # for sha256 hash
using Dates
using Base.Threads
using SHA

using ...DividendManager
using ...PricingEngine
using ...YieldCurve: get_last_zero_curve
using ...Fino
using ..WS

const BACKGROUND_TASKS = Set{Task}()

struct ContractBatch
    contract    :: Option
    quotes      :: Vector          # QuotePb or TradePb
    v_bid_price :: Union{Vector{Float64}, Nothing}
    v_ask_price :: Union{Vector{Float64}, Nothing}
    range       :: UnitRange{Int}  # row slice in the global flat arrays
end

"""
    handle_on_msg(ws::WebSocket, msg::MessagePb)

WebSocket handler for SSVI calibration requests.
"""
function handle_on_msg(ws, msg::MessagePb)

    req = parse_pb(msg.payload, RequestSSVICalibrationPb)
    @info "$handle_on_msg: underlying=$(req.underlying), ts=$(req.ts)"

    sanity_check_payload(req) || return

    cache_req_key = get_cache_key_if_not_present(ws, msg, req, get_cache_request_key, get_cache_request_fn)
    cache_req_key === nothing && return

    spawn_task(BACKGROUND_TASKS, "SSVI Calibration") do
        send_ssvi_params(ws, req, cache_req_key, msg)
    end
end

"""
    sanity_check_payload(payload::RequestSSVICalibrationPb) -> Bool
"""
function sanity_check_payload(payload::RequestSSVICalibrationPb)::Bool
    if isempty(payload.underlying)
        @error "No underlying specified."
        return false
    end
    md = payload.market_data_history
    if isempty(md.quotes) && isempty(md.trades)
        @error "No market data present to calibrate on."
        return false
    end
    return true
end

"""
    get_cache_request_key(payload::RequestSSVICalibrationPb) -> String
"""
function get_cache_request_key(payload::RequestSSVICalibrationPb)::String
    r = payload
    md = r.market_data_history
    return bytes2hex(sha256(string("RequestSSVICalibrationPb",
        r.underlying,
        md.ts_start,
        md.ts_end,
        sum(length(q.quotes) for q in values(md.quotes)),
        sum(length(t.trades) for t in values(md.trades))
    )))
end

"""
    get_cache_request_fn(request::RequestSSVICalibrationPb, key_out::String) -> String
"""
function get_cache_request_fn(request::RequestSSVICalibrationPb, key_out::String)::String
    r = request
    ts = replace(r.ts, ":" => "")
    return "RequestSSVICalibrationPb-$(r.underlying)-$ts-$key_out.bin"
end

"""
    build_calibration_inputs(samples::AbstractDict) -> NamedTuple

Flatten sorted `CalibrationItem` values into the vectors expected by
`CalibrateIVS.calibrate_surface`, and compute `tenor_offsets`.
Mirrors the pre-processing done in the Python `calibrate_surface`.
"""
function build_calibration_inputs(samples::AbstractDict)
    sorted_items = [samples[k] for k in sort(collect(keys(samples)))]
    n_tenors = length(sorted_items)

    # Flat arrays (one entry per option across all tenors, in tenor order)
    bids        = Float64[]
    asks        = Float64[]
    s           = Float64[]
    k           = Float64[]
    t           = Float64[]
    v_is_call   = Int[]
    mny_fwd_ln  = Float64[]
    tenor_index = Int[]   # which tenor (0-based) each option belongs to

    for (i, item) in enumerate(sorted_items)
        n = length(item.price)
        append!(bids,       item.bid_price)
        append!(asks,       item.ask_price)
        append!(s,          item.spot)
        append!(k,          item.strike)
        append!(t,          fill(item.tenor, n))
        append!(v_is_call,  item.v_is_call)
        append!(mny_fwd_ln, item.mny_fwd_ln)
        append!(tenor_index, fill(i - 1, n))   # 0-based tenor index
    end

    # Build (start, stop) offsets for each tenor (1-based Julia ranges)
    tenor_offsets = Tuple{Int,Int}[]
    for i in 0:(n_tenors - 1)
        positions = findall(==(i), tenor_index)
        push!(tenor_offsets, (first(positions), last(positions)))
    end

    # Initial guess: [0.05, -0.1, 0.1] per tenor (no prior available)
    calibration_params = Float64[]
    for _ in 1:n_tenors
        append!(calibration_params, [0.05, -0.1, 0.1])
    end

    calculation_date = sorted_items[1].calculation_date

    return (
        sorted_items     = sorted_items,
        calibration_params = calibration_params,
        bids             = bids,
        asks             = asks,
        s                = s,
        k                = k,
        t                = t,
        v_is_call        = v_is_call,
        mny_fwd_ln       = mny_fwd_ln,
        tenor_offsets    = tenor_offsets,
        calculation_date = calculation_date,
    )
end

"""
    qt2calibration_item(contract, quotes, v_iv, v_vega, v_bid_price, v_ask_price) -> CalibrationItem

Builds a CalibrationItem from pre-computed IV and vega vectors (already sliced for this contract).
"""
function qt2calibration_item(
    contract::Option,
    quotes::Vector,
    v_iv::AbstractVector,
    v_vega::AbstractVector;
    v_bid_price=nothing,
    v_ask_price=nothing,
)
    n = length(quotes)
    n == 0 && throw(ArgumentError("No quotes provided"))

    equity    = Equity(contract.underlying_symbol)
    v_strike  = fill(contract.strike, n)
    v_ts      = [DateTime(q.ts, DT_FMT_PB) for q in quotes]
    calc_date = Date(v_ts[1])

    dividends = DividendManager.get_dividends(equity, calc_date)

    v_spot    = Float32[q.price_underlying for q in quotes]
    v_expiry  = fill(date_to_eod(contract.expiry), n)
    v_tenor   = Float32.(get_v_tenor(v_expiry, v_ts))
    v_mny     = get_moneyness_fwd_ln(equity, v_strike, v_spot, v_tenor, date_to_sod(calc_date))
    v_is_call = UInt8.(fill(contract.right == option_right_call ? 1 : 0, n))

    if v_bid_price !== nothing && v_ask_price !== nothing
        v_price = Float32.((v_bid_price .+ v_ask_price) ./ 2)
    else
        v_price = Float32[q.price for q in quotes]
    end

    v_iv_local   = collect(Float32, v_iv)
    v_vega_local = collect(Float32, v_vega)
    v_iv_local[isnan.(v_iv_local)] .= 0

    # ix_include is guaranteed to have at least one true by the caller
    ix_include = map(v -> !isnan(v) && v != 0, v_vega_local)

    return CalibrationItem(
        mny_fwd_ln=v_mny[ix_include],
        strike=v_strike[ix_include],
        calculation_date=calc_date,
        tenor_dt=contract.expiry,
        v_is_call=v_is_call[ix_include],
        iv=v_iv_local[ix_include],
        price=v_price[ix_include],
        spot=v_spot[ix_include],
        dividends=dividends,
        weights=v_vega_local[ix_include],
        vega=v_vega_local[ix_include],
        ts=v_ts[ix_include],
        bid_price=v_bid_price !== nothing ? v_bid_price[ix_include] : nothing,
        ask_price=v_ask_price !== nothing ? v_ask_price[ix_include] : nothing,
    )
end

"""
    market_history2calibration_items(market_data_history::MarketDataHistoryPb) -> Vector{CalibrationItem}

Batches ALL contracts into a single `get_v_iv_fd` + `get_v_vega_fd` call, then slices
results back per contract.
"""
function market_history2calibration_items(market_data_history::MarketDataHistoryPb)::Vector{CalibrationItem}
    quotes_with_data = 0
    for (ib_symbol, quotes) in market_data_history.quotes
        Int(quotes.security_type) != Int(SecurityTypePb.OPTION) && continue
        if length(quotes.quotes) > 0
#            println(ib_symbol)
            quotes_with_data += 1
        end
    end
    @info "Contracts with quote data: $quotes_with_data / $(length(market_data_history.quotes))"
    # ------------------------------------------------------------------ #
    # 1.  Collect per-contract metadata without calling the pricer yet.   #
    # ------------------------------------------------------------------ #

    batches = ContractBatch[]

    # --- Global flat arrays (filled below) ---
    g_price    = Float32[]
    g_spot     = Float32[]
    g_strike   = Float32[]
    g_tenor    = Float32[]
    g_is_call  = UInt8[]
    # Yield-curve / dividend arrays may differ per underlying, so we carry
    # per-row copies as matrices; simplest approach: store per-batch and
    # build per-call in the pricer loop (see step 2).

    # Accumulate per-contract flat data
    contract_slices = NamedTuple{
        (:contract, :quotes, :v_bid_price, :v_ask_price,
         :v_price, :v_spot, :v_strike, :v_tenor, :v_is_call,
         :yield_curve, :div_amounts, :div_times, :range),
    }[]

    cursor = 0

    # Helper to register one contract's data
    function _register!(contract, quotes, v_bid_price, v_ask_price)
        n = length(quotes)
        n == 0 && return

        v_ts      = [DateTime(q.ts, DT_FMT_PB) for q in quotes]
        calc_date = Date(v_ts[1])
        equity    = Equity(contract.underlying_symbol)

        v_expiry  = fill(date_to_eod(contract.expiry), n)
        v_tenor_c = Float32.(get_v_tenor(v_expiry, v_ts))
        v_spot_c  = Float32[q.price_underlying for q in quotes]
        v_strike_c = Float32.(fill(contract.strike, n))
        v_is_call_c = UInt8.(fill(contract.right == option_right_call ? 1 : 0, n))

        if v_bid_price !== nothing && v_ask_price !== nothing
            v_price_c = Float32.((v_bid_price .+ v_ask_price) ./ 2)
        else
            v_price_c = Float32[q.price for q in quotes]
        end

        yield_curve = get_last_zero_curve(calc_date, equity, -14)
        div_amounts, div_times = DividendManager.get_dividend_amount_times(equity, calc_date)

        rng = (cursor + 1):(cursor + n)
        cursor += n

        push!(contract_slices, (
            contract    = contract,
            quotes      = quotes,
            v_bid_price = v_bid_price,
            v_ask_price = v_ask_price,
            v_price     = v_price_c,
            v_spot      = v_spot_c,
            v_strike    = v_strike_c,
            v_tenor     = v_tenor_c,
            v_is_call   = v_is_call_c,
            yield_curve = yield_curve,
            div_amounts = div_amounts,
            div_times   = div_times,
            range       = rng,
        ))
    end

    # ---------- quotes ----------
    for (ib_symbol, quotes) in market_data_history.quotes
        Int(quotes.security_type) != Int(SecurityTypePb.OPTION) && continue
        length(quotes.quotes) == 0 && continue

        contract    = option_from_ib_symbol(ib_symbol)
        v_bid_price = Float64[q.bid for q in quotes.quotes]
        v_ask_price = Float64[q.ask for q in quotes.quotes]

        ix_nan      = isnan.(v_bid_price) .| isnan.(v_ask_price)
        v_bid_price = v_bid_price[map(!, ix_nan)]
        v_ask_price = v_ask_price[map(!, ix_nan)]
        quotes_cp   = quotes.quotes[map(!, ix_nan)]

        _register!(contract, quotes_cp, v_bid_price, v_ask_price)
    end

    # ---------- trades ----------
    for (ib_symbol, trades) in market_data_history.trades
        trades.security_type != security_type_option && continue
        length(trades.trades) == 0 && continue

        contract  = option_from_ib_symbol(ib_symbol)
        trades_cp = [t for t in trades.trades if t.price > 0]

        _register!(contract, trades_cp, nothing, nothing)
    end

    isempty(contract_slices) && return CalibrationItem[]

    # ------------------------------------------------------------------ #
    # 2.  Single batched pricer call per contract group.                  #
    #     Because yield-curve / dividend data can differ per underlying   #
    #     we group by (yield_curve, div_amounts, div_times) key.          #
    #     In practice most requests share one underlying, so this is      #
    #     usually a single call.                                           #
    # ------------------------------------------------------------------ #

    # Allocate output arrays spanning all contracts
    total_n   = cursor
    g_iv      = Vector{Float32}(undef, total_n)
    g_vega    = Vector{Float32}(undef, total_n)

    # Group slices that share the same curve/dividend data
    # Key: object-id pair (avoids deep equality on arrays)
    groups = Dict{Tuple{UInt,UInt}, Vector{Int}}()
    for (idx, sl) in enumerate(contract_slices)
        key = (objectid(sl.yield_curve), objectid(sl.div_amounts))
        push!(get!(groups, key, Int[]), idx)
    end

    for (_, idxs) in groups
        # Build concatenated vectors for this group
        gv_price   = reduce(vcat, contract_slices[i].v_price   for i in idxs)
        gv_spot    = reduce(vcat, contract_slices[i].v_spot    for i in idxs)
        gv_strike  = reduce(vcat, contract_slices[i].v_strike  for i in idxs)
        gv_tenor   = reduce(vcat, contract_slices[i].v_tenor   for i in idxs)
        gv_is_call = reduce(vcat, contract_slices[i].v_is_call for i in idxs)

        # All slices in this group share the same curve / dividends
        sl1        = contract_slices[idxs[1]]
        yc         = sl1.yield_curve
        div_amounts = sl1.div_amounts
        div_times   = sl1.div_times

        gv_iv = PricingEngine.get_v_iv_fd(
            gv_price, gv_spot, gv_strike, gv_tenor, gv_is_call,
            yc.rates, yc.times, div_amounts, div_times,
        )
        gv_iv[isnan.(gv_iv)] .= 0

        gv_vega = PricingEngine.get_v_vega_fd(
            gv_spot, gv_strike, gv_tenor, gv_iv, gv_is_call,
            yc.rates, yc.times, div_amounts, div_times,
        )

        # Scatter results back into global arrays using each contract's range
        local_cursor = 0
        for i in idxs
            sl  = contract_slices[i]
            n   = length(sl.range)
            src = (local_cursor + 1):(local_cursor + n)
            g_iv[sl.range]   = gv_iv[src]
            g_vega[sl.range] = gv_vega[src]
            local_cursor += n
        end
    end

    # ------------------------------------------------------------------ #
    # 3.  Build CalibrationItems from pre-computed IV / vega slices.      #
    # ------------------------------------------------------------------ #
    calibration_items = CalibrationItem[]
    for sl in contract_slices
        v_iv_sl   = g_iv[sl.range]
        v_vega_sl = g_vega[sl.range]

        ix_include = map(v -> !isnan(v) && v != 0, v_vega_sl)
        if !any(ix_include)
            n = length(sl.range)
#            @warn "market_history2calibration_items: all vegas filtered for $(sl.contract.symbol)" n_quotes=n n_nonzero_iv=count(v_iv_sl .!= 0) n_nonnan_vega=count(map(v -> !isnan(v), v_vega_sl))
            continue
        end

        item = qt2calibration_item(
            sl.contract, sl.quotes,
            v_iv_sl, v_vega_sl;
            v_bid_price = sl.v_bid_price,
            v_ask_price = sl.v_ask_price,
        )
        item !== nothing && push!(calibration_items, item)
    end

    return union_calibration_items(calibration_items)
end

"""
    get_ssvi_params(request_ssvi_calibration) -> SSVISurfParams
"""
function get_ssvi_params(req::RequestSSVICalibrationPb)::SSVISurfParams
    calibration_items = market_history2calibration_items(req.market_data_history)

    if isempty(calibration_items)
        @warn "get_ssvi_params: no valid calibration items for $(req.underlying)"
        return SSVISurfParams()
    end

    samples = Dict(item.tenor_dt => item for item in calibration_items)

    inp = build_calibration_inputs(samples)
    div_amounts, div_times = DividendManager.get_dividend_amount_times(req.underlying, inp.calculation_date)
    yield_curve = get_last_zero_curve(inp.calculation_date, req.underlying, -14)

    res, _ = CalibrateIVS.calibrate_surface(
        inp.calibration_params,
        inp.bids,
        inp.asks,
        inp.s,
        inp.k,
        inp.t,
        inp.v_is_call,
        inp.mny_fwd_ln,
        inp.tenor_offsets,
        yield_curve.rates, yield_curve.times,
        div_amounts, div_times;
        diff_step = 0.01f0,
        max_nfev  = 1000,
        verbose   = true,
    )

    # Decode flat result vector: each tenor occupies 3 consecutive entries
    # (theta, rho, psi) — identical layout to the Python side.
    ssvi_surf_params = SSVISurfParams()
    for (i, item) in enumerate(inp.sorted_items)
        theta = res[i * 3 - 2]   # i=1 → idx 1, i=2 → idx 4, …
        rho   = res[i * 3 - 1]
        psi   = res[i * 3]
        ssvi_surf_params[item.tenor_dt] = SSVITenorParams(theta, rho, psi)
    end

    return ssvi_surf_params
end

"""
    send_ssvi_params(ws::WebSocket, req::RequestSSVICalibrationPb, cache_request_key::String, msg::MessagePb)
"""
function send_ssvi_params(ws, req::RequestSSVICalibrationPb, cache_request_key::String, msg::MessagePb)
    try
        ssvi_surf_params = get_ssvi_params(req)

        isempty(ssvi_surf_params) && error("No ssvi_params for $(req.underlying)")

        ssvi_params = SSVIParamsPb[]
        for (tenor_dt, tenor_params) in ssvi_surf_params
            param = SSVIParamsPb(
                req.underlying,
                string(tenor_dt),
                SSVIModelParamsPb(tenor_params.theta, tenor_params.rho, tenor_params.psi)
            )
            push!(ssvi_params, param)
        end

        resp = ResponseSSVICalibrationPb(req, ssvi_params)
        @info "Sending # calibration results: $(length(ssvi_params))"
        send_response(ws, msg, pb2bytes(resp), cache_request_key)
    catch e
        reason = "Error: ssvi_calibration: $e"
        @error reason exception=(e, catch_backtrace())
        send_empty_response(ws, msg, pb2bytes(ResponseSSVICalibrationPb(req, SSVIParamsPb[])), reason=reason)
    end
end

end # module SSVICalibrationHandler