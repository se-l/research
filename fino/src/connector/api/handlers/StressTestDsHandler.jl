"""
    StressTestDs Module

Julia module for handling stress test calculations for earnings release scenarios.
"""
module StressTestDsHandler

using ProtoBuf
using Dates
using SHA
using Logging
using Serialization
using ..WS
using ...Fino: next_release_date, PyEarningsReleaseSimulation, Portfolio, Holding, EarningsConfig, run_on_py_thread, get_stress_test_ds,
    StressTestDsResult, from_holding_pb, py_ers_mod

# ============================================================================
# Module-level constants and state
# ============================================================================

const BACKGROUND_TASKS = Set{Task}()
const PROCESS_TIMEOUT = 500

# ============================================================================
# Cache functions
# ============================================================================

"""
    get_cache_request_key(request_stress_test::RequestStressTestDsPb) -> String

Generate SHA256 cache key for stress test request.
"""
function get_cache_request_key(req::RequestStressTestDsPb)::String
    ordered_holdings = get_ordered_holdings(req)

    ordered_quotes = ""
    if !isempty(req.market_data_snaps)
        last_snap = req.market_data_snaps[end]
        if hasproperty(last_snap, :option_quotes) && !isempty(last_snap.option_quotes)
            ordered_quotes = join([
                string(t[1]) * string(round(t[2].bid, digits=1)) * string(round(t[2].ask, digits=1))
                for t in sort(collect(last_snap.option_quotes), by=x->x[1])
            ])
        end
    end

    ts = replace(req.ts, ":" => "")
    return bytes2hex(sha256(string(
        "RequestStressTestDs",
        req.underlying,
        ts,
        ordered_holdings,
        ordered_quotes
    )))
end

"""
    get_cache_request_fn(request::RequestStressTestDs_pb, key_out::String) -> String

Generate cache filename for a stress test request.
"""
function get_cache_request_fn(request, key_out::String)::String
    ts = replace(request.ts, ":" => "")
    return "RequestStressTestDs-$(request.underlying)-$ts-$key_out.bin"
end

# ============================================================================
# Helper to get ordered holdings
# ============================================================================

"""
    get_ordered_holdings(request) -> String

Return holdings ordered by symbol as a string.
"""
function get_ordered_holdings(request)
    if !hasproperty(request, :holdings) || isempty(request.holdings)
        return ""
    end
    
    return join([
        string(h.symbol) * string(h.quantity)
        for h in sort(collect(values(request.holdings)), by=x->x.symbol)
    ])
end

# ============================================================================
# Main handler
# ============================================================================

function handle_on_msg(ws, msg)
    """
    handle_on_msg_ds(ws, msg::Message)
    Handle incoming stress test requests from WebSocket.
    """
    req = parse_pb(msg.payload, RequestStressTestDsPb)
    
    sym = req.underlying
    @info "handle_on_msg_ds: underlying=$sym, ts=$(req.ts)"
    
    # Parse calculation date
    calc_date0 = Date(req.ts, DT_FMT_PB)
    
    # Check if there's an earnings release date
    if next_release_date(sym, calc_date0) === nothing
        @info "No earnings release date found for $sym on $calc_date0. Skipping..."
        return
    end
    
    if isempty(req.market_data_snaps)
        @info "No market data found for $sym on $calc_date0."
        return
    end
    
    # Check if any market data snap has option quotes
    if !isempty(req.market_data_snaps) &&
       !any(snap -> !isempty(snap.option_quotes), req.market_data_snaps)
        @info "No market data snap contains any option data found for $sym on $calc_date0."
        return
    end

    cache_key = get_cache_key_if_not_present(ws, msg, req, get_cache_request_key, get_cache_request_fn)
    cache_key === nothing && return

    spawn_task(BACKGROUND_TASKS, "Stress Test Ds") do
        send_stress_test_ds(ws, req, cache_key, msg)
    end
end

# ============================================================================
# Send stress test results
# ============================================================================

function compute_stress_test_ds(req::RequestStressTestDsPb)::ResultStressTestDsPb
    underlying = req.underlying
    s0 = req.market_data_snaps[end].underlying_price
    calc_date = Date(req.ts, DT_FMT_PB)
    release_date = next_release_date(underlying, calc_date)

    holdings = Holding[]
    if hasproperty(req, :holdings)
        for (sym, h_pb) in req.holdings
            if h_pb.quantity != 0
                push!(holdings, from_holding_pb(h_pb, release_date))
            end
        end
    end
    pf = Portfolio(holdings)

    cfg = EarningsConfig(
        underlying,
        release_date,
        plot=false,
        plot_last=false,
        moneyness_limits=(0.8, 1.2),
        abs_delta_limits=(0.1, 0.9),
        min_tenor=0.0,
        max_tenor=1.0,
        add_equity_holdings=false,
        portfolio=pf
    )

    er = PyEarningsReleaseSimulation(cfg)

    if hasproperty(req, :params) && !isempty(req.params)
        er.set_ivs0_params(collect(req.params))
    else
        @warn "No params found in req."
    end
    # Debug
    #    o = collect(er.py_obj.option_universe)[15]
    #    p0 = py_ers_mod.val_from_df(er.py_obj.df0.loc[er.py_obj.ts_pre_release], o.expiry, o.optionContract.strike, o.right, "mid_price")
    #    py_ers_mod.option_in_pf_scope(o, er.py_obj.cfg, er.py_obj.calc_date0, er.py_obj.s0, p0=p0, pf=er.py_obj.pf)
    #    er.py_obj.get_ivs1(0).is_calibrated_slice(o.expiry)
    #    er.py_obj.scoped_symbols
    #    er.py_obj._scoped_options = nothing
    #    er.py_obj.scoped_options

    # er.py_obj._iv_transition_matrices = Dict()

    @info "get_target_portfolios: getting IV transition matrices"
    mats = er.get_iv_transition_matrices(nothing)
    m_buy, m_sell, v_delta = py2stress_inputs(mats)
    scoped_options_jl = map(convert_py_sec2jl_sec2, er.scoped_options)

    stress_test_ds::StressTestDsResult = get_stress_test_ds(
        scoped_options_jl,
        pf,
        m_buy,
        m_sell,
        v_delta,
        cfg,
        Float64(s0)
    )
    stress_test_ds_pb = StressTestDsResult2StressTestDsResult_pb(stress_test_ds, req, pf)

    @info "compute_stress_test_ds() StressTestResults for $(req.underlying) $(cfg.n_contracts) contracts:
            delta_total_across_ds: $(stress_test_ds_pb.delta_total_across_ds)
            delta_total: $(stress_test_ds_pb.delta_total)"

    return stress_test_ds_pb
end

function send_stress_test_ds(
    websocket,
    req::RequestStressTestDsPb,
    cache_request_key::String,
    msg
)
    try
        result = try
            run_on_py_thread(() -> compute_stress_test_ds(req))
        catch e
            rethrow(e)
        end

        if result === nothing
            @error "Error in send_stress_test_ds: result is nothing"
            return
        end

        @info "Sending # stress test results"
        payload = pb2bytes(result)
        send_response(websocket, msg, payload, cache_request_key)

    catch e
        @error "Error: send_stress_test_ds async: $e"
    end
end

end # module StressTestDs