"""
    PfRiskScenarios Module
Julia module for computing portfolio risk scenarios from earnings releases.
Handles portfolio impact analysis under various market scenarios.
"""
module PfRiskScenarioHandler

using PythonCall
using ProtoBuf
using Dates
using SHA
using Logging
using Serialization
using Distributed
using ..WS
using ...Fino: EarningsConfig, Holding, Portfolio, next_release_date, PyEarningsReleaseSimulation, from_holding_pb, pf_to_py_pf

# ============================================================================
# Module-level constants and state
# ============================================================================

const BACKGROUND_TASKS = Set{Task}()
const PROCESS_TIMEOUT = 500
const PROCESS_TIMEOUT_TERMINATE_PARENT = 25_000

# ============================================================================
# Cache functions
# ============================================================================

"""
    get_cache_request_key(payload) -> String

Generate SHA256 cache key for portfolio risk scenarios request.
"""
function get_cache_request_key(payload)::String
    # Build portfolio string representation
    pf = join([string(h) for h in values(payload.holdings)], ",")
    
    market_data_ts = ""
    if !isempty(payload.market_data_snaps)
        market_data_ts = payload.market_data_snaps[1].ts
    end

    ts = replace(payload.ts, ":" => "")
    return bytes2hex(sha256(string(
        "RequestPfRiskScenarios",
        payload.underlying,
        ts,
        market_data_ts,
        pf
    )))
end

"""
    get_cache_request_fn(request, key_out) -> String

Generate cache filename for portfolio risk scenarios request.
"""
function get_cache_request_fn(request::RequestPfRiskScenariosPb, key_outer::String)::String
    ts = replace(request.ts, ":" => "")
    return "RequestPfRiskScenarios-$(request.underlying)-$ts-$key_outer.bin"
end

# ============================================================================
# Payload sanity check
# ============================================================================

"""
    sanity_check_payload(payload) -> Bool
Verify that the request has valid underlying and market data.
"""
function sanity_check_payload(payload)::Bool
    if isempty(payload.underlying)
        @error "No underlying specified."
        return false
    end
    
    if isempty(payload.market_data_snaps)
        @error "No market data present to calibration on."
        return false
    end
    
    return true
end

# ============================================================================
# Portfolio risk scenarios computation
# ============================================================================

"""
    get_pf_risk_scenarios(request_pf_risk_scenarios_b::Vector{UInt8}, cfg::EarningsConfig) -> Vector{PfImpact}

Compute portfolio risk scenarios (impacts) from request bytes and configuration.
"""
function get_pf_risk_scenarios(
    req::RequestPfRiskScenariosPb,
    cfg::EarningsConfig
)
    calc_date = Date(req.ts, DT_FMT_PB)
    holdings = Holding[from_holding_pb(h, calc_date) for h in values(req.holdings)]
    pf = Portfolio(holdings)
    
    # Create earnings release simulation
    er = PyEarningsReleaseSimulation(cfg, collect(req.scoped_symbols))
    
    if isempty(er.scoped_options)
        error("No scoped options found. len of market_data_snaps: $(length(req.market_data_snaps)).")
    end
    
    # Set IV parameters if provided
    if hasproperty(req, :params) && !isempty(req.params)
        er.set_ivs0_params(collect(req.params))
    else
        @warn "No params found in request_pf_risk_scenarios."
    end
    
    # Get portfolio impact by option
    py_release_date = pydate(year(cfg.release_date), month(cfg.release_date), day(cfg.release_date))
    return py_pf_impact_by_option.get_pf_impact_by_option(
        cfg.sym,
        pf=pf_to_py_pf(pf, cfg.release_date),
        release_date=py_release_date
    ).pf_impacts
end

# ============================================================================
# Holdings transformation helpers
# ============================================================================

"""
    holdings_simulated2holdings_hedge(holdings_sim, holdings_current) -> Dict{String, Holding}

Convert simulated holdings to hedge holdings by subtracting current holdings.
"""
function holdings_simulated2holdings_hedge_pb(
    holdings_sim::Vector{<:Holding},
    holdings_current::Vector{<:Holding}
)::Dict{String, HoldingPb}
    holdings_hedge = Dict{String, HoldingPb}()
    
    # Add simulated holdings
    for h in holdings_sim
        holdings_hedge[string(h.symbol)] = holding2holding_pb(h)
    end

    # Subtract current holdings
    for h in holdings_current
        sym = string(h.symbol)
        if haskey(holdings_hedge, sym)
            holdings_hedge[sym] = HoldingPb(sym, h.quantity - h.quantity)
        end
    end
    
    # Filter out zero quantity holdings
    return Dict(k => v for (k, v) in holdings_hedge if v.quantity != 0)
end

# ============================================================================
# WebSocket handler
# ============================================================================

function handle_on_msg(ws, msg)
    """
    handle_on_msg(ws, msg)
    Handle incoming portfolio risk scenarios requests from WebSocket.
    """
    req = parse_pb(msg.payload, RequestPfRiskScenariosPb)
    underlying = req.underlying
    @info "underlying=$underlying, ts=$(req.ts)"

    sanity_check_payload(req) || return

    cache_key = get_cache_key_if_not_present(ws, msg, req, get_cache_request_key, get_cache_request_fn)
    cache_key === nothing && return

    spawn_task(BACKGROUND_TASKS, "PfRisk Scenario") do
        send_pf_risk_scenarios(ws, req, cache_key, msg)
    end
end

# ============================================================================
# Send portfolio risk scenarios response
# ============================================================================

function compute_pf_risk_scenario(req::RequestPfRiskScenariosPb)::Vector{PfRiskScenarioPb}
    underlying = req.underlying
    calc_date = Date(req.ts, DT_FMT_PB)
    release_date = next_release_date(underlying, calc_date)

    holdings = Holding[from_holding_pb(h, calc_date) for h in values(req.holdings)]

    cfg = EarningsConfig(
        underlying,
        release_date,
        portfolio=Portfolio(holdings),
        earnings_iv_drop_regressor_model_name_version=EARNINGS_IV_DROP_REGRESSOR_MODEL_NAME_VERSION
    )

    pf_impacts = get_pf_risk_scenarios(req, cfg)

    pf_risk_scenarios = PfRiskScenarioPb[]
    for p in pf_impacts
        holdings_impact = py2jl_holdings(p.holdings)

        push!(pf_risk_scenarios, PfRiskScenarioPb(
            pyconvert(String, cfg.sym),
            holdings_simulated2holdings_hedge_pb(holdings_impact, holdings),
            pyconvert(Float64, p.dPL),
            pyconvert(Float64, p.utility),
            pyconvert(Float64, p.dPLEqHedged),
            Dict(pyconvert(String, p.option) => pyconvert(Float64, p.iv_enter)),
            req.holdings
        ))
    end

    return pf_risk_scenarios
end

function send_pf_risk_scenarios(websocket, req::RequestPfRiskScenariosPb, cache_request_key::String, msg)
    """
    send_pf_risk_scenarios(websocket, request_pf_risk_scenarios, cache_request_key, msg)
    Process and send portfolio risk scenarios via WebSocket.
    """
    try
        pf_risk_scenarios = try
            run_on_py_thread(() -> compute_pf_risk_scenario(req))
        catch e
            rethrow(e)
        end

        @info "Sending pf risk scenarios"
        payload = pb2bytes(ResponsePfRiskScenariosPb(
            Dates.format(now(Dates.UTC), DT_FMT_PB),
            req.underlying,
            pf_risk_scenarios)
        )
        send_response(websocket, msg, payload, cache_request_key)
        cache_result(cache_request_key, payload)

    catch e
        reason = "Error: pf_risk_scenarios async: $e"
        @error reason
        empty_response = ResponsePfRiskScenarios()
        send_empty_response(websocket, msg, pb2bytes(empty_response), reason=reason)
    end
end

end # module PfRiskScenarios