module TargetPortfoliosHandler

using PythonCall
using ProtoBuf
using DataFrames
using Dates
using SHA
using Serialization
using Logging
using UUIDs
using SparseArrays
using LinearAlgebra
using ..WS
using ...Fino

# ============================================================================
# Module-level constants and state
# ============================================================================

const BACKGROUND_TASKS = Set{Task}()
const PROCESS_TIMEOUT = 1000
const PROCESS_TIMEOUT_TERMINATE_PARENT = 2500

# ============================================================================
# Main handler
# ============================================================================

function handle_on_msg(ws, msg::MessagePb)
    """
    Handle incoming target portfolio requests from ws
    """
    req = parse_pb(msg.payload, RequestTargetPortfoliosPb)

    sym = req.underlying
    @info "handle_on_msg: on_receive: underlying=$sym, ts=$(req.ts)"

    calc_date0 = Date(req.ts, DT_FMT_PB)
    if next_release_date(sym, calc_date0) === nothing
        tag = "No earnings release date found for $sym on $calc_date0. Skipping..."
        send_empty_response_target_portfolio(ws, msg, tag)
        return
    end

    if isempty(req.market_data_snaps)
        tag = "No market data found for $sym on $calc_date0. Returning empty response..."
        send_empty_response_target_portfolio(ws, msg, tag)
        return
    end

    if length(req.market_data_snaps) > 0 &&
       !any(snap -> !isempty(snap.option_quotes), req.market_data_snaps)
        tag = "No market data snap contains any option data found for $sym on $calc_date0. Returning empty response..."
        send_empty_response_target_portfolio(ws, msg, tag)
        return
    end

    cache_key = get_cache_key_if_not_present(ws, msg, req, get_cache_request_key, get_cache_request_fn)
    cache_key === nothing && return

    # send_target_portfolios(ws, req, cache_key, msg)
    spawn_task(BACKGROUND_TASKS, "Target Portfolios") do
        send_target_portfolios(ws, req, cache_key, msg)
    end
end

# ============================================================================
# Send target portfolios
# ============================================================================

function send_target_portfolios(websocket, req::RequestTargetPortfoliosPb, cache_key::String, msg::MessagePb)
    """
    Process and send target portfolios based on request parameters
    """
    try
        t_orig_start = time()
        result = get_target_portfolios(req)
        t_orig = time() - t_orig_start

        # Collection of portfolios to send
        portfolios = Portfolio[result.pf_target]

        t_subs = 0.0
        if !result.is_last_transmission
            t_subs_start = time()
            sub_pfs = run_on_py_thread(() -> compute_target_portfolios_subs(
                result.pf_target,
                result.max_obj_val,
                result.er,
                result.m_buy, result.m_sell,
                result.scoped_options_jl
            ))
            append!(portfolios, sub_pfs)
            t_subs = time() - t_subs_start
        end

        t_format_start = time()
        response_pb = run_on_py_thread(() -> prepare_target_portfolios_response(
            req,
            result.er,
            portfolios,
            result.s0,
            result.calc_date,
            result.m_buy, result.m_sell, result.v_delta,
            result.scoped_options_jl
        ))
        t_format = time() - t_format_start

        @info "Portfolios calculated: Original in $(round(t_orig, digits=3))s, Sub-portfolios in $(round(t_subs, digits=3))s, Formatting in $(round(t_format, digits=3))s. Total: $(round(t_orig + t_subs + t_format, digits=3))s"

        # Update is_last_transmission to true for the final combined response
        response_pb = ResponseTargetPortfoliosPb(
            response_pb.ts,
            response_pb.underlying,
            response_pb.target_portfolios,
            true
        )

        @info "Sending # portfolios: $(length(response_pb.target_portfolios))"
        send_response(websocket, msg, pb2bytes(response_pb), cache_key)
    catch e
        bt = catch_backtrace()
        if e isa PythonCall.PyException
            @error "send_target_portfolios: PythonException" exception=(e, bt) py_msg=sprint(showerror, e)
        else
            @error "send_target_portfolios: exception" exception=(e, bt)
        end
        send_empty_response_target_portfolio(websocket, msg)
    end
end


function compute_target_portfolios_subs(
    pf_target::Portfolio,
    max_obj_val::Float64,
    er::AbstractEarningsReleaseSimulation,
    m_buy, m_sell,
    scoped_options_jl
)::Vector{Portfolio}
    try
        # Get sub-portfolios
        sub_portfolios = get_sub_portfolios(
            pf_target,
            er.cfg,
            scoped_options_jl,
            m_buy,
            m_sell,
            max_obj_val,
        )

        sub_portfolios = remove_portfolios_contradicting_quantity(
            sub_portfolios,
            get_holdings(pf_target)
        )

        @info "Number of sub-portfolios: $(length(sub_portfolios))"
        return sub_portfolios
    catch e
        @error "Error in compute_target_portfolios_subs" exception=(e, catch_backtrace())
        return Portfolio[]
    end
end


function prepare_target_portfolios_response(
    req::RequestTargetPortfoliosPb,
    er::AbstractEarningsReleaseSimulation,
    portfolios::Vector{Portfolio},
    s0,
    calc_date::Date,
    m_buy, m_sell, v_delta,
    scoped_options_jl
)::ResponseTargetPortfoliosPb
    """
    Union, deduplicate, format and sort portfolios into ResponseTargetPortfoliosPb.
    This function is intended to be separate and testable.
    """
    unique_pfs = get_unique_portfolios(portfolios)
    @info "Union of portfolios: $(length(portfolios)) total -> $(length(unique_pfs)) unique"

    pfs_pb = Vector{TargetPortfolioPb}()
    for pf in unique_pfs
        try
            push!(pfs_pb, create_target_portfolio_pb(
                pf, er, req, s0, calc_date, m_buy, m_sell, v_delta, scoped_options_jl
            ))
        catch e
            @error "Error creating TargetPortfolioPb for portfolio $pf" exception=(e, catch_backtrace())
        end
    end

    # Sort by objective descending
    sort!(pfs_pb, by=x -> x.objective, rev=true)

    return ResponseTargetPortfoliosPb(req.ts, req.underlying, pfs_pb, true)
end


function create_target_portfolio_pb(
    pf::Portfolio,
    er::AbstractEarningsReleaseSimulation,
    req::RequestTargetPortfoliosPb,
    s0,
    calc_date::Date,
    m_buy, m_sell, v_delta,
    scoped_options_jl
)::TargetPortfolioPb
    """
    Helper to create a TargetPortfolioPb from a Portfolio and simulation state.
    """
    presumed_fill_ivs = get_assumed_fill_iv(er, pf, s0, calc_date)
    stress_test_ds_result = get_stress_test_ds(
        scoped_options_jl,
        pf,
        m_buy, m_sell, v_delta,
        er.cfg,
        Float64(s0)
    )
    obj_val = get_total_objective(pf, er.cfg, scoped_options_jl, m_buy, m_sell)

    return TargetPortfolioPb(
        req.underlying,
        Dict(string(h.symbol) => holding2holding_pb(h) for h in pf),
        obj_val,
        Dict(string(sec) => v for (sec, v) in presumed_fill_ivs),
        StressTestDsResult2StressTestDsResult_pb(stress_test_ds_result, req, pf)
    )
end


function get_unique_portfolios(pfs::Vector{Portfolio})::Vector{Portfolio}
    """
    Return unique portfolios based on holdings.
    """
    unique_pfs = Portfolio[]
    for p in pfs
        if !any(up -> up.holdings == p.holdings, unique_pfs)
            push!(unique_pfs, p)
        end
    end
    return unique_pfs
end

# ============================================================================
# Async Sub-Portfolio Processing
# ============================================================================

function get_sub_portfolios(
    pf::Portfolio,
    cfg::EarningsConfig,
    scoped_options_jl,
    m_buy,
    m_sell,
    original_objective::Float64
)::Vector{Portfolio}
    """
    get_sub_portfolios(pf, cfg, scoped_options_jl, m_buy, m_sell, original_objective) -> Vector{Portfolio}
    Generate sub-portfolios by slightly adjusting the positions and filter by objective value.
    """
    t0 = time()
    portfolios = Portfolio[]
    min_obj_val = original_objective * cfg.sub_portfolios_threshold

    for h in pf
        pf_reduced = copy(pf)
        # reduce the position of each holding by 1. That is -2 to -1 and 3->2
        q = h.quantity
        if q == 0 continue end
        new_q = q - sign(q)
        if new_q == 0
            delete!(pf_reduced.holdings, h.symbol)
        else
            pf_reduced.holdings[h.symbol] = new_q
        end

        for opt in scoped_options_jl
            if opt == h.symbol continue end

            for side in [1.0, -1.0]
                pf_sub = copy(pf_reduced)
                add_holding!(pf_sub, opt, side)

                obj_adj = get_total_objective(pf_sub, cfg, scoped_options_jl, m_buy, m_sell)
                if obj_adj >= min_obj_val
                    push!(portfolios, pf_sub)
                end
            end
        end
    end

    @info "get_sub_portfolios: Done. Time: $(time() - t0)s, found $(length(portfolios)) sub-portfolios"
    return portfolios
end

# ============================================================================
# Parent Process Monitor Thread
# ============================================================================

"""
    start_thread_to_terminate_when_parent_process_dies_or_timeout(ppid, timeout=200)

Start a daemon thread that monitors the parent process and terminates it if it exceeds timeout.
"""
function start_thread_to_terminate_when_parent_process_dies_or_timeout(ppid::Int, timeout::Int=200)
    pid = getpid()
    t0 = time()
    p = nothing

    try
        p = psutil_process(ppid)
    catch e
        @warn "Could not get process $ppid: $e"
        return
    end

    function f()
        while true
            try
                if !is_running(p)
                    @info "Parent process pid: $ppid has terminated. Terminating monitoring thread pid: $pid."
                    break
                end

                if time() - t0 > timeout
                    @error "Killing parent process pid: $ppid as it has exceeded timeout of $(timeout)s. From monitoring thread pid: $pid."
                    try
                        terminate(p)
                    catch e
                        @warn "Error terminating process: $e"
                    end
                end
                sleep(1)
            catch e
                @warn "Error in monitoring thread: $e"
                break
            end
        end
    end

    t = Threads.Thread(f)
    t isa Threads.Thread && Threads.terminate(t, 30.0)  # Make daemon-like
    return t
end

# ============================================================================
# WebSocket Response Functions
# ============================================================================

function send_empty_response_target_portfolio(websocket, msg::MessagePb, reason::String="")
    """
    send_empty_response_target_websocket(websocket, msg; reason="")
    Send an empty target portfolio response via WebSocket.
    """
    @info "Sending empty response: $reason"
    payload = pb2bytes(ResponseTargetPortfoliosPb(
        Dates.format(now(Dates.UTC), DT_FMT_PB),
        "",
        TargetPortfolioPb[],
        true
    ))
    send_empty_response(websocket, msg, payload, reason=reason)
end

# ============================================================================
# Portfolio Filtering
# ============================================================================

"""
    remove_portfolios_contradicting_quantity(portfolios, ref_holdings, sub_instances)

Remove portfolios that have contradicting quantities with reference portfolio.
"""
function remove_portfolios_contradicting_quantity(
    portfolios::Vector{Portfolio},
    ref_holdings
)::Vector{Portfolio}

    map_direction = Dict(h.symbol => h.quantity for h in ref_holdings)
    portfolio_out = Portfolio[]

    for p in portfolios
        # Check if all holdings have same direction as reference
        is_compatible = all(
            get(p.holdings, k, 0) * v >= 0
            for (k, v) in map_direction
        )

        if is_compatible
            push!(portfolio_out, p)
        else
            @info "Removing portfolio $p as it contradicts with reference portfolio $ref_holdings"
        end
    end

    return portfolio_out
end

# ============================================================================
# Cache Functions
# ============================================================================

"""
    get_cache_request_fn(request, key_out)

Generate cache filename for a request.
"""
function get_cache_request_fn(request::RequestTargetPortfoliosPb, key_out::String)::String
    ts = replace(request.ts, ":" => "")
    return "RequestTargetPortfolios-$(request.underlying)-$ts-$key_out.bin"
end

"""
    get_cache_request_key(request_target_portfolios)

Generate SHA256 cache key for request.
"""
function get_cache_request_key(req::RequestTargetPortfoliosPb)::String
    # Order holdings by symbol
    ordered_holdings = join([
        string(h.symbol) * string(h.quantity)
        for h in sort(collect(values(req.holdings)), by=x->x.symbol)
    ])

    # Order quotes by symbol (from last market data snap)
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

    return bytes2hex(sha256(string(
        "RequestTargetPortfolios",
        req.underlying,
        req.ts,
        ordered_holdings,
        ordered_quotes,
        req.params,
        req.n_contracts,
        req.scoped_symbols
    )))
end

# ============================================================================
# Target Portfolio Solver
# ============================================================================

"""
    get_target_portfolio(request_target_portfolios_b, cfg)

Solve for target portfolio given request bytes and configuration.
"""
function get_target_portfolio(req::RequestTargetPortfoliosPb, cfg::EarningsConfig)::SolverResult

    er = PyEarningsReleaseSimulation(cfg, collect(req.scoped_symbols))

    if isempty(er.scoped_options)
        error("No scoped options found. len of market_data_snaps: $(length(req.market_data_snaps)).")
    end

    if !isempty(req.params)
        er.set_ivs0_params(collect(req.params))
    else
        @warn "No params found in request_target_portfolios."
    end

    return solve(er)
end

function get_target_portfolios(req::RequestTargetPortfoliosPb)
    return try
            underlying = req.underlying
            s0 = req.market_data_snaps[end].underlying_price
            calc_date = Date(req.ts, DT_FMT_PB)
            release_date = next_release_date(underlying, calc_date)
            @debug "get_target_portfolios: start" underlying calc_date release_date s0

            holdings = Holding[from_holding_pb(h_pb, release_date)
                           for h_pb in values(req.holdings)
                           if h_pb.quantity != 0]
            portfolio = Portfolio(holdings)
            @debug "get_target_portfolios: portfolio built" n_holdings=length(holdings)

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
                n_contracts=req.n_contracts,
                portfolio=portfolio
            )

            pf_holdings = cfg.portfolio
            pf_position_abs_total = sum(abs(h.quantity) for h in pf_holdings; init=0.0)

            if true#:#cfg.n_contracts > pf_position_abs_total
                @info "get_target_portfolios: solving for target portfolio..."
                res = run_on_py_thread(() -> get_target_portfolio(req, cfg))
                @info "get_target_portfolios: solver done, extracting holdings"
                jl_holdings = run_on_py_thread() do
                    py_holdings = pyconvert(Vector, res.pf.get_holdings())
                    return py2jl_holdings(py_holdings)
                end
                pf_target = Portfolio(jl_holdings)
                @info "get_target_portfolios: pf_target built" n_holdings=length(jl_holdings)

                er::PyEarningsReleaseSimulation = res.er
                @info "get_target_portfolios: extracting objective value"
#                max_obj_val =pyconvert(Float64, py_ers_mod.get_max_obj_value(res))
                max_obj_val = run_on_py_thread(() -> pyconvert(Float64, py_ers_mod.get_max_obj_value(res)))
                @info "get_target_portfolios: max_obj_val=$max_obj_val"
                is_last_transmission = false
            else
                @info "get_target_portfolios: portfolio already sufficient ($pf_position_abs_total >= $(cfg.n_contracts)), skipping solver"
                er = run_on_py_thread(() -> PyEarningsReleaseSimulation(cfg, collect(req.scoped_symbols)))
                @info "get_target_portfolios: PyEarningsReleaseSimulation created"

                if !isnothing(req.params) && !isempty(req.params)
                    run_on_py_thread(() -> er.set_ivs0_params(collect(req.params)))
                    @info "get_target_portfolios: ivs0 params set"
                else
                    @warn "get_target_portfolios: no params found in request"
                end

                max_obj_val = 0.0
                is_last_transmission = true
                pf_target = pf_holdings
            end

            presumed_fill_ivs = run_on_py_thread(() -> get_assumed_fill_iv(er, pf_target, s0, calc_date))
            @info "get_target_portfolios: presumed_fill_ivs done" n=length(presumed_fill_ivs)

            @info "get_target_portfolios: getting IV transition matrices"
            m_buy, m_sell, v_delta = run_on_py_thread() do
                mats = er.get_iv_transition_matrices(nothing)
                m_buy, m_sell, v_delta = py2stress_inputs(mats)
                return m_buy, m_sell, v_delta
            end

            @info "get_target_portfolios: running stress test"
            scoped_options_jl = run_on_py_thread(() -> map(convert_py_sec2jl_sec2, er.scoped_options))

            stress_test_ds = get_stress_test_ds(
                scoped_options_jl,
                pf_target,
                m_buy,
                m_sell,
                v_delta,
                cfg,
                Float64(s0)
            )
            @info "get_target_portfolios: stress test done"
            # Test get_total_objective given existing pyomo instance
#            obj_jl = get_total_objective(pf_target, cfg, scoped_options_jl, m_buy, m_sell)
#            @info "pyomo max_obj_val=$(max_obj_val); jl max_obj_value=$(obj_jl)"

            log_marginal_utility(stress_test_ds)
            @info "get_target_portfolios: done ✓" underlying is_last_transmission
            return (
                pf_target = pf_target,
                er = er,
                max_obj_val = max_obj_val,
                is_last_transmission = is_last_transmission,
                s0 = s0,
                calc_date = calc_date,
                m_buy = m_buy,
                m_sell = m_sell,
                v_delta = v_delta,
                scoped_options_jl = scoped_options_jl
            )
        catch e
            rethrow(e)
        end
end

end # TargetPortfoliosHandler