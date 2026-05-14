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
using ...Fino: Portfolio, AbstractEarningsReleaseSimulation, PyEarningsReleaseSimulation,
               Holding, EarningsConfig, Option, Equity, next_release_date,
               StressTestDsResult, get_stress_test_ds, holdings2v_q, get_nlv_by_ds, option_from_ib_symbol,
               get_assumed_fill_iv, SolverResult, solve, log_marginal_utility, run_on_py_thread, py_ers_mod,
                from_holding_pb

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

    send_target_portfolios(ws, req, cache_key, msg)
#    spawn_task(BACKGROUND_TASKS, "Target Portfolios") do
#        send_target_portfolios(ws, req, cache_key, msg)
#    end
end

# ============================================================================
# Send target portfolios
# ============================================================================

function send_target_portfolios(websocket, req::RequestTargetPortfoliosPb, cache_key::String, msg::MessagePb)
    """
    Process and send target portfolios based on request parameters
    """
    try
        result = get_target_portfolios(req)
        best_pf_pb = result.best_pf_pb

        @info "Sending # portfolios: $(length(best_pf_pb.target_portfolios))"
        send_response(websocket, msg, pb2bytes(best_pf_pb))

        if !best_pf_pb.is_last_transmission
            s0 = req.market_data_snaps[end].underlying_price
            calc_date = Date(req.ts, DT_FMT_PB)
            send_target_portfolios_subs(
                websocket,
                best_pf_pb,
                result.res.pyo_model,
                result.pf_target,
                result.max_obj_val,
                req,
                cache_key,
                msg,
                s0,
                result.best_pf_pb.target_portfolios,
                calc_date,
                result.er
            )
        end

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

# ============================================================================
# Send target portfolios sub-portfolios
## ============================================================================

function send_target_portfolios_subs(
        websocket,
        best_pf_pb::ResponseTargetPortfoliosPb,
        pyo_model,
        pf_target::Portfolio,
        max_obj_val::Float64,
        req::RequestTargetPortfoliosPb,
        cache_key::String,
        msg::MessagePb,
        s0,
        best_portfolio_lst_pb,
        calc_date0::Date,
        er::AbstractEarningsReleaseSimulation
    )
        response_target_portfolios_pb = run_on_py_thread(() -> compute_target_portfolios_subs(
            best_pf_pb,
            pyo_model,
            pf_target,
            max_obj_val,
            req,
            s0,
            best_portfolio_lst_pb,
            calc_date0,
            er
        ))
        @info "Sending # portfolios: $(length(response_target_portfolios_pb.target_portfolios))"
        send_response(websocket, msg, pb2bytes(response_target_portfolios_pb), cache_key)
        cache_result(cache_key, pb2bytes(response_target_portfolios_pb))
    end

    function compute_target_portfolios_subs(
        best_pf_pb::ResponseTargetPortfoliosPb,
        pyo_model,
        pf_target::Portfolio,
        max_obj_val::Float64,
        req::RequestTargetPortfoliosPb,
        s0,
        best_portfolio_lst_pb,
        calc_date0::Date,
        er::AbstractEarningsReleaseSimulation
    )::ResponseTargetPortfoliosPb
        try
            # Get sub-portfolios
            sub_portfolios, sub_instances = get_sub_portfolios(
                pyo_model,
                pf_target,
                er.scoped_options,
                min_obj_val=0.99 * max_obj_val
            )

            if length(best_pf_pb.target_portfolios) > 0
                first_tf = first(best_pf_pb.target_portfolios)
                holdings = [from_holding_pb(h, calc_date0) for h in values(first_tf.holdings)]
                sub_portfolios, sub_instances = remove_portfolios_contradicting_quantity(
                    sub_portfolios,
                    holdings,
                    sub_instances
                )
            end

            @info "Number of sub-portfolios / instance: $(length(sub_portfolios)) / $(length(sub_instances))"

            mats = er.get_iv_transition_matrices(nothing)
            m_buy, m_sell, v_delta = py2stress_inputs(mats)

            sub_portfolios_pb = Vector{TargetPortfolioPb}()
            for (pf, inst) in zip(sub_portfolios, sub_instances)
                presumed_fill_ivs = er.get_assumed_fill_iv(pf, s0, calc_date0)
                scoped_options_jl = map(convert_py_sec2jl_sec2, er.scoped_options)
                stress_test_ds_result = get_stress_test_ds(
                    req,
                    scoped_options_jl,
                    pf_target,
                    m_buy, m_sell, v_delta,
                    er.cfg,
                    Float64(s0)
                )
                push!(sub_portfolios_pb, TargetPortfolioPb(
                    req.underlying,
                    Dict(string(h.symbol) => holding2holding_pb(h) for h in pf_target),
                    objective_value(inst),
                    Dict(string(sec) => v for (sec, v) in presumed_fill_ivs),
                    scoped_options_jl,
                    StressTestDsResult2StressTestDsResult_pb(stress_test_ds_result, req, pf)
                ))
            end

            # Combine and sort by objective
            sub_portfolios_pb = vcat(sub_portfolios_pb, best_pf_pb.target_portfolios)
            sort!(sub_portfolios_pb, by=x -> x.objective, rev=true)

            return ResponseTargetPortfoliosPb(req.ts, req.underlying, sub_portfolios_pb, true)
        catch e
            @error "Error in compute_target_portfolios_subs" exception=(e, catch_backtrace())
            return ResponseTargetPortfoliosPb(Dates.format(now(Dates.UTC), DT_FMT_PB), req.underlying, best_portfolio_lst_pb, true)
        end
    end

# ============================================================================
# Async Sub-Portfolio Processing
# ============================================================================

function get_sub_portfolios(
    m,
    pf::Portfolio,
    scoped_options;
    min_obj_val::Union{Float64, Nothing}=nothing
)::Tuple{Vector{Portfolio}, Vector{Py}}
    """
    get_sub_portfolios(m, pf, scoped_options, min_obj_val=nothing)
    Asynchronously solve sub-problems in parallel using process pools.
    """
    t0 = time()

    unsolved_instances = py_pf_opt_minlp_pyomo.get_sub_problems_lst(m, pf, m.scoped_options)
    if isempty(unsolved_instances)
        @info "get_sub_portfolios(): No unsolved_instances..."
        return (Portfolio[], Py[])
    end

    sub_instances = Py[]

    # Use Distributed.jl for parallel processing (Julia's equivalent to ProcessPoolExecutor)
    n_workers = min(length(unsolved_instances), Sys.CPU_THREADS ÷ 2)

    # Run parallel tasks using pmap
    results = @sync begin
        pmap(unsolved_instances) do inst
            try
                pp_solve(inst)
            catch e
                @error "Error solving sub-problem: $e"
                nothing
            end
        end
    end

    # Filter and deserialize results
    for r in results
        if r !== nothing
            try
                inst = deserialize(IOBuffer(r))
                push!(sub_instances, inst)
            catch e
                @error "Error deserializing result: $e"
            end
        end
    end

    @info "get_sub_portfolios: Done MP Time: $(time() - t0)s"

    portfolios = Portfolio[]
    out_instances = Py[]
    s_holdings = Set{Symbol}()

    for s_inst in sub_instances
        obj_val = objective_value(s_inst)
        if min_obj_val !== nothing
            @info "Sub obj: $(objective_value(s_inst)); % of min_obj_val: $(100 * (objective_value(s_inst) / min_obj_val))%"
        end
        if min_obj_val !== nothing && obj_val < min_obj_val
            @info "Skipping. sub obj value too low: $(objective_value(s_inst))"
            continue
        end

        holdings = get_instance_holdings(s_inst, scoped_options)
        push!(portfolios, holdings)
        push!(out_instances, s_inst)

        @info "$holdings"
        s_holdings = union(s_holdings, Set{Symbol}(keys(holdings)))
    end

    @info "# Viable additional options: $(length(s_holdings) - length(pf))"
    return portfolios, out_instances
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
    ref_holdings,
    sub_instances
)::Tuple{Vector{Portfolio}, Vector{Py}}

    map_direction = Dict(h.symbol => h.quantity for h in ref_holdings)
    portfolio_out = Portfolio[]
    instances_out = Py[]

    for (i, p) in enumerate(portfolios)
        # Check if all holdings have same direction as reference
        is_compatible = all(
            get(p.holdings, k, 0) * v >= 0
            for (k, v) in map_direction
        )

        if is_compatible
            push!(portfolio_out, p)
            push!(instances_out, sub_instances[i])
        else
            @info "Removing portfolio $p as it contradicts with reference portfolio $ref_holdings"
        end
    end

    return portfolio_out, instances_out
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
#            @info "get_target_portfolios: start" underlying calc_date release_date s0

            holdings = Holding[from_holding_pb(h_pb, release_date)
                           for h_pb in values(req.holdings)
                           if h_pb.quantity != 0]
            portfolio = Portfolio(holdings)
#            @info "get_target_portfolios: portfolio built" n_holdings=length(holdings)

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

            if cfg.n_contracts > pf_position_abs_total
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
                max_obj_val = run_on_py_thread(() -> pyconvert(Float64, py_ers_mod.get_max_obj_value(res)))
                @info "get_target_portfolios: max_obj_val=$max_obj_val"
                is_last_transmission = false
            else
                @info "get_target_portfolios: portfolio already sufficient ($pf_position_abs_total >= $(cfg.n_contracts)), skipping solver"
                er = run_on_py_thread(() -> PyEarningsReleaseSimulation(cfg, scoped_symbols=collect(req.scoped_symbols)))
                @info "get_target_portfolios: PyEarningsReleaseSimulation created"

                if req.params
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

            log_marginal_utility(stress_test_ds)
            stress_test_ds_pb = StressTestDsResult2StressTestDsResult_pb(stress_test_ds, req, pf_target)

            max_obj_val = max_obj_val > 0 ? max_obj_val : stress_test_ds_pb.total_objective
            @info "get_target_portfolios: final objective" max_obj_val

            best_portfolio_lst_pb = [
                TargetPortfolioPb(
                    req.underlying,
                    Dict(string(h.symbol) => holding2holding_pb(h) for h in pf_target),
                    max_obj_val,
                    presumed_fill_ivs,
                    stress_test_ds_pb
                )
            ]

            best_pf_pb = ResponseTargetPortfoliosPb(
                req.ts,
                req.underlying,
                best_portfolio_lst_pb,
                is_last_transmission
            )

            @info "get_target_portfolios: done ✓" underlying is_last_transmission n_portfolios=length(best_portfolio_lst_pb)
            return (best_pf_pb=best_pf_pb, er=er, pf_target=pf_target, res=res, max_obj_val=max_obj_val)
        catch e
            rethrow(e)
        end
end

end # TargetPortfoliosHandler