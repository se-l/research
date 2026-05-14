"""
    Kalman Module

Julia module for handling Kalman filter initialization and calibration.
Processes SSVI surface parameters using Kalman filtering for earnings IV estimation.
"""
module KalmanHandler

using PythonCall
using ProtoBuf
using Dates
using SHA
using Logging
using Serialization
using Base.Threads
using LinearAlgebra
using ...Fino: Equity, trade_days_between_dates, resolution_second, run_on_py_thread
using ..WS

# ============================================================================
# Module-level constants and state
# ============================================================================

const BACKGROUND_TASKS = Set{Task}()
const PROCESS_TIMEOUT = 5000

# ============================================================================
# Cache functions
# ============================================================================

"""
    get_cache_request_key(request::RequestKalmanInitPb) -> String
    Generate SHA256 cache key for RequestKalmanInitPb.
"""
function get_cache_request_key(req::RequestKalmanInitPb)::String
    return bytes2hex(sha256(string(
    "RequestKalmanInit",
        req.underlying,
        req.date_fit_start,
        req.date_fit_end,
        )))
end

"""
    get_cache_request_fn(r, key_out) -> String

Generate cache filename for a Kalman initialization request.
"""
function get_cache_request_fn(r, key_out::String)::String
    ts = replace(r.date_fit_start, ":" => "")
    return "RequestKalmanInit-$(r.underlying)-$ts-$key_out.bin"
end

# ============================================================================
# Payload sanity check
# ============================================================================

"""
    sanity_check_payload(request_kalman_init) -> Bool

Verify that the request has valid start and end dates.
"""
function sanity_check_payload(request_kalman_init)::Bool
    if isempty(request_kalman_init.date_fit_start) || isempty(request_kalman_init.date_fit_end)
        @info "No start or end date specified."
        return false
    end
    return true
end

# ============================================================================
# Main Kalman initialization computation
# ============================================================================

"""
    get_kalman_init(request_kalman_init_b::Vector{UInt8}) -> Tuple{Vector{Vector{UInt8}}, Matrix{Float64}}

Parse request, compute Kalman initial state and covariance matrix from SSVI surfaces.
"""
function get_kalman_init(req::RequestKalmanInitPb)::Tuple{Vector{Vector{UInt8}}, Matrix{Float64}}
    ensure_py_initialized()
    
    # Create Equity object
    py_equity = py_equity_mod.Equity(uppercase(req.underlying))
    
    # Parse dates
    dt_start = Date(req.date_fit_start, DT_FMT_PB)
    dt_end = Date(req.date_fit_end, DT_FMT_PB)
    trade_days = trade_days_between_dates(dt_start, dt_end)

    py_dates = pylist([pydate(year(d), month(d), day(d)) for d in trade_days])

    py_resolution_second = py_enums.Resolution.second
    py_v_ivs = py_surfaces.get_v_ivs(
        py_equity,
        py_dates,
        py_resolution_second,
        0.002;
        arb_free = false
    )
    
    # 5. Filter in Julia, keeping everything as Python objects
    #    ivs.params is a Python dict — len() works via PythonCall
    v_len_params = [pyconvert(Int, pylen(ivs.params)) for ivs in py_v_ivs]
    n_params = first(first(sort(collect(countmap(v_len_params)), by=x -> last(x), rev=true)))

    py_v_ivs_filtered = pylist([ivs for ivs in py_v_ivs
                                    if pyconvert(Int, pylen(ivs.params)) == n_params])

    n_excluded = length(py_v_ivs) - pyconvert(Int, pylen(py_v_ivs_filtered))
    @info "Found $n_params parameters in $(pyconvert(Int, pylen(py_v_ivs))) surfaces. " *
             "Excluding $n_excluded with </> than $n_params parameters."

    # 6. Pass filtered Python list directly to Python train_kalman_initial_state
    kalman_initial_state = py_kalman.train_kalman_initial_state(py_equity, py_v_ivs_filtered)
    
    X       = pyconvert(Vector{Float64}, kalman_initial_state.X)
    tenors  = pyconvert(Vector{Date},     kalman_initial_state.tenors)
    P_py    = kalman_initial_state.P
    n_states = length(X) ÷ 3

    @assert length(X) == 3 * length(tenors)

    # 7. Convert covariance matrix to Julia
    P = pyconvert(Matrix{Float64}, P_py)

    # 8. Build protobuf bytes for each SSVI tenor slice
    params = Vector{Vector{UInt8}}()
    equity_symbol = uppercase(req.underlying)

    for i in 1:n_states
        tenor_iso = string(tenors[i])

        ssvi_params = SSVIParamsPb(
            equity_symbol,
            tenor_iso,
            SSVIModelParamsPb(
                X[(i-1)*3 + 1],
                X[(i-1)*3 + 2],
                X[(i-1)*3 + 3]
            )
        )
        push!(params, pb2bytes(ssvi_params))
    end

    return params, P
end

# ============================================================================
# WebSocket handler
# ============================================================================

function handle_on_msg(ws, msg)
    """
    handle_on_msg(ws, msg)
    Handle incoming Kalman initialization requests from WebSocket.
    """
    req = parse_pb(msg.payload, RequestKalmanInitPb)
    @info "handle_on_msg: underlying=$(req.underlying), ts=$(req.ts)"

    sanity_check_payload(req) || return

    cache_req_key = get_cache_key_if_not_present(ws, msg, req, get_cache_request_key, get_cache_request_fn)
    cache_req_key === nothing && return

    spawn_task(BACKGROUND_TASKS, "Kalman") do
        send_kalman_init(ws, req, cache_req_key, msg)
    end
end

# ============================================================================
# Send Kalman init response
# ============================================================================

function send_kalman_init(
    websocket,
    req::RequestKalmanInitPb,
    cache_request_key::String,
    msg
)
    """
    send_kalman_init(websocket, req, cache_request_key, msg)
    Process and send Kalman initialization results via WebSocket.
    """
    try
        result = run_on_py_thread(() -> get_kalman_init(req))

        if result === nothing || isempty(first(result))
            error("No kalman init result for $(req.underlying)")
        end
        
        if isempty(result) || result === nothing
            tag = "No fitFillIVResult for $(req.underlying)"
            error(tag)
        end
        
        # Build response protobuf
        params_raw, covariance = result

        init_state = [parse_pb(UInt8.(p), SSVIParamsPb) for p in params_raw]
        init_covariance = [VectorDoublePb(covariance[i, :]) for i in 1:size(covariance, 1)]
        
        resp = ResponseKalmanInitPb(
            req,
            init_state,
            init_covariance
        )
        
        @info "Sending # kalman init results"
        payload = pb2bytes(resp)
        send_response(websocket, msg, payload, cache_request_key)
    catch e
        reason = "Error: kalman_init: $e"
        @error reason exception=(e, catch_backtrace())
        send_empty_response(websocket, msg, pb2bytes(ResponseKalmanInitPb(
                req, Vector{SSVIParamsPb}(), Vector{VectorDoublePb}()
            )), reason=reason)
    end
end

# ============================================================================
# Helper function for counting elements
# ============================================================================

"""
    countmap(arr) -> Dict{T, Int}

Count occurrences of each element in array (equivalent to Counter in Python).
"""
function countmap(arr::Vector{T})::Dict{T, Int} where T
    result = Dict{T, Int}()
    for x in arr
        result[x] = get(result, x, 0) + 1
    end
    return result
end

end # module Kalman