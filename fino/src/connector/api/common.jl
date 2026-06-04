using Dates
using Serialization
using Distributions
using HTTP
using HTTP.WebSockets
using Logging
using ProtoBuf

using ...Fino: Paths, Security, Holding, SecurityType, security_type_equity, security_type_option, option_right_call, option_right_put, OptionRight, Option,
    option_from_ib_symbol, EarningsConfig, Portfolio, StressTestDsResult, Equity
using ..WS

# Define mappings as constants
const _SEC_TYPE_TO_PB = Dict(
    security_type_equity => SecurityTypePb.EQUITY,
    security_type_option => SecurityTypePb.OPTION
)
const _SEC_TYPE_FROM_PB = Dict(v => k for (k, v) in _SEC_TYPE_TO_PB)

# Mapping dictionaries for OptionRight
const _RIGHT_TO_PB = Dict(
    option_right_call => OptionRightPb.CALL,
    option_right_put => OptionRightPb.PUT,
)
const _RIGHT_FROM_PB = Dict(v => k for (k, v) in _RIGHT_TO_PB)


# Convert functions using dictionary lookup
Base.convert(::Type{SecurityTypePb.T}, s::SecurityType) = _SEC_TYPE_TO_PB[s]
Base.convert(::Type{SecurityType}, pb::SecurityTypePb.T) = _SEC_TYPE_FROM_PB[pb]

Base.convert(::Type{OptionRightPb.T}, s::OptionRight) = _RIGHT_TO_PB[s]
Base.convert(::Type{OptionRight}, pb::OptionRightPb.T) = _RIGHT_FROM_PB[pb]


# Holding conversions
function holding2holding_pb(h::T)::HoldingPb where {T <: Holding}
    return HoldingPb(
        string(h.symbol),
        h.quantity,
        contains(string(h.symbol), " ") ? SecurityTypePb.OPTION : SecurityTypePb.EQUITY
    )
end

function Common_pb.HoldingPb(sym::String, quantity::Real)
    security_type = contains(sym, " ") ? SecurityTypePb.OPTION : SecurityTypePb.EQUITY
    return HoldingPb(sym, Float32(quantity), security_type)
end

function py2jl_holdings(py_holdings)
    return Holding[
        let sym = pyconvert(String, pybuiltins.str(h.symbol))
            security = contains(sym, " ") ? option_from_ib_symbol(sym) : Equity(sym)
            Holding(security, pyconvert(Float64, h.quantity))
        end
        for h in py_holdings
    ]
end

"""
    is_response_cached(cache_request_key::String) -> Bool

Check if a cached response exists for the given key.
"""
function is_response_cached(cache_request_key::String)::Bool
    cache_path = joinpath(Paths.PATH_API_CACHE, "$(cache_request_key)Response.bin")
    return isfile(cache_path)
end

"""
    cache_request(msg::MessagePb, fn::String)
    Cache a request message to disk.
"""
function cache_request(msg::MessagePb, fn::String)
    mkpath(Paths.PATH_API_CACHE)
    cache_path = joinpath(Paths.PATH_API_CACHE, fn)
    open(cache_path, "w+") do f
        serialize(f, pb2bytes(msg))
    end
end

"""
    cache_result(fn::String, payload::Vector{UInt8})
    Cache a result payload to disk.
"""
function cache_result(fn::String, payload::Vector{UInt8})
    mkpath(Paths.PATH_API_CACHE)
    cache_path = joinpath(Paths.PATH_API_CACHE, "$(fn)Response.bin")
    open(cache_path, "w+") do f
        write(f, payload)
    end
end


"""
    load_response_from_cache(websocket, cache_key::String, msg::MessagePb)
    Load and send a cached response.
"""
function load_response_from_cache(websocket, cache_key::String, msg::MessagePb)
    @info "Loading response from cache: $cache_key"
    cache_path = joinpath(Paths.PATH_API_CACHE, "$(cache_key)Response.bin")
    open(cache_path, "r") do f
        payload = read(f)
        send_response(websocket, msg, payload)
    end
end

"""
    get_cache_key_if_not_present(websocket, msg, request, get_key_fn, get_fn_fn) -> Union{String, Nothing}
Check cache and return cache key if not cached, otherwise load from cache.
"""
function get_cache_key_if_not_present(websocket, msg, request, get_key_fn::Function, get_fn_fn::Function)::Union{String, Nothing}
    cache_request_key = get_key_fn(request)

    if is_response_cached(cache_request_key)
        load_response_from_cache(websocket, cache_request_key, msg)
        return nothing
    else
        cache_request_fn = get_fn_fn(request, cache_request_key)
        cache_request(msg, cache_request_fn)
        return cache_request_key
    end
end


function send_response(websocket, msg::MessagePb, payload::Vector{UInt8}, cache_fn::Union{String, Nothing}=nothing)
    """
    send_response(websocket, msg::MessagePb, payload::Vector{UInt8}, cache_fn::Union{String, Nothing}=nothing)
    Send a response via WebSocket and optionally cache it.
    """
    response = MessagePb(msg.channel, msg.id, msg.action, payload)
    if cache_fn !== nothing
        cache_result(cache_fn, payload)
    end
    
    # Serialize and send - adjust based on your WebSocket library
    lock(WS_SEND_LOCK) do
        HTTP.WebSockets.send(websocket, pb2bytes(response))
    end
end


function send_empty_response(websocket, msg, payload::Vector{UInt8}; reason::String="")
    """
    send_empty_response(websocket, msg::MessagePb, payload::Vector{UInt8}; reason::String="")
    Send an empty/error response via WebSocket.
    """
    @info "$reason. Sending empty response."
    response = MessagePb(msg.channel, msg.id, ActionPb.SUBSCRIBE, isempty(payload) ? UInt8[] : payload)
    lock(WS_SEND_LOCK) do
        HTTP.WebSockets.send(websocket, pb2bytes(response))
    end
end

"""
    get_density_for_bimodal_t_dist(returns::Vector{Float64}, dx::Float64, a::Float64, b::Float64, c::Float64)

Calculate density for a bimodal t-distribution.
"""
function get_density_for_bimodal_t_dist(returns::Vector{Float64}, dx::Float64, a::Float64, b::Float64, c::Float64)
    d = TDist(a)
    return (pdf.(d, returns .+ dx, loc=b, scale=c) .+ pdf.(d, returns .- dx, loc=b, scale=c)) ./ 2
end

"""
    holdings_pb2portfolio(holdings_pb) -> Portfolio

Convert protobuf holdings map to Portfolio.
"""
function holdings_pb2portfolio(holdings_pb)::Portfolio
    return Portfolio(Dict(k => v.quantity for (k, v) in holdings_pb))
end

"""
    parse_pb(str_::Vector{UInt8}, pb_type::Type)

Parse a protobuf message from bytes.
"""
function parse_pb(payload::Vector{UInt8}, T)
    ProtoBuf.decode(ProtoBuf.ProtoDecoder(IOBuffer(payload)), T)
end

function pb2bytes(payload)
    buf = IOBuffer()
    encoder = ProtoBuf.ProtoEncoder(buf)
    ProtoBuf.encode(encoder, payload)
    take!(buf)
end

"""
    get_mid_iv_from_cache(cache_iv0::Dict{Option, Dict{Float64, Float64}}) -> Dict{String, Float64}

Extract mid IV from cache.
"""
function get_mid_iv_from_cache(cache_iv0::Dict{Option, Dict{Float64, Float64}})
    try
        return Dict(string(o) => cache_iv0[o][1.0] for o in keys(cache_iv0))
    catch
        return Dict{String, Float64}()
    end
end

# Error handling macro (similar to Python decorator)
macro handle_error_async(try_send_empty_response=true, do_log_health=true)
    # This would be implemented as a macro wrapping async functions
    # Julia handles this differently than Python decorators
    quote
        # Implementation depends on your error handling strategy
    end
end

# Helper to get ordered holdings string
function get_ordered_holdings(request_target_portfolios)::String
    holdings = collect(values(request_target_portfolios.holdings))
    sort!(holdings, by=x -> x.symbol)
    return join(["$(h.symbol)$(h.quantity)" for h in holdings], "")
end

function convert_py_sec2jl_sec2(py_sec)
    s = pyconvert(String, py_sec.symbol)
    contains(s, " ") ? option_from_ib_symbol(s) : Equity(s)
end

# ─── Protobuf serialisation ───────────────────────────────────────────────────

"""
    StressTestDsResult2StressTestDsResult_pb(res, request, pf) -> ResultStressTestDsPb
"""
function StressTestDsResult2StressTestDsResult_pb(res::StressTestDsResult, request, pf::Portfolio)::ResultStressTestDsPb
    return ResultStressTestDsPb(
        request.ts,
        request.underlying,
        Dict(string(h.symbol) => holding2holding_pb(h) for h in pf),
        Dict(string(k) => v for (k, v) in res.ds_dnlv),
        res.delta_total,
        res.delta_total_across_ds,
        res.weighted_dnlv,
        Dict(string(h.symbol) => v for (h, v) in res.marginal_utility_by_holding),
        res.total_objective,
        Dict(string(h.symbol) => v for (h, v) in res.marginal_weighted_objective_by_holding),
    )
end

"""
    get_stress_test_ds_pb(request, scoped_options, pf, m_dnlv01_buy, m_dnlv01_sell, v_delta0, cfg, s0) -> ResultStressTestDsPb
"""
function get_stress_test_ds_pb(request, scoped_options::Vector{Option}, pf::Portfolio, m_dnlv01_buy, m_dnlv01_sell, v_delta0, cfg::EarningsConfig, s0::Float64)::ResultStressTestDsPb
    res = get_stress_test_ds(scoped_options, pf, m_dnlv01_buy, m_dnlv01_sell, v_delta0, cfg, s0)
    return StressTestDsResult2StressTestDsResult_pb(res, request, pf)
end

function py2stress_inputs(mats)
    return (
        pyconvert(Matrix{Float64}, mats.m_dnlv01_estimated_buy),
        pyconvert(Matrix{Float64}, mats.m_dnlv01_estimated_sell),
        pyconvert(Vector{Float64}, mats.v_delta0_mid),
    )
end