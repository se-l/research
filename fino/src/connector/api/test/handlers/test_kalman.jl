# test_kalman.jl

using Pkg
Pkg.activate("C:\\repos\\research\\fino")

using Serialization
using Fino: Paths, Security, Holding, SecurityType, security_type_equity, security_type_option, option_right_call,
option_right_put, OptionRight, Option, ensure_py_initialized_fino, start_py_worker
using Fino.WS: parse_pb, RequestKalmanInitPb, KalmanHandler, MessagePb, ResponseKalmanInitPb, pb2bytes, MessagePb, ChannelPb, ActionPb,
SSVIParamsPb, ensure_py_initialized

# ============================================================
# Load cached request
# ============================================================
fn = raw"RequestKalmanInit-CSCO-2026-02-10T000000-f8a29cebcd3fc983d03c3980d1c5a1ceb69d53d542c19f8bdaee6b5885bfad69.bin"
fp = joinpath(Paths.PATH_API_CACHE, fn)

@info "Loading cached request from: $fp"
msg = open(fp, "r") do f
    Serialization.deserialize(f)
end
msg = parse_pb(msg, MessagePb)
# force type conversion
channel = ChannelPb.T(Int(msg.channel))
action  = ActionPb.T(Int(msg.action))
msg = MessagePb(channel, msg.id, action, msg.payload)

@info "Loaded MessagePb" channel=msg.channel action=msg.action id=msg.id payload_bytes=length(msg.payload)

ensure_py_initialized_fino()
ensure_py_initialized()
start_py_worker()

# ============================================================
# Decode the inner RequestKalmanInitPb from the payload
# ============================================================
request = parse_pb(msg.payload, RequestKalmanInitPb)
cache_req_key = KalmanHandler.get_cache_request_key(request)

@info "Decoded request" underlying=request.underlying cache_key=cache_req_key

# ============================================================
# Run directly — no WS needed
# ============================================================
params_raw, covariance = KalmanHandler.get_kalman_init(request)

@info "get_kalman_init result:"
@info "  # SSVI param blobs: $(length(params_raw))"
@info "  covariance size:    $(size(covariance))"

for (i, p) in enumerate(params_raw)
    state = parse_pb(UInt8.(p), SSVIParamsPb)
    mp = state.model_params
    @info "  tenor[$i]: $(state.tenor_dt)  θ=$(round(mp.theta, digits=6))  ρ=$(round(mp.rho, digits=6))  ψ=$(round(mp.psi, digits=6))"
end
