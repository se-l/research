# test_kalman.jl

using Pkg
Pkg.activate("C:\\repos\\research\\fino")

using Serialization
using Fino: Paths, Security, Holding, SecurityType, security_type_equity, security_type_option, option_right_call, option_right_put, OptionRight, Option
using Fino.WS: parse_pb, RequestKalmanInitPb, Kalman, MessagePb, ResponseKalmanInitPb, pb2bytes

# ============================================================
# Load cached request
# ============================================================
fn = raw"RequestKalmanInit-FDX-2025-12-17T000000-0a4976768bb5637fa66d8c59bf21c1c2020394fbe77bd608ece8393503636aba.bin"
fp = joinpath(Paths.PATH_API_CACHE, fn)

@info "Loading cached request from: $fp"
msg = open(fp, "r") do f
    Serialization.deserialize(f)
end
# force type conversion
using Fino.WS: MessagePb, ChannelPb, ActionPb
channel = ChannelPb.T(Int(msg.channel))
action  = ActionPb.T(Int(msg.action))
msg = MessagePb(channel, msg.id, action, msg.payload)

@info "Loaded MessagePb" channel=msg.channel action=msg.action id=msg.id payload_bytes=length(msg.payload)

# ============================================================
# Decode the inner RequestKalmanInitPb from the payload
# ============================================================
request = parse_pb(msg.payload, RequestKalmanInitPb)
cache_req_key = Kalman.get_cache_request_key(request)

@info "Decoded request" underlying=request.underlying cache_key=cache_req_key

# ============================================================
# Run directly — no WS needed
# ============================================================
params_raw, covariance = Kalman.get_kalman_init(pb2bytes(request))

@info "get_kalman_init result:"
@info "  # SSVI param blobs: $(length(params_raw))"
@info "  covariance size:    $(size(covariance))"


using Fino.WS: SSVIParamsPb

for (i, p) in enumerate(params_raw)
    state = parse_pb(UInt8.(p), SSVIParamsPb)
    mp = state.model_params
    @info "  tenor[$i]: $(state.tenor_dt)  θ=$(round(mp.theta, digits=6))  ρ=$(round(mp.rho, digits=6))  ψ=$(round(mp.psi, digits=6))"
end
