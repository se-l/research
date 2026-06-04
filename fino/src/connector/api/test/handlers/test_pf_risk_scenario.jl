# test_ssvi_calibration.jl

include(joinpath(@__DIR__, "src\\init.jl"))

using Serialization
using Fino: Paths, Security, Holding, SecurityType, security_type_equity, security_type_option, option_right_call, option_right_put, OptionRight, Option,
    ensure_py_initialized_fino, start_py_worker
using Fino.WS: parse_pb, RequestPfRiskScenariosPb, PfRiskScenarioHandler, MessagePb, ResponsePfRiskScenarioPb, pb2bytes, ensure_py_initialized

# ============================================================
# Load cached req
# ============================================================
fn = raw"RequestPfRiskScenarios-HPE-2026-06-01T123001-0d391a0b59ecde3c9ac97d364d239fa4d5a7ddc4d0bb972445d3a2299f5a20f9.bin"
fp = joinpath(Paths.PATH_API_CACHE, fn)

@info "Loading cached req from: $fp"
msg = open(fp, "r") do f
    Serialization.deserialize(f)
end
msg = parse_pb(msg, MessagePb)
# force type conversion
using Fino.WS: MessagePb, ChannelPb, ActionPb
channel = ChannelPb.T(Int(msg.channel))
action  = ActionPb.T(Int(msg.action))
msg = MessagePb(channel, msg.id, action, msg.payload)

@info "Loaded MessagePb" channel=msg.channel action=msg.action id=msg.id payload_bytes=length(msg.payload)

# ============================================================
# Decode the inner RequestKalmanInitPb from the payload
# ============================================================
req = parse_pb(msg.payload, RequestPfRiskScenariosPb)
cache_req_key = PfRiskScenarioHandler.get_cache_request_key(req)

@info "Decoded request" underlying=req.underlying cache_key=cache_req_key

ensure_py_initialized_fino()
ensure_py_initialized()
start_py_worker()

# ============================================================
# Run directly — no WS needed
# ============================================================
res = PfRiskScenarioHandler.compute_pf_risk_scenario(req)

@info "  # PfRiskScenarioHandler res: $(res)"
