# test_ssvi_calibration.jl

include(joinpath(@__DIR__, "src\\init.jl"))

using Serialization
using Fino: Paths, Security, Holding, SecurityType, security_type_equity, security_type_option, option_right_call, option_right_put, OptionRight, Option
using Fino.WS: parse_pb, RequestSSVICalibrationPb, SSVICalibrationHandler, MessagePb, ResponseSSVICalibrationPb, pb2bytes

# ============================================================
# Load cached req
# ============================================================
fn = raw"RequestSSVICalibrationPb-CSCO-2026-02-11T093242-d4aea7d880f9e17c6f4ef68059958544dc5271e06a106931d61a4f231c5fc5c5.bin"
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
req = parse_pb(msg.payload, RequestSSVICalibrationPb)
cache_req_key = SSVICalibrationHandler.get_cache_request_key(req)

@info "Decoded request" underlying=req.underlying cache_key=cache_req_key

# ============================================================
# Run directly — no WS needed
# ============================================================
res = SSVICalibrationHandler.get_ssvi_params(req)

@info "get_ssvi_params result:"
@info "  # SSVI param blobs: $(length(res))"
