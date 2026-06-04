# test_ssvi_calibration.jl

include(joinpath(@__DIR__, "src\\init.jl"))

using Serialization
using Fino: Paths, Security, Holding, SecurityType, security_type_equity, security_type_option, option_right_call, option_right_put, OptionRight, Option,
        ensure_py_initialized_fino, start_py_worker
using Fino.WS: parse_pb, RequestStressTestDsPb, StressTestDsHandler, MessagePb, pb2bytes, ensure_py_initialized

#using Fino.DividendManager
#using Fino.PricingEngine
#using Fino.YieldCurve: get_last_zero_curve
#using Fino: CalibrationItem, Option, SSVISurfParams, SSVITenorParams, Equity, SecurityType, security_type_option, union_calibration_items, date_to_eod,
#        CalibrateIVS, option_from_ib_symbol, get_v_tenor, get_moneyness_fwd_ln, date_to_sod, option_right_call, option_right_put
#using Fino.WS

# ============================================================
# Load cached req
# ============================================================
fn = raw"RequestStressTestDs-CRWD-2026-06-03T134424-f49d5798f8e70dd5e2aad89b9b330e44c8e073e06c07e72c16398b73231ac321.bin"
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
req = parse_pb(msg.payload, RequestStressTestDsPb)
cache_key = StressTestDsHandler.get_cache_request_key(req)

@info "Decoded request" underlying=req.underlying cache_key=cache_key

ensure_py_initialized_fino()
ensure_py_initialized()
start_py_worker()

# ============================================================
# Run directly — no WS needed
# ============================================================
@info "StressTestRsult: Holdings $(req.holdings)"
@info "StressTestRsult: Holdings $(req.params)"
result = StressTestDsHandler.compute_stress_test_ds(req)

@info "StressTestRsult $(result)"
@info "StressTestRsult ds_dnlv $(result.ds_dnlv)"
@info "StressTestRsult delta_across_ds $(result.delta_across_ds)"
