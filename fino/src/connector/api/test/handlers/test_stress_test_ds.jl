# test_ssvi_calibration.jl

using Pkg
Pkg.activate("C:\\repos\\research\\fino")
#include("src\\Fino.jl")

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
fn = raw"RequestStressTestDs-FDX-2025-12-18T155900-c31d9ddcd902a3a2aafb60f15272d10bdb31f1ecd25e4ea301cf43f7c20b0585.bin"
fn = raw"RequestStressTestDs-FDX-2025-12-18T093734-2eeb3a5a9aa0e3776de865d97cfbe58148b920fa2259e24a1c1e4a7b0199985e.bin"
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
