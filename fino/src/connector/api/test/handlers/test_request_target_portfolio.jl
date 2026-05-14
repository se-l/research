# test_ssvi_calibration.jl

using Revise
using Pkg
Pkg.activate("C:\\repos\\research\\fino")
#include("src\\Fino.jl")

using Serialization
using Dates
using Fino: Paths, Security, Holding, SecurityType, security_type_equity, security_type_option, option_right_call, option_right_put, OptionRight, Option,
    ensure_py_initialized_fino, start_py_worker, pf_to_py_pf, pydate, WS
using Fino.WS: DT_FMT_PB, parse_pb, RequestTargetPortfoliosPb, TargetPortfoliosHandler, MessagePb, ResponseTargetPortfoliosPb, pb2bytes, ensure_py_initialized

# ============================================================
# Load cached req
# ============================================================
fn = raw"RequestTargetPortfolios-FDX-2025-12-18T094500-c232f7b7fa6d71dd5a2d8de6fcf67efa0ac83ceb81ac33bf6bdc42cdb14b5edc.bin"
fn = raw"RequestTargetPortfolios-CSCO-2026-02-11T094500-b94d334cb60d4c9d838c3de9e710da631ebb3240ac53f6e122594f6d8c2f1dd9.bin"
fp = joinpath(Paths.PATH_API_CACHE, fn)

@info "Loading cached req from: $fp"
msg = open(fp, "r") do f
    Serialization.deserialize(f)
end
msg = parse_pb(msg, MessagePb)
# force type conversion
using Fino.WS: MessagePb, ChannelPb, ActionPb, ensure_py_initialized
channel = ChannelPb.T(Int(msg.channel))
action  = ActionPb.T(Int(msg.action))
msg = MessagePb(channel, msg.id, action, msg.payload)

@info "Loaded MessagePb" channel=msg.channel action=msg.action id=msg.id payload_bytes=length(msg.payload)

# ============================================================
# Decode the inner RequestKalmanInitPb from the payload
# ============================================================
req = parse_pb(msg.payload, RequestTargetPortfoliosPb)
cache_key = TargetPortfoliosHandler.get_cache_request_key(req)

#@info "Decoded request" underlying=req.underlying cache_key=cache_req_key

# ============================================================
# Run directly — no WS needed
# ============================================================
ensure_py_initialized_fino()
ensure_py_initialized()
#start_py_worker()

result = TargetPortfoliosHandler.get_target_portfolios(req)
(best_pf_pb, er, pf_target, res, max_obj_val) = result

@info "TargetPortfoliosHandler compute_target_portfolios_subs: $(best_pf_pb)"
@info "TargetPortfoliosHandler compute_target_portfolios_subs: $(best_pf_pb.target_portfolios[1].holdings)"
@info "TargetPortfoliosHandler compute_target_portfolios_subs: $(best_pf_pb.target_portfolios[1].result_stress_test_ds.marginal_utility_by_holding)"
@info "TargetPortfoliosHandler compute_target_portfolios_subs: $res"

@info "TargetPortfoliosHandler get_target_portfolios: $result"
s0 = req.market_data_snaps[end].underlying_price
calc_date = Date(req.ts, DT_FMT_PB)
res = TargetPortfoliosHandler.compute_target_portfolios_subs(
    best_pf_pb,
    result.res.pyo_model,
    result.pf_target,
    result.max_obj_val,
    req,
    s0,
    result.best_pf_pb.target_portfolios,
    calc_date,
    result.er
)

@info "TargetPortfoliosHandler compute_target_portfolios_subs: $(res.target_portfolios[1])"
@info "TargetPortfoliosHandler compute_target_portfolios_subs: $(res.target_portfolios[1].result_stress_test_ds.marginal_utility_by_holding)"
@info "TargetPortfoliosHandler compute_target_portfolios_subs: $res"