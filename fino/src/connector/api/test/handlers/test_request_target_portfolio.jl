# test_ssvi_calibration.jl

using Revise
include(joinpath(@__DIR__, "src\\init.jl"))

using Serialization
using Dates
using Fino: Paths, Security, Holding, SecurityType, security_type_equity, security_type_option, option_right_call, option_right_put, OptionRight, Option,
    ensure_py_initialized_fino, start_py_worker, pf_to_py_pf, pydate, WS, get_total_objective, get_total_objective, Portfolio
using Fino.WS: DT_FMT_PB, parse_pb, RequestTargetPortfoliosPb, TargetPortfoliosHandler, MessagePb, ResponseTargetPortfoliosPb, pb2bytes, ensure_py_initialized

# ============================================================
# Load cached req
# ============================================================
fn = raw"RequestTargetPortfolios-HPE-2026-06-01T105811-8250c111c6a67d4f7d45a9b33153dd52d045ae4265fbbce74ec7d9791753e2b8.bin"
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
start_py_worker()

result = TargetPortfoliosHandler.get_target_portfolios(req)
(
    pf_target,
    er,
    max_obj_val,
    is_last_transmission,
    s0,
    calc_date,
    m_buy,
    m_sell,
    v_delta,
    scoped_options_jl,
) = result

@info "TargetPortfoliosHandler compute_target_portfolios holdings: $(pf_target)"
#@info "TargetPortfoliosHandler compute_target_portfolios ds_dnlv: $(best_pf_pb.target_portfolios[1].result_stress_test_ds.ds_dnlv)"
#@info "TargetPortfoliosHandler compute_target_portfolios weighted_dnlv: $(best_pf_pb.target_portfolios[1].result_stress_test_ds.weighted_dnlv)"
#@info "TargetPortfoliosHandler compute_target_portfolios marginal_utility_by_holding: $(best_pf_pb.target_portfolios[1].result_stress_test_ds.marginal_utility_by_holding)"
#@info "TargetPortfoliosHandler compute_target_portfolios total_objective: $(best_pf_pb.target_portfolios[1].result_stress_test_ds.total_objective)"
#@info "TargetPortfoliosHandler compute_target_portfolios marginal_scaled_objective_by_holding: $(best_pf_pb.target_portfolios[1].result_stress_test_ds.marginal_scaled_objective_by_holding)"
@info "TargetPortfoliosHandler compute_target_portfolios max_obj_val: $(max_obj_val)"

@info "TargetPortfoliosHandler get_target_portfolios: $result"
sub_pfs = TargetPortfoliosHandler.compute_target_portfolios_subs(
    result.pf_target,
    result.max_obj_val,
    result.er,
    result.m_buy, result.m_sell,
    result.scoped_options_jl
)
@info "Received $(length(sub_pfs)) sub pfs. Original max_obj_val=$(result.max_obj_val)"
for pf in sub_pfs
    @info "obj_val = $(get_total_objective(pf, result.er.cfg, result.scoped_options_jl, result.m_buy, result.m_sell))"
end

portfolios = Portfolio[result.pf_target]
response_pb =  TargetPortfoliosHandler.prepare_target_portfolios_response(
    req,
    result.er,
    portfolios,
    result.s0,
    result.calc_date,
    result.m_buy, result.m_sell, result.v_delta,
    result.scoped_options_jl
)

# Update is_last_transmission to true for the final combined response
response_pb = ResponseTargetPortfoliosPb(
    response_pb.ts,
    response_pb.underlying,
    response_pb.target_portfolios,
    true
)