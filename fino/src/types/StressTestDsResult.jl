using Dates

struct StressTestDsResult
    holdings::Portfolio
    ds_dnlv::Dict{Float64, Float64}
    delta_total::Float64
    delta_total_across_ds::Float64
    weighted_dnlv::Float64
    marginal_utility_by_holding::Dict{Holding, Float64}
    total_objective::Float64
    marginal_weighted_objective_by_holding::Dict{Holding, Float64}
    tag::String
end

function StressTestDsResult(
    holdings::Portfolio,
    ds_dnlv::Dict{Float64, Float64},
    delta_total::Float64,
    delta_total_across_ds::Float64;
    weighted_dnlv::Float64 = 0.0,
    marginal_utility_by_holding::Dict = Dict{Holding, Float64}(),
    total_objective::Float64 = 0.0,
    marginal_weighted_objective_by_holding::Dict = Dict{Holding, Float64}(),
    tag::String = ""
)
    StressTestDsResult(holdings, ds_dnlv, delta_total, delta_total_across_ds,
        weighted_dnlv, marginal_utility_by_holding, total_objective,
        marginal_weighted_objective_by_holding, tag)
end

function log_marginal_utility(res::StressTestDsResult)
    for h in res.holdings
        key = Holding(h.symbol, sign(h.quantity))
        util = get(res.marginal_utility_by_holding, key, nothing)
        @info "$(h.symbol): quantity=$(sign(h.quantity)), utility=$util"
    end
end

function log_marginal_objective(res::StressTestDsResult)
    for h in res.holdings
        key = Holding(h.symbol, sign(h.quantity))
        obj = get(res.marginal_weighted_objective_by_holding, key, nothing)
        @info "$(h.symbol): quantity=$(sign(h.quantity)), objective=$obj"
    end
end