# stress_test.jl
using Distributions
using LinearAlgebra

# ─── T-distribution density for bimodal ───────────────────────────────────────

"""
    get_density_for_bimodal_t_dist(returns, dx, a, b, c) -> Vector{Float64}

Bimodal t-distribution density — mirrors Python scipy.stats.t.pdf usage.
"""
function get_density_for_bimodal_t_dist(returns::Vector{Float64}, dx::Float64, a::Float64, b::Float64, c::Float64)::Vector{Float64}
    d = LocationScale(b, c, TDist(a))
    return (pdf.(d, returns .+ dx) .+ pdf.(d, returns .- dx)) ./ 2
end

# ─── Core helpers ─────────────────────────────────────────────────────────────

"""
    holdings2v_q(portfolio, scoped_options) -> Vector{Float64}

Map portfolio quantities onto the scoped_options vector.
"""
function holdings2v_q(portfolio::Portfolio, scoped_options)::Vector{Float64}
    return Float64[get(portfolio, o, 0) for o in scoped_options]
end

"""
    get_nlv_by_ds(v_q, m_dnlv01_buy, m_dnlv01_sell) -> Vector{Float64}

PnL across dS scenarios given long/short quantity vectors and NLV matrices.
"""
function get_nlv_by_ds(v_q::Vector{Float64}, m_dnlv01_buy::Matrix{Float64}, m_dnlv01_sell::Matrix{Float64})::Vector{Float64}
    v_q_p = max.(v_q, 0.0)
    v_q_n = min.(v_q, 0.0)
    return vec(sum(v_q_p' .* m_dnlv01_buy, dims=2)) .+
           vec(sum(v_q_n' .* m_dnlv01_sell, dims=2))
end

"""
    get_t_curve(cfg, v_x) -> Vector{Float64}

T-distribution weights for each dS scenario.
"""
function get_t_curve(cfg::EarningsConfig, v_x)::Vector{Float64}
    v_ds_pct = [100.0 * (x - 1.0) for x in sort(collect(v_x))]
    dx, a, b, c = cfg.solver_t_params
    return get_density_for_bimodal_t_dist(v_ds_pct, dx, a, b, c)
end

"""
    get_weighted_dlnv(dct_ds_dnlv, cfg) -> Float64

T-curve weighted average PnL across dS scenarios.
"""
function get_weighted_dlnv(dct_ds_dnlv::Dict{Float64, Float64}, cfg::EarningsConfig)::Float64
    t_curve = get_t_curve(cfg, keys(dct_ds_dnlv))
    t_curve ./= sum(t_curve)
    return dot(t_curve, collect(values(dct_ds_dnlv)))
end

"""
    get_delta_total_across_ds(dct_ds_dnlv, cfg, s0) -> Float64

Weighted average delta across symmetric dS scenarios.
"""
function get_delta_total_across_ds(dct_ds_dnlv::Dict{Float64, Float64}, cfg::EarningsConfig, s0)::Float64
    v_ds_pct = [100.0 * (x - 1.0) for x in cfg.v_ds_ret]
    dx, a, b, c = cfg.solver_t_params
    arr_weight = get_density_for_bimodal_t_dist(v_ds_pct, dx, a, b, c)
    half = length(cfg.v_ds_ret) ÷ 2
    arr_weight_half = arr_weight[1:half]

    deltas = Float64[]
    weights = Float64[]
    for i in 1:half
        i_opp = length(cfg.v_ds_ret) - i + 1
        push!(weights, arr_weight_half[i])
        dnlv_left  = dct_ds_dnlv[cfg.v_ds_ret[i]]
        dnlv_right = dct_ds_dnlv[cfg.v_ds_ret[i_opp]]
        ds = s0 * (cfg.v_ds_ret[i] - cfg.v_ds_ret[i_opp])
        push!(deltas, (dnlv_right - dnlv_left) / ds)
    end
    return -sum(deltas .* weights) / sum(weights)
end

"""
    get_v_target_dnlv(pf, cfg, scoped_options, m_dnlv01_buy, m_dnlv01_sell) -> Vector{Float64}
"""
function get_v_target_dnlv(pf::Portfolio, cfg::EarningsConfig, scoped_options, m_dnlv01_buy, m_dnlv01_sell)::Vector{Float64}
    v_q = holdings2v_q(pf, scoped_options)
    nlv_by_ds = get_nlv_by_ds(v_q, m_dnlv01_buy, m_dnlv01_sell)
    dct_nlv_by_ds = Dict(zip(cfg.v_ds_ret, nlv_by_ds))
    y = get_t_curve(cfg, cfg.v_ds_ret)
    y_scaled = y ./ maximum(y)
    return y_scaled .* dct_nlv_by_ds[1.0]
end

"""
    get_total_objective(pf, cfg, scoped_options, m_dnlv01_buy, m_dnlv01_sell; weight_max_t_curve) -> Float64
"""
function get_total_objective(pf::Portfolio, cfg::EarningsConfig, scoped_options, m_dnlv01_buy, m_dnlv01_sell; weight_max_t_curve::Float64=5.0)::Float64
    v_q = holdings2v_q(pf, scoped_options)
    nlv_by_ds = get_nlv_by_ds(v_q, m_dnlv01_buy, m_dnlv01_sell)
    dct_nlv_by_ds = Dict(zip(cfg.v_ds_ret, nlv_by_ds))
    v_target_dnlv = get_v_target_dnlv(pf, cfg, scoped_options, m_dnlv01_buy, m_dnlv01_sell)
    earning_ds0 = dct_nlv_by_ds[1.0] * weight_max_t_curve
    nlv_mn_t_curve = min.(nlv_by_ds .- v_target_dnlv, 0.0)
    return earning_ds0 + sum(nlv_mn_t_curve)
end

"""
    get_marginal_objective_by_holding(pf, cfg, scoped_options, m_dnlv01_buy, m_dnlv01_sell; weight_max_t_curve) -> Dict
"""
function get_marginal_objective_by_holding(pf::Portfolio, cfg::EarningsConfig, scoped_options, m_dnlv01_buy, m_dnlv01_sell; weight_max_t_curve::Float64=1.0)::Dict
    total_obj_pf = get_total_objective(pf, cfg, scoped_options, m_dnlv01_buy, m_dnlv01_sell; weight_max_t_curve)
    out = Dict()
    for h in pf
        pf_tmp = copy(pf)
        d_q = sign(h.quantity)
        add_holding!(pf_tmp, h.symbol, -d_q)
        obj_tmp = get_total_objective(pf_tmp, cfg, scoped_options, m_dnlv01_buy, m_dnlv01_sell; weight_max_t_curve)
        out[h.symbol] = d_q * (total_obj_pf - obj_tmp)
    end
    return out
end


"""
    get_dPLuPLd(dct_nlv_by_ds, cfg) -> Float64

Weighted PnL up minus weighted PnL down — measures directional skew.
"""
function get_dPLuPLd(dct_nlv_by_ds::Dict{Float64, Float64}, cfg::EarningsConfig)::Float64
    t_curve = get_t_curve(cfg, keys(dct_nlv_by_ds))
    t_curve ./= sum(t_curve)

    sorted_keys = sort(collect(keys(dct_nlv_by_ds)))
    n = length(sorted_keys)

    up_keys   = filter(r -> r > 1.0, sorted_keys)
    down_keys = filter(r -> r < 1.0, sorted_keys)

    n_down = length(down_keys)
    n_up   = length(up_keys)

    weighted_down = sum(t_curve[1:n_down]       .* [dct_nlv_by_ds[r] for r in down_keys])
    weighted_up   = sum(t_curve[end-n_up+1:end] .* [dct_nlv_by_ds[r] for r in up_keys])

    return weighted_up - weighted_down
end

"""
    get_marginal_utility_by_holding(scoped_options, pf, m_dnlv01_buy, m_dnlv01_sell, cfg, dct_nlv_by_ds) -> Dict

For each holding, remove 1 unit and measure the drop in weighted PnL + directional skew improvement.
Note: unlike the Python version, this does NOT delta-hedge when computing marginal utility
(that would require scipy.optimize). The hedge effect is omitted here.
"""
function get_marginal_utility_by_holding(
    scoped_options,
    pf::Portfolio,
    m_dnlv01_buy::Matrix{Float64},
    m_dnlv01_sell::Matrix{Float64},
    cfg::EarningsConfig,
    dct_nlv_by_ds::Dict{Float64, Float64}
)::Dict
    weighted_dnlv_1  = get_weighted_dlnv(dct_nlv_by_ds, cfg)
    dPLuPLd_1        = get_dPLuPLd(dct_nlv_by_ds, cfg)
    # omitted py
#    q_eq_1 = get_hedge_quantity(pf_1, scoped_options=scoped_options, m_dnlv01_buy=m_dnlv01_buy, m_dnlv01_sell=m_dnlv01_sell, cfg=cfg)
#    pf_1.add_holding(equity, q_eq_1)

    out = Dict()
    for h in pf
        pf_0 = copy(pf)
        d_q  = -sign(h.quantity)
        add_holding!(pf_0, h.symbol, d_q)

        v_q_0         = holdings2v_q(pf_0, scoped_options)
        nlv_by_ds_0   = get_nlv_by_ds(v_q_0, m_dnlv01_buy, m_dnlv01_sell)
        dct_nlv_0     = Dict(zip(cfg.v_ds_ret, nlv_by_ds_0))

        weighted_dnlv_0 = get_weighted_dlnv(dct_nlv_0, cfg)
        dPLuPLd_0       = get_dPLuPLd(dct_nlv_0, cfg)

        dPL            = weighted_dnlv_1 - weighted_dnlv_0
        delta_dPLuPLd  = dPLuPLd_1 - dPLuPLd_0

#        d_abs_eq_position = abs(q_eq_1) - abs(pf_0.get(equity, 0))  # The reduction in abs eq exposure

        utility        = dPL + delta_dPLuPLd #+ weight_util_q_eq * d_abs_eq_position

        @info "$(h.symbol): utility=$(utility), dPL=$(dPL), delta_dPLuPLd=$(delta_dPLuPLd)"  # , deltaReduction={weight_util_q_eq * d_abs_eq_position}')
        out[h.symbol] = utility
    end
    return out
end

# ─── Main entry point ─────────────────────────────────────────────────────────

"""
    get_stress_test_ds(scoped_options, pf, m_dnlv01_buy, m_dnlv01_sell, v_delta0, cfg, s0; tag) -> StressTestDsResult
"""
function get_stress_test_ds(
    scoped_options,
    pf::Portfolio,
    m_dnlv01_buy::Matrix{Float64},
    m_dnlv01_sell::Matrix{Float64},
    v_delta0::Vector{Float64},
    cfg::EarningsConfig,
    s0::Float64;
    tag::String = ""
)::StressTestDsResult
    v_q        = holdings2v_q(pf, scoped_options)
    nlv_by_ds  = get_nlv_by_ds(v_q, m_dnlv01_buy, m_dnlv01_sell)
    dct_nlv_by_ds = Dict(zip(cfg.v_ds_ret, nlv_by_ds))
    delta_total = dot(v_q, v_delta0)

    return StressTestDsResult(
        pf,
        dct_nlv_by_ds,
        delta_total,
        get_delta_total_across_ds(dct_nlv_by_ds, cfg, s0);
        weighted_dnlv = get_weighted_dlnv(dct_nlv_by_ds, cfg),
        marginal_utility_by_holding = get_marginal_utility_by_holding(scoped_options, pf, m_dnlv01_buy, m_dnlv01_sell, cfg, dct_nlv_by_ds),
        total_objective = get_total_objective(pf, cfg, scoped_options, m_dnlv01_buy, m_dnlv01_sell; weight_max_t_curve=1.0),
        marginal_weighted_objective_by_holding = get_marginal_objective_by_holding(pf, cfg, scoped_options, m_dnlv01_buy, m_dnlv01_sell; weight_max_t_curve=1.0),
        tag = tag
    )
end
