module CalibrateIVSCpu

using LinearAlgebra
using LeastSquaresOptim

using ..PricingEngine

# ─────────────────────────────────────────────────────────────
# eSSVI total variance
# ─────────────────────────────────────────────────────────────

function f_essvi_total_variance(
    k::AbstractVector{Float64},
    theta::Float64, rho::Float64, psi::Float64
)::Vector{Float64}
    inner = @. (psi * k + theta * rho)^2 + theta^2 * (1.0 - rho^2)
    @. 0.5 * (theta + rho * psi * k + sqrt(max(inner, 0.0)))
end

function f_essvi_iv(
k::AbstractVector{Float64},
theta::Float64, rho::Float64, psi::Float64,
tenor::Float64
)::Vector{Float64}
    tv = f_essvi_total_variance(k, theta, rho, psi)
    @. sqrt(max(tv / tenor, 0.0))
end

# ─────────────────────────────────────────────────────────────
# Logistic spread-weighted residuals
# ─────────────────────────────────────────────────────────────

function logistic_spread_weighted_residuals(
    model_prices::AbstractVector{Float64},
    bids::AbstractVector{Float64},
    asks::AbstractVector{Float64};
    w_min::Float64 = 0.05,
    alpha::Float64 = 15.0,
    eps::Float64 = 0.01,
    min_half_spread::Float64 = 1e-12
)::Vector{Float64}
    mid = @. 0.5 * (bids + asks)
    half_spread = @. max(0.5 * (asks - bids), min_half_spread)

    e = model_prices .- mid
    u = abs.(e) ./ half_spread

    s_star = clamp(((1.0 - eps) - w_min) / (1.0 - w_min), 1e-12, 1.0 - 1e-12)
    logit_s = log(s_star / (1.0 - s_star))
    beta = 1.0 - logit_s / alpha

    z = @. alpha * (u - beta)
    w = @. w_min + (1.0 - w_min) / (1.0 + exp(- z))

    return e .* w
end

# ─────────────────────────────────────────────────────────────
# Build per-option (theta, rho, psi) vectors from flat params
# ─────────────────────────────────────────────────────────────

function expand_params(
    calibration_params::AbstractVector{Float64},
    tenor_offsets::Vector{Tuple{Int, Int}},
    n::Int
)
    v_theta = Vector{Float64}(undef, n)
    v_rho = Vector{Float64}(undef, n)
    v_psi = Vector{Float64}(undef, n)
    for (i, (start, stop)) in enumerate(tenor_offsets)
        v_theta[start:stop] .= calibration_params[(i - 1) * 3 + 1]
        v_rho[start:stop] .= calibration_params[(i - 1) * 3 + 2]
        v_psi[start:stop] .= calibration_params[(i - 1) * 3 + 3]
    end
    return v_theta, v_rho, v_psi
end

# ─────────────────────────────────────────────────────────────
# Residual function  (fills F in-place for LeastSquaresOptim)
# ─────────────────────────────────────────────────────────────

function make_residual_fn(
    bids, asks, s, k, t, v_is_call,
    mny_fwd_ln, # log-forward moneyness, precomputed on host
    tenor_offsets,
    dividends,
    calculation_date
)
    n = length(t)

    function residual!(F::AbstractVector, x::AbstractVector)
        v_theta, v_rho, v_psi = expand_params(x, tenor_offsets, n)

        variance = f_essvi_total_variance.(mny_fwd_ln, v_theta, v_rho, v_psi) ./ t
        variance .= max.(variance, 0.0)
        model_iv = sqrt.(variance)
        replace!(model_iv, NaN => 0.0)

        # ← Replace with your GPU pricing call
        model_prices = PricingEngine.get_v_price_fd(
            s, k, t, v_is_call, model_iv,
            dividends, calculation_date
        )

        residuals = logistic_spread_weighted_residuals(
            Float64.(model_prices), bids, asks
        )
        replace!(residuals, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
        F .= residuals
    end

    return residual!
end

# ─────────────────────────────────────────────────────────────
# Jacobian  (central-difference, one GPU batch)
# ─────────────────────────────────────────────────────────────

function make_jacobian_fn(
    bids, asks, s, k, t, v_is_call,
    mny_fwd_ln,
    tenor_offsets,
    dividends,
    calculation_date,
    diff_step::Float64
)
    n = length(t)
    n_params = length(tenor_offsets) * 3

    function jacobian!(J::AbstractMatrix, x::AbstractVector)
        n_scenarios = 1 + 2 * n_params
        X = repeat(x', n_scenarios, 1) # (n_scenarios × n_params)
        h = Vector{Float64}(undef, n_params)

        for j in 1:n_params
            hj = x[j] != 0.0 ? abs(x[j]) * diff_step : diff_step
            h[j] = hj
            X[2j, j] = x[j] + hj # +h row
            X[2j + 1, j] = x[j] - hj # -h row
        end

        # Build all model_iv scenarios
        model_iv_all = Matrix{Float64}(undef, n_scenarios, n)
        for p in 1:n_scenarios
            params = X[p, :]
            v_theta, v_rho, v_psi = expand_params(params, tenor_offsets, n)
            variance = f_essvi_total_variance.(mny_fwd_ln, v_theta, v_rho, v_psi) ./ t
            variance .= max.(variance, 0.0)
            iv = sqrt.(variance)
            replace!(iv, NaN => 0.0)
            model_iv_all[p, :] = iv
        end

        # Single GPU batch call for all scenarios
        s_all = repeat(s, n_scenarios)
        k_all = repeat(k, n_scenarios)
        t_all = repeat(t, n_scenarios)
        is_call_all = repeat(v_is_call, n_scenarios)
        iv_flat = vec(model_iv_all) # row-major flatten

        prices_all = PricingEngine.get_v_price_fd(
            s_all, k_all, t_all, is_call_all, iv_flat,
            dividends, calculation_date
        )
        prices_mat = reshape(Float64.(prices_all), n_scenarios, n)

        # Residuals per scenario
        res_all = Matrix{Float64}(undef, n_scenarios, n)
        for p in 1:n_scenarios
            res_all[p, :] = logistic_spread_weighted_residuals(
                prices_mat[p, :], bids, asks
            )
        end

        # Central-difference columns:  J[residual, param]
        for j in 1:n_params
            r_plus = res_all[2j, :]
            r_minus = res_all[2j + 1, :]
            J[:, j] = (r_plus .- r_minus) ./ (2.0 * h[j])
        end
    end

    return jacobian!
end

# ─────────────────────────────────────────────────────────────
# Main calibration entry point
# ─────────────────────────────────────────────────────────────

"""
    calibrate_surface(
        calibration_params, bids, asks, s, k, t, v_is_call,
        mny_fwd_ln, tenor_offsets, dividends, calculation_date;
        diff_step, max_nfev, verbose
    ) -> (x_opt, result)

Drop-in Julia replacement for the Python `least_squares` call.
Bounds enforce theta > 0 per tenor; rho and psi are unconstrained.
"""
function calibrate_surface(
    calibration_params,
    bids,
    asks,
    s, k, t,
    v_is_call,
    mny_fwd_ln,
    tenor_offsets::Vector{Tuple{Int, Int}},
    dividends,
    calculation_date;
    diff_step = 0.01,
    max_nfev::Int = 1000,
    verbose::Bool = true
)
    n_tenors = length(tenor_offsets)

    # Bounds: theta > 0, rho and psi unconstrained
    lower = repeat([0.0, -Inf, -Inf], n_tenors)
    upper = repeat([Inf, Inf, Inf], n_tenors)

    residual! = make_residual_fn(
        bids, asks, s, k, t, v_is_call,
        mny_fwd_ln, tenor_offsets, dividends, calculation_date
    )
    jacobian! = make_jacobian_fn(
        bids, asks, s, k, t, v_is_call,
        mny_fwd_ln, tenor_offsets, dividends, calculation_date,
        diff_step
    )

    n_residuals = length(t)
    x0 = copy(calibration_params)

    problem = LeastSquaresProblem(
        x = x0,
        f! = residual!,
        j! = jacobian!,
        output_length = n_residuals
    )

    result = optimize!(
        problem,
        LevenbergMarquardt(); # algorithm
            lower = lower,
            upper = upper,
            iterations = max_nfev,
            show_trace = verbose
            # Note: LeastSquaresOptim does not have a built-in Huber loss selector.
            # To get Huber-like behaviour, scale residuals inside residual!() via
            # the logistic_spread_weighted_residuals weighting, which already
            # down-weights outliers similarly to Huber.
    )

    verbose && @info "Calibration done. converged=$(result.converged) cost=$(result.ssr) iters=$(result.iterations)"

    return result.minimizer, result
end

end  # module