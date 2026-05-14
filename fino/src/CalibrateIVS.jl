module CalibrateIVS

using LinearAlgebra
using LeastSquaresOptim
using CUDA

using ..PricingEngine
# ─────────────────────────────────────────────────────────────
# eSSVI scalar (CPU only, used for small host-side checks)
# ─────────────────────────────────────────────────────────────

@inline function _essvi_tv_scalar(k::Float32, theta::Float32, rho::Float32, psi::Float32)::Float32
    inner = (psi * k + theta * rho)^2 + theta^2 * (1.0 - rho^2)
    0.5 * (theta + rho * psi * k + sqrt(max(inner, 0.0)))
end

# ─────────────────────────────────────────────────────────────
# GPU kernel: build model_iv for all scenarios in one launch
#
# iv_out : (n_scenarios × n_options)  flat row-major
# params : (n_scenarios × n_params)   flat row-major
# tenor_starts / tenor_stops : per-tenor index ranges (1-based)
# ─────────────────────────────────────────────────────────────

function _essvi_iv_kernel!(
    iv_out,                         # CuMatrix Float32 (n_scenarios × n)
    params,                         # CuMatrix Float32 (n_scenarios × n_params)
    mny_fwd_ln,                     # CuVector Float32 (n)
    t,                              # CuVector Float32 (n)
    tenor_starts, tenor_stops,      # CuVector Int32
    n_scenarios::Int32,
    n_options::Int32,
    n_tenors::Int32
)
    s_idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)  # scenario
    o_idx = Int32((blockIdx().y - 1) * blockDim().y + threadIdx().y)  # option

    (s_idx > n_scenarios || o_idx > n_options) && return nothing

    # find which tenor this option belongs to
    tenor_id = Int32(1)
    @inbounds for t_i in Int32(1):n_tenors
        if o_idx >= tenor_starts[t_i] && o_idx <= tenor_stops[t_i]
            tenor_id = t_i
            break
        end
    end

    @inbounds begin
        theta = params[s_idx, (tenor_id - Int32(1)) * Int32(3) + Int32(1)]
        rho   = params[s_idx, (tenor_id - Int32(1)) * Int32(3) + Int32(2)]
        psi   = params[s_idx, (tenor_id - Int32(1)) * Int32(3) + Int32(3)]

        k   = mny_fwd_ln[o_idx]
        ten = t[o_idx]

        inner = (psi * k + theta * rho)^2 + theta^2 * (1.0f0 - rho^2)
        tv    = 0.5f0 * (theta + rho * psi * k + CUDA.sqrt(max(inner, 0.0f0)))
        var   = tv / ten
        iv    = var > 0.0f0 ? CUDA.sqrt(var) : 0.0f0

        iv_out[s_idx, o_idx] = CUDA.isfinite(iv) ? iv : 0.0f0
    end

    return nothing
end

# ─────────────────────────────────────────────────────────────
# GPU kernel: logistic spread-weighted residuals
# ─────────────────────────────────────────────────────────────

function _residuals_kernel!(
    res_out,        # CuMatrix Float32 (n_scenarios × n)
    prices,         # CuMatrix Float32 (n_scenarios × n)
    bids, asks,     # CuVector Float32 (n)
    n_scenarios::Int32,
    n_options::Int32,
    beta::Float32,
    w_min::Float32  = 0.05f0,
    alpha::Float32  = 15.0f0,
)
    s_idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    o_idx = Int32((blockIdx().y - 1) * blockDim().y + threadIdx().y)

    (s_idx > n_scenarios || o_idx > n_options) && return nothing

    @inbounds begin
        mid         = 0.5f0 * (bids[o_idx] + asks[o_idx])
        half_spread = max(0.5f0 * (asks[o_idx] - bids[o_idx]), 1f-12)

        e = prices[s_idx, o_idx] - mid
        u = abs(e) / half_spread

        z = alpha * (u - beta)
        w = w_min + (1.0f0 - w_min) / (1.0f0 + CUDA.exp(-z))

        r = e * w
        res_out[s_idx, o_idx] = CUDA.isfinite(r) ? r : 0.0f0
    end

    return nothing
end

# ─────────────────────────────────────────────────────────────
# Host helper: precompute logistic beta once
# ─────────────────────────────────────────────────────────────

function _logistic_beta(; w_min=0.05, alpha=15.0, eps=0.01)
    s_star  = clamp(((1.0 - eps) - w_min) / (1.0 - w_min), 1e-12, 1.0 - 1e-12)
    logit_s = log(s_star / (1.0 - s_star))
    Float32(1.0 - logit_s / alpha)
end

# ─────────────────────────────────────────────────────────────
# GPU state: everything that lives in VRAM for the full
# calibration run
# ─────────────────────────────────────────────────────────────

struct GPUCalibState
    # Fixed data (uploaded once)
    cu_s         :: CuVector{Float32}
    cu_k         :: CuVector{Float32}
    cu_t         :: CuVector{Float32}
    cu_v_is_call :: CuVector{UInt8}
    cu_bids      :: CuVector{Float32}
    cu_asks      :: CuVector{Float32}
    cu_mny       :: CuVector{Float32}
    cu_t_starts  :: CuVector{Int32}
    cu_t_stops   :: CuVector{Int32}
    rates_curve  :: Vector{Float32}   # kept on host for PricingEngine API
    rates_times  :: Vector{Float32}
    div_amounts  :: Vector{Float32}
    div_times    :: Vector{Float32}

    # Dimensions
    n_options    :: Int
    n_tenors     :: Int
    n_params     :: Int

    # Precomputed
    logistic_beta :: Float32
end

function GPUCalibState(
    s, k, t, v_is_call, bids, asks, mny_fwd_ln,
    tenor_offsets::Vector{Tuple{Int,Int}},
    rates_curve, rates_times, div_amounts, div_times
)
    n_tenors = length(tenor_offsets)
    starts   = Int32[s for (s,_) in tenor_offsets]
    stops    = Int32[e for (_,e) in tenor_offsets]

    GPUCalibState(
        CuArray(Float32.(s)),
        CuArray(Float32.(k)),
        CuArray(Float32.(t)),
        CuArray(UInt8.(v_is_call)),
        CuArray(Float32.(bids)),
        CuArray(Float32.(asks)),
        CuArray(Float32.(mny_fwd_ln)),
        CuArray(starts),
        CuArray(stops),
        Float32.(rates_curve),
        Float32.(rates_times),
        Float32.(div_amounts),
        Float32.(div_times),
        length(t),
        n_tenors,
        n_tenors * 3,
        _logistic_beta()
    )
end

# ─────────────────────────────────────────────────────────────
# Core: run n_scenarios through GPU, return residuals matrix
# (n_scenarios × n_options) on CPU as Float32
# ─────────────────────────────────────────────────────────────

function _run_scenarios(
    state::GPUCalibState,
    params_mat::Matrix{T}   # (n_scenarios × n_params)
)::Matrix{Float32} where {T<:AbstractFloat}
    n_scenarios = size(params_mat, 1)
    n           = state.n_options

    cu_params  = CuArray(Float32.(params_mat))
    cu_iv_out  = CUDA.zeros(Float32, n_scenarios, n)
    cu_res_out = CUDA.zeros(Float32, n_scenarios, n)

    # 1. Build model_iv on GPU
    threads = (16, 16)
    blocks  = (cld(n_scenarios, 16), cld(n, 16))

    @cuda threads=threads blocks=blocks _essvi_iv_kernel!(
        cu_iv_out, cu_params,
        state.cu_mny, state.cu_t,
        state.cu_t_starts, state.cu_t_stops,
        Int32(n_scenarios), Int32(n), Int32(state.n_tenors)
    )
    CUDA.synchronize()

    # 2. Price all scenarios in one GPU batch (PricingEngine handles CUDA internally)
    s_all       = repeat(Array(state.cu_s),        n_scenarios)
    k_all       = repeat(Array(state.cu_k),        n_scenarios)
    t_all       = repeat(Array(state.cu_t),        n_scenarios)
    is_call_all = repeat(Array(state.cu_v_is_call), n_scenarios)
    iv_flat     = vec(Array(cu_iv_out)')            # column-major → row-major flatten

    prices_f32 = get_v_price_fd(
        s_all, k_all, t_all, Float32.(iv_flat), is_call_all,
        state.rates_curve, state.rates_times,
        state.div_amounts, state.div_times
    )
    cu_prices = CuArray(reshape(Float32.(prices_f32), n, n_scenarios)')  # (n_scenarios × n)

    # 3. Compute residuals on GPU
    @cuda threads=threads blocks=blocks _residuals_kernel!(
        cu_res_out, cu_prices,
        state.cu_bids, state.cu_asks,
        Int32(n_scenarios), Int32(n),
        state.logistic_beta
    )
    CUDA.synchronize()

    return Float32.(Array(cu_res_out))
end

# ─────────────────────────────────────────────────────────────
# Residual closure
# ─────────────────────────────────────────────────────────────

function make_residual_fn(state::GPUCalibState)
    function residual!(F::AbstractVector, x::AbstractVector)
        params_mat = reshape(x, 1, state.n_params)   # 1 scenario
        res = _run_scenarios(state, params_mat)
        F  .= res[1, :]
    end
    return residual!
end

# ─────────────────────────────────────────────────────────────
# Jacobian closure  (central-difference, one GPU batch)
# ─────────────────────────────────────────────────────────────

function make_jacobian_fn(state::GPUCalibState, diff_step::Float32)
    n_params    = state.n_params
    n_scenarios = 1 + 2 * n_params

    function jacobian!(J::AbstractMatrix, x::AbstractVector)
        params_mat = repeat(x', n_scenarios, 1)   # (n_scenarios × n_params)
        h          = Vector{Float32}(undef, n_params)

        for j in 1:n_params
            hj             = x[j] != 0.0f0 ? abs(x[j]) * diff_step : diff_step
            h[j]           = hj
            params_mat[2j,   j] = x[j] + hj
            params_mat[2j+1, j] = x[j] - hj
        end

        res_all = _run_scenarios(state, params_mat)  # (n_scenarios × n)

        for j in 1:n_params
            @. J[:, j] = (res_all[2j, :] - res_all[2j+1, :]) / (2.0f0 * h[j])
        end
    end

    return jacobian!
end

# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

function calibrate_surface(
    calibration_params,
    bids, asks, s, k, t, v_is_call,
    mny_fwd_ln,
    tenor_offsets::Vector{Tuple{Int,Int}},
    rates_curve, rates_times,
    div_amounts, div_times;
    diff_step::Float32 = 0.01f0,
    max_nfev::Int      = 1000,
    verbose::Bool      = true
)
    # Upload everything to VRAM once
    state = GPUCalibState(
        s, k, t, v_is_call, bids, asks, mny_fwd_ln,
        tenor_offsets,
        rates_curve, rates_times, div_amounts, div_times
    )

    n_tenors = state.n_tenors
    lower    = repeat([ 0.0, -Inf, -Inf], n_tenors)
    upper    = repeat([  Inf,  Inf,  Inf], n_tenors)

    problem = LeastSquaresProblem(
        x             = copy(calibration_params),
        f!            = make_residual_fn(state),
        g!            = make_jacobian_fn(state, diff_step),
        output_length = state.n_options
    )

    result = optimize!(
        problem,
        LevenbergMarquardt();
        lower      = lower,
        upper      = upper,
        iterations = max_nfev,
        show_trace = verbose
    )

    verbose && @info "Done. converged=$(result.converged) cost=$(result.ssr) iters=$(result.iterations)"

    return result.minimizer, result
end

end # module