module PricingEngine

export get_price_fd, get_iv_fd, get_vega_fd, get_delta_fd, get_theta_fd,
     get_gamma_fd, get_speed_fd, get_gamma_decay_fd, get_gamma_vol_fd,
     get_theta_decay_fd, get_vega_decay_fd, get_vanna_fd, get_volga_fd,
     get_v_price_fd, get_v_iv_fd, get_v_vega_fd, get_v_delta_fd,
     get_v_theta_fd, get_v_gamma_fd, get_v_speed_fd, get_v_gamma_decay_fd,
     get_v_gamma_vol_fd, get_v_theta_decay_fd, get_v_vega_decay_fd,
     get_v_vanna_fd, get_v_volga_fd,
     df_0_t, df_t_T, get_rate_at_time_convex_monotone

using LinearAlgebra
using StaticArrays

using CUDA
CUDA.functional()
const CUDA_AVAILABLE = true

# ────────────────────────────────────────────────────────────────────────────
# Rate curve helpers
# ────────────────────────────────────────────────────────────────────────────

@inline function _clampf(x::Float32, lo::Float32, hi::Float32)::Float32
    min(max(x, lo), hi)
end

@inline function get_Z_at_time(t::Float32, rates_curve::AbstractVector{Float32}, rates_times::AbstractVector{Float32})::Float32
    n = length(rates_curve)
    n == 0 && return 0f0
    t <= 0f0 && return 0f0

    @inbounds first_time = rates_times[1]
    @inbounds first_rate = rates_curve[1]
    t <= first_time && return first_rate * t

    @inbounds last_time = rates_times[n]
    @inbounds last_rate = rates_curve[n]
    t >= last_time && return last_rate * t

    for i in 1:n - 1
        @inbounds t0, t1 = rates_times[i], rates_times[i + 1]
        if t <= t1
            @inbounds z0, z1 = rates_curve[i] * t0, rates_curve[i + 1] * t1
            w = (t - t0) / (t1 - t0)
            return (1f0 - w) * z0 + w * z1
        end
    end
    @inbounds return rates_curve[n] * t
end

@inline function df_0_t(t::Float32, rates_curve::Union{Vector{Float32}, CUDA.CuDeviceVector{Float32, 1}}, rates_times::Union{Vector{Float32}, CUDA.CuDeviceVector{Float32, 1}})::Float32
    CUDA.exp(- get_Z_at_time(t, rates_curve, rates_times))
end

@inline function df_0_t(t::Float32, rates_curve::AbstractVector{Float32}, rates_times::AbstractVector{Float32})::Float32
     exp(- get_Z_at_time(t, rates_curve, rates_times))
 end

@inline function df_t_T(t::Float32, T::Float32, rates_curve::Union{Vector{Float32}, CUDA.CuDeviceVector{Float32, 1}}, rates_times::Union{Vector{Float32}, CUDA.CuDeviceVector{Float32, 1}})::Float32
    Zt = get_Z_at_time(t, rates_curve, rates_times)
    ZT = get_Z_at_time(T, rates_curve, rates_times)
    CUDA.exp(- (ZT - Zt))
end

@inline function df_t_T(t::Float32, T::Float32, rates_curve::AbstractVector{Float32}, rates_times::AbstractVector{Float32})::Float32
     Zt = get_Z_at_time(t, rates_curve, rates_times)
     ZT = get_Z_at_time(T, rates_curve, rates_times)
     exp(- (ZT - Zt))
 end

@inline function get_rate_at_time_convex_monotone(t::Float32, rates_curve::AbstractVector{Float32}, rates_times::AbstractVector{Float32})::Float32
    n = length(rates_curve)
    n == 0 && return 0f0

    @inbounds first_rate = rates_curve[1]
    t <= 0f0 && return first_rate

    @inbounds first_time = rates_times[1]
    t <= first_time && return first_rate

    @inbounds last_time = rates_times[n]
    @inbounds last_rate = rates_curve[n]
    t >= last_time && return last_rate

    k = 1
    @inbounds for i in 1:n - 1
        if t <= rates_times[i + 1]
            k = i
            break
        end
    end

    @inline function f_interval(i::Int32, rc::AbstractVector{Float32}, rt::AbstractVector{Float32})::Float32
        @inbounds begin
            h_raw = rt[i + Int32(1)] - rt[i]
            h_raw <= 0f0 && return 0f0

            z_i = rc[i] * rt[i]
            z_ip1 = rc[i + Int32(1)] * rt[i + Int32(1)]
            return (z_ip1 - z_i) / h_raw
        end
    end

    @inline function fhat(i::Int32, rc, rt, n::Int32)::Float32
        if n < Int32(2)
            return 0f0
        end

        if i <= Int32(1)
            return f_interval(Int32(1), rc, rt)
        elseif i >= n
            return f_interval(n - Int32(1), rc, rt)
        end

        fL = f_interval(i - Int32(1), rc, rt)
        fR = f_interval(i, rc, rt)

        if !(isfinite(fL) && isfinite(fR))
            return 0f0
        end
        if fL * fR <= 0f0
            return 0f0
        end

        @inbounds begin
            hL = rt[i] - rt[i - Int32(1)]
            hR = rt[i + Int32(1)] - rt[i]
        end

        if hL <= 0f0 || hR <= 0f0
            return 0f0
        end

        wL = 2f0 * hR + hL
        wR = hR + 2f0 * hL

        denom = (wL / fL) + (wR / fR)
        if !isfinite(denom) || abs(denom) < 1f-20
            return 0f0
        end

        fh = (wL + wR) / denom
        lo = min(fL, fR)
        hi = max(fL, fR)
        return _clampf(fh, lo, hi)
    end

    k32 = Int32(k)
    @inbounds t0 = rates_times[k]
    @inbounds t1 = rates_times[k + 1]
    h = t1 - t0
    if h <= 0f0
        @inbounds return rates_curve[k]
    end

    x::Float32 = (t - t0) / h
    fbar::Float32 = f_interval(k32, rates_curve, rates_times)
    fL_::Float32 = fhat(k32, rates_curve, rates_times, Int32(n))
    fR_::Float32 = fhat(k32 + Int32(1), rates_curve, rates_times, Int32(n))

    lo::Float32 = min(fL_, fR_)
    hi::Float32 = max(fL_, fR_)

    c = 6f0 * (fbar - 0.5f0 * (fL_ + fR_))
    f_mid = 0.5f0 * (fL_ + fR_) + 0.25f0 * c

    if f_mid < lo || f_mid > hi
        target = f_mid < lo ? lo : hi
        c::Float32 = 4f0 * (target - 0.5f0 * (fL_ + fR_))
    end

    f = fL_ + (fR_ - fL_) * x + c * x * (1f0 - x)

    f = _clampf(f, lo - 1f6, hi + 1f6)
#    if CUDA.isfinite(f)
    if isfinite(f)
        return f
    else
        @inbounds return rates_curve[k]
    end
end

function _build_rate_sched_cpu!(r_sched, df_sched, tenor, N_t, rates_curve, rates_times)
     if !isempty(rates_curve)
         dt = tenor > 0f0 ? tenor / Float32(N_t) : 0f0
         for i in 1:RATE_SCHED_STRIDE
             n = i - 1
             t = Float32(n) * dt
             r_sched[i] = get_rate_at_time_convex_monotone(t, rates_curve, rates_times)
             df_sched[i] = (n <= N_t && tenor > 0f0) ? df_t_T(t, tenor, rates_curve, rates_times) : 1f0
         end
     else
         fill!(r_sched, 0f0)
         fill!(df_sched, 1f0)
     end
     return nothing
 end

function _safe_div_pv(div_amounts, div_times, T, rates_curve, rates_times)
    pv = 0f0
    for j in eachindex(div_times)
        td = div_times[j]
        0f0 < td <= T && (pv += div_amounts[j] * df_0_t(td, rates_curve, rates_times))
    end
    return pv
end

if CUDA.functional()

    const RATE_SCHED_STRIDE = Int32(257) # store [0..N_t] inclusive

    function _build_rate_sched_kernel!(
    unique_tenors, n_unique::Int32,
    rates_curve, rates_times,
    time_steps::Int32,
    r_by_tenor, df_by_tenor
    )
        tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        tid > n_unique && return nothing

        n = (blockIdx().y - 1) * blockDim().y + threadIdx().y - Int32(1)
        n >= RATE_SCHED_STRIDE && return nothing

        T = unique_tenors[tid]
        N_t = min(max(time_steps, Int32(10)), Int32(256))
        dt = T > 0f0 ? (T / Float32(N_t)) : 0f0

        r = 0f0
        df = 1f0

        if n <= N_t && T > 0f0 && length(rates_curve) > 0
            t = Float32(n) * dt
            r = get_rate_at_time_convex_monotone(t, rates_curve, rates_times)
            Zt = get_Z_at_time(t, rates_curve, rates_times)
            ZT = get_Z_at_time(T, rates_curve, rates_times)
            df = exp(- (ZT - Zt))
        elseif n > N_t && T > 0f0 && length(rates_curve) > 0
            r = get_rate_at_time_convex_monotone(T, rates_curve, rates_times)
            df = 1f0
        end

        idx = (tid - Int32(1)) * RATE_SCHED_STRIDE + n + Int32(1)
        r_by_tenor[idx] = r
        df_by_tenor[idx] = df

        return nothing
    end

    function _d_fd_price!(
    S::Float32, K::Float32, T::Float32, sigma::Float32, is_call::Bool,
    pv_divs::Float32,
    r_sched, df_sched,
    da, dtimes, n_divs::Int32,
    N_t::Int32, N_s::Int32
    )::Float32
        # Early exit for degenerate cases
        if T <= 0f0 || S <= 0f0 || K <= 0f0 || sigma <= 0f0 || !isfinite(pv_divs)
            return is_call ? max(S - K, 0f0) : max(K - S, 0f0)
        end

        dt = T / Float32(N_t)
        r0 = length(r_sched) > 0 ? @inbounds(r_sched[1]) : 0f0
        S_max = max((S + pv_divs) * exp(min(r0 * T + 4f0 * sigma * sqrt(T), 80f0)), 3f0 * max(S, K))
        dS = S_max / Float32(N_s)

        if !isfinite(dS) || dS <= 0f0
            return is_call ? max(S - K, 0f0) : max(K - S, 0f0)
        end

        # Use stack-allocated arrays (register/local memory)
        # MVector is fixed-size and doesn't allocate on heap
        Vm = @MVector zeros(Float32, 257)
        a = @MVector zeros(Float32, 257)
        b = @MVector zeros(Float32, 257)
        c = @MVector zeros(Float32, 257)
        rhs = @MVector zeros(Float32, 257)

        # Initialize terminal payoff
        @inbounds for i in Int32(1):N_s + Int32(1)
            Si = Float32(i - Int32(1)) * dS
            Vm[i] = is_call ? max(Si - K, 0f0) : max(K - Si, 0f0)
        end

        θ = 0.5f0
        σ² = sigma * sigma

        # Backward time stepping
        n = N_t - Int32(1)
        while n >= Int32(0)
            r = length(r_sched) > 0 ? @inbounds(r_sched[n + Int32(1)]) : 0f0
            df_bnd = length(df_sched) > 0 ? @inbounds(df_sched[n + Int32(1)]) : 1f0

            # Lower boundary (S=0)
            @inbounds begin
                a[Int32(1)] = 0f0
                b[Int32(1)] = 1f0
                c[Int32(1)] = 0f0
                rhs[Int32(1)] = is_call ? 0f0 : K * df_bnd
            end

            # Interior points: Crank-Nicolson discretization
            @inbounds for i in Int32(2) : N_s
                Si = Float32(i - Int32(1)) * dS

                if Si < 1f-8
                    a[i] = 0f0
                    b[i] = 1f0
                    c[i] = 0f0
                    rhs[i] = Vm[i]
                    continue
                end

                Vi = Vm[i]
                Vim1 = Vm[i - Int32(1)]
                Vip1 = Vm[i + Int32(1)]

                cpp = 0.5f0 * σ² * Si * Si / (dS * dS)
                cp = 0.5f0 * r * Si / dS
                c0 = - σ² * Si * Si / (dS * dS) - r

                α = cpp - cp
                β = c0
                γ = cpp + cp

                a[i] = - θ * dt * α
                b[i] = 1f0 - θ * dt * β
                c[i] = - θ * dt * γ
                rhs[i] = Vi * (1f0 + (1f0 - θ) * dt * β) +
                Vim1 * (1f0 - θ) * dt * α +
                Vip1 * (1f0 - θ) * dt * γ
            end

            # Upper boundary (S=S_max)
            @inbounds begin
                a[N_s + Int32(1)] = 0f0
                b[N_s + Int32(1)] = 1f0
                c[N_s + Int32(1)] = 0f0
                rhs[N_s + Int32(1)] = is_call ? (Float32(N_s) * dS - K * df_bnd) : 0f0
            end

            # Thomas algorithm (tridiagonal solver) - INLINED for performance
            nn = N_s + Int32(1)

            # Forward elimination
            @inbounds begin
                inv_b1 = 1f0 / b[Int32(1)]
                c[Int32(1)] *= inv_b1
                rhs[Int32(1)] *= inv_b1
            end

            @inbounds for i in Int32(2):nn
                den = b[i] - a[i] * c[i - Int32(1)]
                if abs(den) < 1f-20
                    den = den < 0f0 ? - 1f-20 : 1f-20
                end
                inv_den = 1f0 / den
                if i < nn
                    c[i] *= inv_den
                end
                rhs[i] = (rhs[i] - a[i] * rhs[i - Int32(1)]) * inv_den
            end

            # Back substitution
            @inbounds for i in nn - Int32(1):- Int32(1):Int32(1)
                rhs[i] -= c[i] * rhs[i + Int32(1)]
            end

            # Apply solution and enforce American early exercise
            @inbounds for i in Int32(1):nn
                Si = Float32(i - Int32(1)) * dS
                val = rhs[i]
                intr = is_call ? max(Si - K, 0f0) : max(K - Si, 0f0)
                Vm[i] = max(val, intr)
            end

            # Apply dividend jumps at this time step
            j = Int32(1)
            while j <= n_divs
                td = @inbounds dtimes[j]
                amt = @inbounds da[j]

                if 0f0 < td < T
                    step_f = td / dt
                    step = Int32(floor(step_f))
                    step = max(Int32(0), min(step, N_t - Int32(1)))

                    if step == n
                        # Interpolate option value after dividend drop
                        @inbounds for i in Int32(1):nn
                            S_post = Float32(i - Int32(1)) * dS - amt

                            if S_post <= 0f0
                                rhs[i] = is_call ? 0f0 : K
                            elseif S_post >= S_max
                                rhs[i] = Vm[N_s + Int32(1)]
                            else
                                idf = S_post / dS
                                ii = Int32(floor(idf))
                                ii = max(Int32(0), min(ii, N_s - Int32(1))) + Int32(1)
                                w = idf - Float32(ii - Int32(1))
                                rhs[i] = (1f0 - w) * Vm[ii] + w * Vm[ii + Int32(1)]
                            end
                        end

                        # Copy back and re-apply early exercise
                        @inbounds for i in Int32(1):nn
                            Si = Float32(i - Int32(1)) * dS
                            intr = is_call ? max(Si - K, 0f0) : max(K - Si, 0f0)
                            Vm[i] = max(rhs[i], intr)
                        end
                    end
                end

                j += Int32(1)
            end

            n -= Int32(1)
        end

        # Final interpolation at spot S
        if S <= 0f0
            return @inbounds Vm[Int32(1)]
        elseif S >= S_max
            return @inbounds Vm[N_s + Int32(1)]
        else
            idf = S / dS
            ii = Int32(floor(idf))
            ii = max(Int32(0), min(ii, N_s - Int32(1))) + Int32(1)
            w = idf - Float32(ii - Int32(1))
            return @inbounds((1f0 - w) * Vm[ii] + w * Vm[ii + Int32(1)])
        end
    end

    function _price_kernel!(
    out,
    spots, strikes, tenors, sigmas, rights,
    tenor_ids, pv_divs_arr,
    r_by_tenor, df_by_tenor,
    da, dtimes, n_divs::Int32,
    N_t::Int32, N_s::Int32
    )
        idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        idx > Int32(length(spots)) && return nothing

        S, K, T, sigma = spots[idx], strikes[idx], tenors[idx], sigmas[idx]
        is_call = rights[idx] != 0x00

        if !isfinite(S) || !isfinite(K) || !isfinite(T) || !isfinite(sigma)
            out[idx] = NaN32
            return nothing
        end

        if T <= 0f0 || S <= 0f0 || K <= 0f0 || sigma <= 0f0
            out[idx] = is_call ? max(S - K, 0f0) : max(K - S, 0f0)
            return nothing
        end

        tid = tenor_ids[idx]
        pv_divs = pv_divs_arr[tid]

        # Extract this tenor's rate schedule
        r_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)
        df_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)

        out[idx] = _d_fd_price!(
            S, K, T, sigma, is_call, pv_divs,
            view(r_by_tenor, r_start:r_start + RATE_SCHED_STRIDE - Int32(1)),
            view(df_by_tenor, df_start:df_start + RATE_SCHED_STRIDE - Int32(1)),
            da, dtimes, n_divs,
            N_t, N_s)

        return nothing
    end

    function _iv_kernel!(
    out,
    prices, spots, strikes, tenors, rights,
    tenor_ids, pv_divs_arr,
    r_by_tenor, df_by_tenor,
    da, dtimes, n_divs::Int32,
    tol::Float32, max_iter::Int32, v_min::Float32, v_max::Float32,
    N_t::Int32, N_s::Int32
    )
        idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        idx > Int32(length(prices)) && return nothing

        p, S, K, T = prices[idx], spots[idx], strikes[idx], tenors[idx]
        is_call = rights[idx] != 0x00

        if !isfinite(p) || !isfinite(S) || !isfinite(K) || !isfinite(T)
            out[idx] = NaN32
            return nothing
        end

        if T <= 0f0 || S <= 0f0 || K <= 0f0 || p <= 0f0
            out[idx] = T <= 0f0 ? 0f0 : NaN32
            return nothing
        end

        intrinsic = is_call ? max(S - K, 0f0) : max(K - S, 0f0)
        if abs(p - intrinsic) < 1f-6
            out[idx] = v_min
            return nothing
        end
        if p < intrinsic - 1f-6
            out[idx] = v_min
            return nothing
        end

        tid = tenor_ids[idx]
        pv_divs = pv_divs_arr[tid]

        r_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)
        df_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)
        r_view = view(r_by_tenor, r_start:r_start + RATE_SCHED_STRIDE - Int32(1))
        df_view = view(df_by_tenor, df_start:df_start + RATE_SCHED_STRIDE - Int32(1))

        p_low = _d_fd_price!(
            S, K, T, v_min, is_call, pv_divs,
            r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        p_high = _d_fd_price!(
            S, K, T, v_max, is_call, pv_divs,
            r_view, df_view, da, dtimes, n_divs, N_t, N_s)

        if p <= p_low + 1f-9
            out[idx] = v_min
            return nothing
        end
        if p >= p_high - 1f-9
            out[idx] = v_max
            return nothing
        end

        lo, hi = v_min, v_max
        mid = 0f0
        rel_tol = tol / max(p, 1f0)

        for _ in Int32(1):max_iter
            mid = 0.5f0 * (lo + hi)
            p_mid = _d_fd_price!(
                S, K, T, mid, is_call, pv_divs,
                r_view, df_view, da, dtimes, n_divs, N_t, N_s)

            diff = p_mid - p
            if abs(diff) < tol || abs(diff) / p < rel_tol
                break
            end
            diff < 0f0 ? (lo = mid) : (hi = mid)
            if hi - lo < v_min * 0.01f0
                break
            end
        end

        out[idx] = mid
        return nothing
    end

    function _pv_divs_host(tenors, div_amounts, div_times, rates_curve, rates_times)
        n = length(tenors)
        pv = Vector{Float32}(undef, n)
        for i in 1:n
            acc = 0f0
            for j in eachindex(div_amounts)
                td = div_times[j]
                0f0 < td <= tenors[i] &&
                (acc += div_amounts[j] * df_0_t(td, rates_curve, rates_times))
            end
            pv[i] = acc
        end
        pv
    end

    function _build_tenor_map(tenors)
        unique_map = Dict{Int32, Int32}()
        tenor_ids = Vector{Int32}(undef, length(tenors))
        unique_tenors = Float32[]

        for i in eachindex(tenors)
            T = tenors[i]
            day_bucket = isfinite(T) && T > 0f0 ? Int32(round(T * 365f0)) : Int32(0)

            if !haskey(unique_map, day_bucket)
                T_bucket = Float32(day_bucket) / 365f0
                push!(unique_tenors, T_bucket)
                unique_map[day_bucket] = Int32(length(unique_tenors))
            end

            tenor_ids[i] = unique_map[day_bucket]
        end

        return tenor_ids, unique_tenors
    end

    function _v_price_gpu(spots, strikes, tenors, sigmas, rights,
    rates_curve, rates_times, div_amounts, div_times,
    time_steps, space_steps)

        n = length(spots)
        N_t = Int32(clamp(time_steps, 10, 256))
        N_s = Int32(clamp(space_steps, 10, 256))

        # Build tenor mapping
        tenor_ids, unique_tenors = _build_tenor_map(tenors)
        n_unique = length(unique_tenors)

        # Compute PV of dividends per unique tenor
        pv_by_tenor = Vector{Float32}(undef, n_unique)
        for tid in 1:n_unique
            T = unique_tenors[tid]
            acc = 0f0
            for j in eachindex(div_amounts)
                td = div_times[j]
                if 0f0 < td <= T
                    acc += div_amounts[j] * df_0_t(td, rates_curve, rates_times)
                end
            end
            pv_by_tenor[tid] = acc
        end

        # Transfer to device - SEQUENTIAL ALLOCATIONS
        cu_spots = CuArray(spots)
        cu_strikes = CuArray(strikes)
        cu_tenors = CuArray(tenors)
        cu_sigmas = CuArray(sigmas)
        cu_rights = CuArray(rights)
        cu_tenor_ids = CuArray(tenor_ids)
        cu_pv = CuArray(pv_by_tenor)
        cu_da = CuArray(div_amounts)
        cu_dt = CuArray(div_times)
        out = CUDA.fill(NaN32, n)

        cu_r_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_df_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))

        cu_unique_tenors = CuArray(unique_tenors)
        cu_rates_curve = CuArray(rates_curve)
        cu_rates_times = CuArray(rates_times)

        # Build on GPU in parallel
        threads_2d = (min(n_unique, 32), min(Int(RATE_SCHED_STRIDE), 32))
        blocks_2d = (cld(n_unique, threads_2d[1]), cld(Int(RATE_SCHED_STRIDE), threads_2d[2]))

        @cuda threads = threads_2d blocks = blocks_2d _build_rate_sched_kernel!(
            cu_unique_tenors, Int32(n_unique),
            cu_rates_curve, cu_rates_times,
            N_t,
            cu_r_by_tenor, cu_df_by_tenor
        )

        nthreads = 256 # CHANGED from 64 - better occupancy
        nblocks = cld(n, nthreads)

        @cuda threads = nthreads blocks = nblocks _price_kernel!(
            out,
            cu_spots, cu_strikes, cu_tenors, cu_sigmas, cu_rights,
            cu_tenor_ids, cu_pv,
            cu_r_by_tenor, cu_df_by_tenor,
            cu_da, cu_dt, Int32(length(div_amounts)),
            N_t, N_s)

        CUDA.synchronize()
        Array(out)
    end

    function _v_iv_gpu(prices, spots, strikes, tenors, rights,
    rates_curve, rates_times, div_amounts, div_times,
    tol, max_iter, v_min, v_max, time_steps, space_steps)

        n = length(prices)
        N_t = Int32(clamp(time_steps, 10, 256))
        N_s = Int32(clamp(space_steps, 10, 256))

        tenor_ids, unique_tenors = _build_tenor_map(tenors)
        n_unique = length(unique_tenors)

        pv_by_tenor = Vector{Float32}(undef, n_unique)
        for tid in 1:n_unique
            T = unique_tenors[tid]
            acc = 0f0
            for j in eachindex(div_amounts)
                td = div_times[j]
                if 0f0 < td <= T
                    acc += div_amounts[j] * df_0_t(td, rates_curve, rates_times)
                end
            end
            pv_by_tenor[tid] = acc
        end

        cu_r_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_df_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))

        cu_unique_tenors = CuArray(unique_tenors)
        cu_rates_curve = CuArray(rates_curve)
        cu_rates_times = CuArray(rates_times)

        # Build on GPU in parallel
        threads_2d = (min(n_unique, 32), min(Int(RATE_SCHED_STRIDE), 32))
        blocks_2d = (cld(n_unique, threads_2d[1]), cld(Int(RATE_SCHED_STRIDE), threads_2d[2]))

        @cuda threads = threads_2d blocks = blocks_2d _build_rate_sched_kernel!(
            cu_unique_tenors, Int32(n_unique),
            cu_rates_curve, cu_rates_times,
            N_t,
            cu_r_by_tenor, cu_df_by_tenor
        )

        cu_prices = CuArray(prices)
        #        cu_spots = CuArray(spots)
        #        cu_strikes = CuArray(strikes)
        cu_spots = CuArray{Float32}(undef, n)
        copyto!(cu_spots, spots)

        cu_strikes = CuArray{Float32}(undef, n)
        copyto!(cu_strikes, strikes)

        cu_tenors = CuArray(tenors)
        cu_rights = CuArray(rights)
        cu_tenor_ids = CuArray(tenor_ids)
        cu_pv = CuArray(pv_by_tenor)
        cu_da = CuArray(div_amounts)
        cu_dt = CuArray(div_times)
        out = CUDA.fill(NaN32, n)

        nthreads = 64
        nblocks = cld(n, nthreads)

        @cuda threads = nthreads blocks = nblocks _iv_kernel!(
            out,
            cu_prices, cu_spots, cu_strikes, cu_tenors, cu_rights,
            cu_tenor_ids, cu_pv,
            cu_r_by_tenor, cu_df_by_tenor,
            cu_da, cu_dt, Int32(length(div_amounts)),
            Float32(tol), Int32(max_iter), Float32(v_min), Float32(v_max),
            N_t, N_s)

        CUDA.synchronize()
        Array(out)
    end

    function _vega_kernel!(
    out,
    spots, strikes, tenors, sigmas, rights,
    tenor_ids, pv_divs_arr,
    r_by_tenor, df_by_tenor,
    da, dtimes, n_divs::Int32,
    d_sigma::Float32,
    N_t::Int32, N_s::Int32
    )
        idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        idx > Int32(length(spots)) && return nothing

        S, K, T, sigma = spots[idx], strikes[idx], tenors[idx], sigmas[idx]
        is_call = rights[idx] != 0x00

        if !isfinite(S) || !isfinite(K) || !isfinite(T) || !isfinite(sigma)
            out[idx] = NaN32
            return nothing
        end

        if T <= 0f0 || S <= 0f0 || K <= 0f0 || sigma <= 0f0
            out[idx] = 0f0
            return nothing
        end

        tid = tenor_ids[idx]
        pv_divs = pv_divs_arr[tid]

        r_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)
        df_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)
        r_view = view(r_by_tenor, r_start:r_start + RATE_SCHED_STRIDE - Int32(1))
        df_view = view(df_by_tenor, df_start:df_start + RATE_SCHED_STRIDE - Int32(1))

        sigma_up = sigma + d_sigma
        sigma_down = max(sigma - d_sigma, 1f-6)

        p_up = _d_fd_price!(S, K, T, sigma_up, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        p_down = _d_fd_price!(S, K, T, sigma_down, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)

        out[idx] = (p_up - p_down) / (sigma_up - sigma_down)

        return nothing
    end

    function _v_vega_gpu(
    spots, strikes, tenors, sigmas, rights,
    rates_curve, rates_times, div_amounts, div_times,
    d_sigma, time_steps, space_steps
    )
        n = length(spots)
        N_t = Int32(clamp(time_steps, 10, 256))
        N_s = Int32(clamp(space_steps, 10, 256))

        tenor_ids, unique_tenors = _build_tenor_map(tenors)
        n_unique = length(unique_tenors)

        pv_by_tenor = Vector{Float32}(undef, n_unique)
        for tid in 1:n_unique
            T = unique_tenors[tid]
            acc = 0f0
            for j in eachindex(div_amounts)
                td = div_times[j]
                if 0f0 < td <= T
                    acc += div_amounts[j] * df_0_t(td, rates_curve, rates_times)
                end
            end
            pv_by_tenor[tid] = acc
        end

        cu_r_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_df_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_unique_tenors = CuArray(unique_tenors)
        cu_rates_curve = CuArray(rates_curve)
        cu_rates_times = CuArray(rates_times)

        threads_2d = (min(n_unique, 32), min(Int(RATE_SCHED_STRIDE), 32))
        blocks_2d = (cld(n_unique, threads_2d[1]), cld(Int(RATE_SCHED_STRIDE), threads_2d[2]))

        @cuda threads = threads_2d blocks = blocks_2d _build_rate_sched_kernel!(
            cu_unique_tenors, Int32(n_unique),
            cu_rates_curve, cu_rates_times,
            N_t,
            cu_r_by_tenor, cu_df_by_tenor
        )

        cu_spots = CuArray(spots)
        cu_strikes = CuArray(strikes)
        cu_tenors = CuArray(tenors)
        cu_sigmas = CuArray(sigmas)
        cu_rights = CuArray(rights)
        cu_tenor_ids = CuArray(tenor_ids)
        cu_pv = CuArray(pv_by_tenor)
        cu_da = CuArray(div_amounts)
        cu_dt = CuArray(div_times)
        out = CUDA.fill(NaN32, n)

        nthreads = 256
        nblocks = cld(n, nthreads)

        @cuda threads = nthreads blocks = nblocks _vega_kernel!(
            out,
            cu_spots, cu_strikes, cu_tenors, cu_sigmas, cu_rights,
            cu_tenor_ids, cu_pv,
            cu_r_by_tenor, cu_df_by_tenor,
            cu_da, cu_dt, Int32(length(div_amounts)),
            Float32(d_sigma),
            N_t, N_s
        )

        CUDA.synchronize()
        Array(out)
    end

    function _delta_kernel!(
    out,
    spots, strikes, tenors, sigmas, rights,
    tenor_ids, pv_divs_arr,
    r_by_tenor, df_by_tenor,
    da, dtimes, n_divs::Int32,
    d_spot::Float32,
    N_t::Int32, N_s::Int32
    )
        idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        idx > Int32(length(spots)) && return nothing

        S, K, T, sigma = spots[idx], strikes[idx], tenors[idx], sigmas[idx]
        is_call = rights[idx] != 0x00

        if !isfinite(S) || !isfinite(K) || !isfinite(T) || !isfinite(sigma)
            out[idx] = NaN32
            return nothing
        end

        if T <= 0f0 || S <= 0f0 || K <= 0f0 || sigma <= 0f0
            out[idx] = 0f0
            return nothing
        end

        tid = tenor_ids[idx]
        pv_divs = pv_divs_arr[tid]

        r_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)
        df_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)
        r_view = view(r_by_tenor, r_start:r_start + RATE_SCHED_STRIDE - Int32(1))
        df_view = view(df_by_tenor, df_start:df_start + RATE_SCHED_STRIDE - Int32(1))

        S_up = S + d_spot
        S_down = max(S - d_spot, 1f-6)

        p_up = _d_fd_price!(S_up, K, T, sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        p_down = _d_fd_price!(S_down, K, T, sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)

        out[idx] = (p_up - p_down) / (S_up - S_down)

        return nothing
    end

    function _v_delta_gpu(
    spots, strikes, tenors, sigmas, rights,
    rates_curve, rates_times, div_amounts, div_times,
    d_spot, time_steps, space_steps
    )
        n = length(spots)
        N_t = Int32(clamp(time_steps, 10, 256))
        N_s = Int32(clamp(space_steps, 10, 256))

        tenor_ids, unique_tenors = _build_tenor_map(tenors)
        n_unique = length(unique_tenors)

        pv_by_tenor = Vector{Float32}(undef, n_unique)
        for tid in 1:n_unique
            T = unique_tenors[tid]
            acc = 0f0
            for j in eachindex(div_amounts)
                td = div_times[j]
                if 0f0 < td <= T
                    acc += div_amounts[j] * df_0_t(td, rates_curve, rates_times)
                end
            end
            pv_by_tenor[tid] = acc
        end

        cu_r_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_df_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_unique_tenors = CuArray(unique_tenors)
        cu_rates_curve = CuArray(rates_curve)
        cu_rates_times = CuArray(rates_times)

        threads_2d = (min(n_unique, 32), min(Int(RATE_SCHED_STRIDE), 32))
        blocks_2d = (cld(n_unique, threads_2d[1]), cld(Int(RATE_SCHED_STRIDE), threads_2d[2]))

        @cuda threads = threads_2d blocks = blocks_2d _build_rate_sched_kernel!(
            cu_unique_tenors, Int32(n_unique),
            cu_rates_curve, cu_rates_times,
            N_t,
            cu_r_by_tenor, cu_df_by_tenor
        )

        cu_spots = CuArray(spots)
        cu_strikes = CuArray(strikes)
        cu_tenors = CuArray(tenors)
        cu_sigmas = CuArray(sigmas)
        cu_rights = CuArray(rights)
        cu_tenor_ids = CuArray(tenor_ids)
        cu_pv = CuArray(pv_by_tenor)
        cu_da = CuArray(div_amounts)
        cu_dt = CuArray(div_times)
        out = CUDA.fill(NaN32, n)

        nthreads = 256
        nblocks = cld(n, nthreads)

        @cuda threads = nthreads blocks = nblocks _delta_kernel!(
            out,
            cu_spots, cu_strikes, cu_tenors, cu_sigmas, cu_rights,
            cu_tenor_ids, cu_pv,
            cu_r_by_tenor, cu_df_by_tenor,
            cu_da, cu_dt, Int32(length(div_amounts)),
            Float32(d_spot),
            N_t, N_s
        )

        CUDA.synchronize()
        Array(out)
    end # end _v_delta_gpu

    # ────────────────────────────────────────────────────────────────────────────
    # Theta: dP/dt  (time decay, positive = value lost per unit time)
    # Uses one-sided backward difference: [P(T) - P(T - dt)] / dt
    # ────────────────────────────────────────────────────────────────────────────

    function _theta_kernel!(
        out,
        spots, strikes, tenors, sigmas, rights,
        tenor_ids, pv_divs_arr,
        r_by_tenor, df_by_tenor,
        da, dtimes, n_divs::Int32,
        d_time::Float32,
        N_t::Int32, N_s::Int32
    )
        idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        idx > Int32(length(spots)) && return nothing

        S, K, T, sigma = spots[idx], strikes[idx], tenors[idx], sigmas[idx]
        is_call = rights[idx] != 0x00

        if !isfinite(S) || !isfinite(K) || !isfinite(T) || !isfinite(sigma)
            out[idx] = NaN32
            return nothing
        end

        if T <= 0f0 || S <= 0f0 || K <= 0f0 || sigma <= 0f0
            out[idx] = 0f0
            return nothing
        end

        tid = tenor_ids[idx]
        pv_divs = pv_divs_arr[tid]

        r_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)
        r_view = view(r_by_tenor, r_start:r_start + RATE_SCHED_STRIDE - Int32(1))
        df_view = view(df_by_tenor, r_start:r_start + RATE_SCHED_STRIDE - Int32(1))

        T_down = max(T - d_time, 1f-6)

        p_now  = _d_fd_price!(S, K, T,      sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        p_down = _d_fd_price!(S, K, T_down, sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)

        # Theta is conventionally quoted as value change per calendar day (negative)
        out[idx] = (p_down - p_now) / (T - T_down)

        return nothing
    end

    function _v_theta_gpu(
        spots, strikes, tenors, sigmas, rights,
        rates_curve, rates_times, div_amounts, div_times,
        d_time, time_steps, space_steps
    )
        n = length(spots)
        N_t = Int32(clamp(time_steps, 10, 256))
        N_s = Int32(clamp(space_steps, 10, 256))

        tenor_ids, unique_tenors = _build_tenor_map(tenors)
        n_unique = length(unique_tenors)

        pv_by_tenor = Vector{Float32}(undef, n_unique)
        for tid in 1:n_unique
            T = unique_tenors[tid]
            acc = 0f0
            for j in eachindex(div_amounts)
                td = div_times[j]
                if 0f0 < td <= T
                    acc += div_amounts[j] * df_0_t(td, rates_curve, rates_times)
                end
            end
            pv_by_tenor[tid] = acc
        end

        cu_r_by_tenor  = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_df_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_unique_tenors = CuArray(unique_tenors)
        cu_rates_curve   = CuArray(rates_curve)
        cu_rates_times   = CuArray(rates_times)

        threads_2d = (min(n_unique, 32), min(Int(RATE_SCHED_STRIDE), 32))
        blocks_2d  = (cld(n_unique, threads_2d[1]), cld(Int(RATE_SCHED_STRIDE), threads_2d[2]))

        @cuda threads=threads_2d blocks=blocks_2d _build_rate_sched_kernel!(
            cu_unique_tenors, Int32(n_unique),
            cu_rates_curve, cu_rates_times, N_t,
            cu_r_by_tenor, cu_df_by_tenor)

        cu_spots    = CuArray(spots);   cu_strikes = CuArray(strikes)
        cu_tenors   = CuArray(tenors);  cu_sigmas  = CuArray(sigmas)
        cu_rights   = CuArray(rights);  cu_tenor_ids = CuArray(tenor_ids)
        cu_pv = CuArray(pv_by_tenor)
        cu_da = CuArray(div_amounts);   cu_dt = CuArray(div_times)
        out = CUDA.fill(NaN32, n)

        nthreads = 256
        @cuda threads=nthreads blocks=cld(n, nthreads) _theta_kernel!(
            out,
            cu_spots, cu_strikes, cu_tenors, cu_sigmas, cu_rights,
            cu_tenor_ids, cu_pv,
            cu_r_by_tenor, cu_df_by_tenor,
            cu_da, cu_dt, Int32(length(div_amounts)),
            Float32(d_time), N_t, N_s)

        CUDA.synchronize()
        Array(out)
    end

    # ────────────────────────────────────────────────────────────────────────────
    # Gamma: d²P/dS²   (central second difference)
    # ────────────────────────────────────────────────────────────────────────────

    function _gamma_kernel!(
        out,
        spots, strikes, tenors, sigmas, rights,
        tenor_ids, pv_divs_arr,
        r_by_tenor, df_by_tenor,
        da, dtimes, n_divs::Int32,
        d_spot::Float32,
        N_t::Int32, N_s::Int32
    )
        idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        idx > Int32(length(spots)) && return nothing

        S, K, T, sigma = spots[idx], strikes[idx], tenors[idx], sigmas[idx]
        is_call = rights[idx] != 0x00

        if !isfinite(S) || !isfinite(K) || !isfinite(T) || !isfinite(sigma)
            out[idx] = NaN32
            return nothing
        end
        if T <= 0f0 || S <= 0f0 || K <= 0f0 || sigma <= 0f0
            out[idx] = 0f0
            return nothing
        end

        tid = tenor_ids[idx]
        pv_divs = pv_divs_arr[tid]
        r_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)
        r_view  = view(r_by_tenor,  r_start:r_start + RATE_SCHED_STRIDE - Int32(1))
        df_view = view(df_by_tenor, r_start:r_start + RATE_SCHED_STRIDE - Int32(1))

        S_up   = S + d_spot
        S_down = max(S - d_spot, 1f-6)

        p_up   = _d_fd_price!(S_up,   K, T, sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        p_mid  = _d_fd_price!(S,      K, T, sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        p_down = _d_fd_price!(S_down, K, T, sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)

        h = S_up - S_down   # = 2 * d_spot (or less near zero)
        out[idx] = (p_up - 2f0 * p_mid + p_down) / (0.5f0 * h * 0.5f0 * h)

        return nothing
    end

    function _v_gamma_gpu(
        spots, strikes, tenors, sigmas, rights,
        rates_curve, rates_times, div_amounts, div_times,
        d_spot, time_steps, space_steps
    )
        n = length(spots)
        N_t = Int32(clamp(time_steps, 10, 256))
        N_s = Int32(clamp(space_steps, 10, 256))

        tenor_ids, unique_tenors = _build_tenor_map(tenors)
        n_unique = length(unique_tenors)

        pv_by_tenor = Vector{Float32}(undef, n_unique)
        for tid in 1:n_unique
            T = unique_tenors[tid]
            acc = 0f0
            for j in eachindex(div_amounts)
                td = div_times[j]
                if 0f0 < td <= T
                    acc += div_amounts[j] * df_0_t(td, rates_curve, rates_times)
                end
            end
            pv_by_tenor[tid] = acc
        end

        cu_r_by_tenor  = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_df_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_unique_tenors = CuArray(unique_tenors)
        cu_rates_curve   = CuArray(rates_curve)
        cu_rates_times   = CuArray(rates_times)

        threads_2d = (min(n_unique, 32), min(Int(RATE_SCHED_STRIDE), 32))
        blocks_2d  = (cld(n_unique, threads_2d[1]), cld(Int(RATE_SCHED_STRIDE), threads_2d[2]))

        @cuda threads=threads_2d blocks=blocks_2d _build_rate_sched_kernel!(
            cu_unique_tenors, Int32(n_unique),
            cu_rates_curve, cu_rates_times, N_t,
            cu_r_by_tenor, cu_df_by_tenor)

        cu_spots = CuArray(spots); cu_strikes = CuArray(strikes)
        cu_tenors = CuArray(tenors); cu_sigmas = CuArray(sigmas)
        cu_rights = CuArray(rights); cu_tenor_ids = CuArray(tenor_ids)
        cu_pv = CuArray(pv_by_tenor)
        cu_da = CuArray(div_amounts); cu_dt = CuArray(div_times)
        out = CUDA.fill(NaN32, n)

        nthreads = 256
        @cuda threads=nthreads blocks=cld(n, nthreads) _gamma_kernel!(
            out,
            cu_spots, cu_strikes, cu_tenors, cu_sigmas, cu_rights,
            cu_tenor_ids, cu_pv,
            cu_r_by_tenor, cu_df_by_tenor,
            cu_da, cu_dt, Int32(length(div_amounts)),
            Float32(d_spot), N_t, N_s)

        CUDA.synchronize()
        Array(out)
    end

    # ────────────────────────────────────────────────────────────────────────────
    # Speed: dΓ/dS  (third derivative w.r.t. spot)
    # Uses the 5-point stencil: (-P(2h) + 2P(h) - 2P(-h) + P(-2h)) / (2h³)
    # ────────────────────────────────────────────────────────────────────────────

    function _speed_kernel!(
        out,
        spots, strikes, tenors, sigmas, rights,
        tenor_ids, pv_divs_arr,
        r_by_tenor, df_by_tenor,
        da, dtimes, n_divs::Int32,
        d_spot::Float32,
        N_t::Int32, N_s::Int32
    )
        idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        idx > Int32(length(spots)) && return nothing

        S, K, T, sigma = spots[idx], strikes[idx], tenors[idx], sigmas[idx]
        is_call = rights[idx] != 0x00

        if !isfinite(S) || !isfinite(K) || !isfinite(T) || !isfinite(sigma)
            out[idx] = NaN32
            return nothing
        end
        if T <= 0f0 || S <= 0f0 || K <= 0f0 || sigma <= 0f0
            out[idx] = 0f0
            return nothing
        end

        tid = tenor_ids[idx]
        pv_divs = pv_divs_arr[tid]
        r_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)
        r_view  = view(r_by_tenor,  r_start:r_start + RATE_SCHED_STRIDE - Int32(1))
        df_view = view(df_by_tenor, r_start:r_start + RATE_SCHED_STRIDE - Int32(1))

        h  = d_spot
        S_p2 = S + 2f0 * h
        S_p1 = S + h
        S_m1 = max(S - h,       1f-6)
        S_m2 = max(S - 2f0 * h, 1f-6)

        p_p2 = _d_fd_price!(S_p2, K, T, sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        p_p1 = _d_fd_price!(S_p1, K, T, sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        p_m1 = _d_fd_price!(S_m1, K, T, sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        p_m2 = _d_fd_price!(S_m2, K, T, sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)

        out[idx] = (-p_p2 + 2f0 * p_p1 - 2f0 * p_m1 + p_m2) / (2f0 * h * h * h)

        return nothing
    end

    function _v_speed_gpu(
        spots, strikes, tenors, sigmas, rights,
        rates_curve, rates_times, div_amounts, div_times,
        d_spot, time_steps, space_steps
    )
        n = length(spots)
        N_t = Int32(clamp(time_steps, 10, 256))
        N_s = Int32(clamp(space_steps, 10, 256))

        tenor_ids, unique_tenors = _build_tenor_map(tenors)
        n_unique = length(unique_tenors)

        pv_by_tenor = Vector{Float32}(undef, n_unique)
        for tid in 1:n_unique
            T = unique_tenors[tid]
            acc = 0f0
            for j in eachindex(div_amounts)
                td = div_times[j]
                if 0f0 < td <= T
                    acc += div_amounts[j] * df_0_t(td, rates_curve, rates_times)
                end
            end
            pv_by_tenor[tid] = acc
        end

        cu_r_by_tenor  = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_df_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_unique_tenors = CuArray(unique_tenors)
        cu_rates_curve   = CuArray(rates_curve)
        cu_rates_times   = CuArray(rates_times)

        threads_2d = (min(n_unique, 32), min(Int(RATE_SCHED_STRIDE), 32))
        blocks_2d  = (cld(n_unique, threads_2d[1]), cld(Int(RATE_SCHED_STRIDE), threads_2d[2]))

        @cuda threads=threads_2d blocks=blocks_2d _build_rate_sched_kernel!(
            cu_unique_tenors, Int32(n_unique),
            cu_rates_curve, cu_rates_times, N_t,
            cu_r_by_tenor, cu_df_by_tenor)

        cu_spots = CuArray(spots); cu_strikes = CuArray(strikes)
        cu_tenors = CuArray(tenors); cu_sigmas = CuArray(sigmas)
        cu_rights = CuArray(rights); cu_tenor_ids = CuArray(tenor_ids)
        cu_pv = CuArray(pv_by_tenor)
        cu_da = CuArray(div_amounts); cu_dt = CuArray(div_times)
        out = CUDA.fill(NaN32, n)

        nthreads = 256
        @cuda threads=nthreads blocks=cld(n, nthreads) _speed_kernel!(
            out,
            cu_spots, cu_strikes, cu_tenors, cu_sigmas, cu_rights,
            cu_tenor_ids, cu_pv,
            cu_r_by_tenor, cu_df_by_tenor,
            cu_da, cu_dt, Int32(length(div_amounts)),
            Float32(d_spot), N_t, N_s)

        CUDA.synchronize()
        Array(out)
    end

    # ────────────────────────────────────────────────────────────────────────────
    # GammaDecay: dΓ/dt   (how gamma changes with time)
    # Central difference of the second spot derivative over a time bump
    # ────────────────────────────────────────────────────────────────────────────

    @inline function _gamma_at(S, K, T, sigma, is_call, pv_divs, r_view, df_view,
                               da, dtimes, n_divs, h, N_t, N_s)::Float32
        S_up   = S + h
        S_down = max(S - h, 1f-6)
        p_up   = _d_fd_price!(S_up,   K, T, sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        p_mid  = _d_fd_price!(S,      K, T, sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        p_down = _d_fd_price!(S_down, K, T, sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        (p_up - 2f0 * p_mid + p_down) / (h * h)
    end

    function _gamma_decay_kernel!(
        out,
        spots, strikes, tenors, sigmas, rights,
        tenor_ids, pv_divs_arr,
        r_by_tenor, df_by_tenor,
        da, dtimes, n_divs::Int32,
        d_spot::Float32, d_time::Float32,
        N_t::Int32, N_s::Int32
    )
        idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        idx > Int32(length(spots)) && return nothing

        S, K, T, sigma = spots[idx], strikes[idx], tenors[idx], sigmas[idx]
        is_call = rights[idx] != 0x00

        if !isfinite(S) || !isfinite(K) || !isfinite(T) || !isfinite(sigma)
            out[idx] = NaN32
            return nothing
        end
        if T <= 0f0 || S <= 0f0 || K <= 0f0 || sigma <= 0f0
            out[idx] = 0f0
            return nothing
        end

        tid = tenor_ids[idx]
        pv_divs = pv_divs_arr[tid]
        r_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)
        r_view  = view(r_by_tenor,  r_start:r_start + RATE_SCHED_STRIDE - Int32(1))
        df_view = view(df_by_tenor, r_start:r_start + RATE_SCHED_STRIDE - Int32(1))

        T_up   = T + d_time
        T_down = max(T - d_time, 1f-6)

        g_up   = _gamma_at(S, K, T_up,   sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, d_spot, N_t, N_s)
        g_down = _gamma_at(S, K, T_down, sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, d_spot, N_t, N_s)

        # Sign convention: decay = value at T_down - T_up (shorter tenor has less gamma time)
        out[idx] = (g_down - g_up) / (T_up - T_down)

        return nothing
    end

    function _v_gamma_decay_gpu(
        spots, strikes, tenors, sigmas, rights,
        rates_curve, rates_times, div_amounts, div_times,
        d_spot, d_time, time_steps, space_steps
    )
        n = length(spots)
        N_t = Int32(clamp(time_steps, 10, 256))
        N_s = Int32(clamp(space_steps, 10, 256))

        tenor_ids, unique_tenors = _build_tenor_map(tenors)
        n_unique = length(unique_tenors)

        pv_by_tenor = Vector{Float32}(undef, n_unique)
        for tid in 1:n_unique
            T = unique_tenors[tid]
            acc = 0f0
            for j in eachindex(div_amounts)
                td = div_times[j]
                if 0f0 < td <= T
                    acc += div_amounts[j] * df_0_t(td, rates_curve, rates_times)
                end
            end
            pv_by_tenor[tid] = acc
        end

        cu_r_by_tenor  = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_df_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_unique_tenors = CuArray(unique_tenors)
        cu_rates_curve   = CuArray(rates_curve)
        cu_rates_times   = CuArray(rates_times)

        threads_2d = (min(n_unique, 32), min(Int(RATE_SCHED_STRIDE), 32))
        blocks_2d  = (cld(n_unique, threads_2d[1]), cld(Int(RATE_SCHED_STRIDE), threads_2d[2]))

        @cuda threads=threads_2d blocks=blocks_2d _build_rate_sched_kernel!(
            cu_unique_tenors, Int32(n_unique),
            cu_rates_curve, cu_rates_times, N_t,
            cu_r_by_tenor, cu_df_by_tenor)

        cu_spots = CuArray(spots); cu_strikes = CuArray(strikes)
        cu_tenors = CuArray(tenors); cu_sigmas = CuArray(sigmas)
        cu_rights = CuArray(rights); cu_tenor_ids = CuArray(tenor_ids)
        cu_pv = CuArray(pv_by_tenor)
        cu_da = CuArray(div_amounts); cu_dt = CuArray(div_times)
        out = CUDA.fill(NaN32, n)

        nthreads = 256
        @cuda threads=nthreads blocks=cld(n, nthreads) _gamma_decay_kernel!(
            out,
            cu_spots, cu_strikes, cu_tenors, cu_sigmas, cu_rights,
            cu_tenor_ids, cu_pv,
            cu_r_by_tenor, cu_df_by_tenor,
            cu_da, cu_dt, Int32(length(div_amounts)),
            Float32(d_spot), Float32(d_time), N_t, N_s)

        CUDA.synchronize()
        Array(out)
    end

    # ────────────────────────────────────────────────────────────────────────────
    # GammaVol: dΓ/dσ   (how gamma changes with implied vol)
    # Central difference of the second spot derivative over a vol bump
    # ────────────────────────────────────────────────────────────────────────────

    function _gamma_vol_kernel!(
        out,
        spots, strikes, tenors, sigmas, rights,
        tenor_ids, pv_divs_arr,
        r_by_tenor, df_by_tenor,
        da, dtimes, n_divs::Int32,
        d_spot::Float32, d_sigma::Float32,
        N_t::Int32, N_s::Int32
    )
        idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        idx > Int32(length(spots)) && return nothing

        S, K, T, sigma = spots[idx], strikes[idx], tenors[idx], sigmas[idx]
        is_call = rights[idx] != 0x00

        if !isfinite(S) || !isfinite(K) || !isfinite(T) || !isfinite(sigma)
            out[idx] = NaN32
            return nothing
        end
        if T <= 0f0 || S <= 0f0 || K <= 0f0 || sigma <= 0f0
            out[idx] = 0f0
            return nothing
        end

        tid = tenor_ids[idx]
        pv_divs = pv_divs_arr[tid]
        r_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)
        r_view  = view(r_by_tenor,  r_start:r_start + RATE_SCHED_STRIDE - Int32(1))
        df_view = view(df_by_tenor, r_start:r_start + RATE_SCHED_STRIDE - Int32(1))

        sig_up   = sigma + d_sigma
        sig_down = max(sigma - d_sigma, 1f-6)

        g_up   = _gamma_at(S, K, T, sig_up,   is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, d_spot, N_t, N_s)
        g_down = _gamma_at(S, K, T, sig_down, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, d_spot, N_t, N_s)

        out[idx] = (g_up - g_down) / (sig_up - sig_down)

        return nothing
    end

    function _v_gamma_vol_gpu(
        spots, strikes, tenors, sigmas, rights,
        rates_curve, rates_times, div_amounts, div_times,
        d_spot, d_sigma, time_steps, space_steps
    )
        n = length(spots)
        N_t = Int32(clamp(time_steps, 10, 256))
        N_s = Int32(clamp(space_steps, 10, 256))

        tenor_ids, unique_tenors = _build_tenor_map(tenors)
        n_unique = length(unique_tenors)

        pv_by_tenor = Vector{Float32}(undef, n_unique)
        for tid in 1:n_unique
            T = unique_tenors[tid]
            acc = 0f0
            for j in eachindex(div_amounts)
                td = div_times[j]
                if 0f0 < td <= T
                    acc += div_amounts[j] * df_0_t(td, rates_curve, rates_times)
                end
            end
            pv_by_tenor[tid] = acc
        end

        cu_r_by_tenor  = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_df_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_unique_tenors = CuArray(unique_tenors)
        cu_rates_curve   = CuArray(rates_curve)
        cu_rates_times   = CuArray(rates_times)

        threads_2d = (min(n_unique, 32), min(Int(RATE_SCHED_STRIDE), 32))
        blocks_2d  = (cld(n_unique, threads_2d[1]), cld(Int(RATE_SCHED_STRIDE), threads_2d[2]))

        @cuda threads=threads_2d blocks=blocks_2d _build_rate_sched_kernel!(
            cu_unique_tenors, Int32(n_unique),
            cu_rates_curve, cu_rates_times, N_t,
            cu_r_by_tenor, cu_df_by_tenor)

        cu_spots = CuArray(spots); cu_strikes = CuArray(strikes)
        cu_tenors = CuArray(tenors); cu_sigmas = CuArray(sigmas)
        cu_rights = CuArray(rights); cu_tenor_ids = CuArray(tenor_ids)
        cu_pv = CuArray(pv_by_tenor)
        cu_da = CuArray(div_amounts); cu_dt = CuArray(div_times)
        out = CUDA.fill(NaN32, n)

        nthreads = 256
        @cuda threads=nthreads blocks=cld(n, nthreads) _gamma_vol_kernel!(
            out,
            cu_spots, cu_strikes, cu_tenors, cu_sigmas, cu_rights,
            cu_tenor_ids, cu_pv,
            cu_r_by_tenor, cu_df_by_tenor,
            cu_da, cu_dt, Int32(length(div_amounts)),
            Float32(d_spot), Float32(d_sigma), N_t, N_s)

        CUDA.synchronize()
        Array(out)
    end

    # ────────────────────────────────────────────────────────────────────────────
    # ThetaDecay: dΘ/dt  (second-order time derivative d²P/dt²)
    # Central second difference over time
    # ────────────────────────────────────────────────────────────────────────────

    function _theta_decay_kernel!(
        out,
        spots, strikes, tenors, sigmas, rights,
        tenor_ids, pv_divs_arr,
        r_by_tenor, df_by_tenor,
        da, dtimes, n_divs::Int32,
        d_time::Float32,
        N_t::Int32, N_s::Int32
    )
        idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        idx > Int32(length(spots)) && return nothing

        S, K, T, sigma = spots[idx], strikes[idx], tenors[idx], sigmas[idx]
        is_call = rights[idx] != 0x00

        if !isfinite(S) || !isfinite(K) || !isfinite(T) || !isfinite(sigma)
            out[idx] = NaN32
            return nothing
        end
        if T <= 0f0 || S <= 0f0 || K <= 0f0 || sigma <= 0f0
            out[idx] = 0f0
            return nothing
        end

        tid = tenor_ids[idx]
        pv_divs = pv_divs_arr[tid]
        r_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)
        r_view  = view(r_by_tenor,  r_start:r_start + RATE_SCHED_STRIDE - Int32(1))
        df_view = view(df_by_tenor, r_start:r_start + RATE_SCHED_STRIDE - Int32(1))

        T_up   = T + d_time
        T_down = max(T - d_time, 1f-6)

        p_up   = _d_fd_price!(S, K, T_up,   sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        p_mid  = _d_fd_price!(S, K, T,      sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        p_down = _d_fd_price!(S, K, T_down, sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)

        # d²P/dt²: positive when theta itself accelerates as we approach expiry
        # Note: here T is time-to-expiry so dT > 0 means further from expiry
        out[idx] = (p_up - 2f0 * p_mid + p_down) / (d_time * d_time)

        return nothing
    end

    function _v_theta_decay_gpu(
        spots, strikes, tenors, sigmas, rights,
        rates_curve, rates_times, div_amounts, div_times,
        d_time, time_steps, space_steps
    )
        n = length(spots)
        N_t = Int32(clamp(time_steps, 10, 256))
        N_s = Int32(clamp(space_steps, 10, 256))

        tenor_ids, unique_tenors = _build_tenor_map(tenors)
        n_unique = length(unique_tenors)

        pv_by_tenor = Vector{Float32}(undef, n_unique)
        for tid in 1:n_unique
            T = unique_tenors[tid]
            acc = 0f0
            for j in eachindex(div_amounts)
                td = div_times[j]
                if 0f0 < td <= T
                    acc += div_amounts[j] * df_0_t(td, rates_curve, rates_times)
                end
            end
            pv_by_tenor[tid] = acc
        end

        cu_r_by_tenor  = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_df_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_unique_tenors = CuArray(unique_tenors)
        cu_rates_curve   = CuArray(rates_curve)
        cu_rates_times   = CuArray(rates_times)

        threads_2d = (min(n_unique, 32), min(Int(RATE_SCHED_STRIDE), 32))
        blocks_2d  = (cld(n_unique, threads_2d[1]), cld(Int(RATE_SCHED_STRIDE), threads_2d[2]))

        @cuda threads=threads_2d blocks=blocks_2d _build_rate_sched_kernel!(
            cu_unique_tenors, Int32(n_unique),
            cu_rates_curve, cu_rates_times, N_t,
            cu_r_by_tenor, cu_df_by_tenor)

        cu_spots = CuArray(spots); cu_strikes = CuArray(strikes)
        cu_tenors = CuArray(tenors); cu_sigmas = CuArray(sigmas)
        cu_rights = CuArray(rights); cu_tenor_ids = CuArray(tenor_ids)
        cu_pv = CuArray(pv_by_tenor)
        cu_da = CuArray(div_amounts); cu_dt = CuArray(div_times)
        out = CUDA.fill(NaN32, n)

        nthreads = 256
        @cuda threads=nthreads blocks=cld(n, nthreads) _theta_decay_kernel!(
            out,
            cu_spots, cu_strikes, cu_tenors, cu_sigmas, cu_rights,
            cu_tenor_ids, cu_pv,
            cu_r_by_tenor, cu_df_by_tenor,
            cu_da, cu_dt, Int32(length(div_amounts)),
            Float32(d_time), N_t, N_s)

        CUDA.synchronize()
        Array(out)
    end

    # ────────────────────────────────────────────────────────────────────────────
    # VegaDecay: dVega/dt  (how vega changes with time to expiry)
    # Central difference of the vol first derivative over a time bump
    # ────────────────────────────────────────────────────────────────────────────

    @inline function _vega_at(S, K, T, sigma, is_call, pv_divs, r_view, df_view,
                              da, dtimes, n_divs, dv, N_t, N_s)::Float32
        sig_up   = sigma + dv
        sig_down = max(sigma - dv, 1f-6)
        p_up   = _d_fd_price!(S, K, T, sig_up,   is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        p_down = _d_fd_price!(S, K, T, sig_down, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        (p_up - p_down) / (sig_up - sig_down)
    end

    function _vega_decay_kernel!(
        out,
        spots, strikes, tenors, sigmas, rights,
        tenor_ids, pv_divs_arr,
        r_by_tenor, df_by_tenor,
        da, dtimes, n_divs::Int32,
        d_sigma::Float32, d_time::Float32,
        N_t::Int32, N_s::Int32
    )
        idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        idx > Int32(length(spots)) && return nothing

        S, K, T, sigma = spots[idx], strikes[idx], tenors[idx], sigmas[idx]
        is_call = rights[idx] != 0x00

        if !isfinite(S) || !isfinite(K) || !isfinite(T) || !isfinite(sigma)
            out[idx] = NaN32
            return nothing
        end
        if T <= 0f0 || S <= 0f0 || K <= 0f0 || sigma <= 0f0
            out[idx] = 0f0
            return nothing
        end

        tid = tenor_ids[idx]
        pv_divs = pv_divs_arr[tid]
        r_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)
        r_view  = view(r_by_tenor,  r_start:r_start + RATE_SCHED_STRIDE - Int32(1))
        df_view = view(df_by_tenor, r_start:r_start + RATE_SCHED_STRIDE - Int32(1))

        T_up   = T + d_time
        T_down = max(T - d_time, 1f-6)

        vega_up   = _vega_at(S, K, T_up,   sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, d_sigma, N_t, N_s)
        vega_down = _vega_at(S, K, T_down, sigma, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, d_sigma, N_t, N_s)

        # Decay: shorter tenor → less vega, so we report the rate of vega loss as T decreases
        out[idx] = (vega_down - vega_up) / (T_up - T_down)

        return nothing
    end

    function _v_vega_decay_gpu(
        spots, strikes, tenors, sigmas, rights,
        rates_curve, rates_times, div_amounts, div_times,
        d_sigma, d_time, time_steps, space_steps
    )
        n = length(spots)
        N_t = Int32(clamp(time_steps, 10, 256))
        N_s = Int32(clamp(space_steps, 10, 256))

        tenor_ids, unique_tenors = _build_tenor_map(tenors)
        n_unique = length(unique_tenors)

        pv_by_tenor = Vector{Float32}(undef, n_unique)
        for tid in 1:n_unique
            T = unique_tenors[tid]
            acc = 0f0
            for j in eachindex(div_amounts)
                td = div_times[j]
                if 0f0 < td <= T
                    acc += div_amounts[j] * df_0_t(td, rates_curve, rates_times)
                end
            end
            pv_by_tenor[tid] = acc
        end

        cu_r_by_tenor  = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_df_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_unique_tenors = CuArray(unique_tenors)
        cu_rates_curve   = CuArray(rates_curve)
        cu_rates_times   = CuArray(rates_times)

        threads_2d = (min(n_unique, 32), min(Int(RATE_SCHED_STRIDE), 32))
        blocks_2d  = (cld(n_unique, threads_2d[1]), cld(Int(RATE_SCHED_STRIDE), threads_2d[2]))

        @cuda threads=threads_2d blocks=blocks_2d _build_rate_sched_kernel!(
            cu_unique_tenors, Int32(n_unique),
            cu_rates_curve, cu_rates_times, N_t,
            cu_r_by_tenor, cu_df_by_tenor)

        cu_spots = CuArray(spots); cu_strikes = CuArray(strikes)
        cu_tenors = CuArray(tenors); cu_sigmas = CuArray(sigmas)
        cu_rights = CuArray(rights); cu_tenor_ids = CuArray(tenor_ids)
        cu_pv = CuArray(pv_by_tenor)
        cu_da = CuArray(div_amounts); cu_dt = CuArray(div_times)
        out = CUDA.fill(NaN32, n)

        nthreads = 256
        @cuda threads=nthreads blocks=cld(n, nthreads) _vega_decay_kernel!(
            out,
            cu_spots, cu_strikes, cu_tenors, cu_sigmas, cu_rights,
            cu_tenor_ids, cu_pv,
            cu_r_by_tenor, cu_df_by_tenor,
            cu_da, cu_dt, Int32(length(div_amounts)),
            Float32(d_sigma), Float32(d_time), N_t, N_s)

        CUDA.synchronize()
        Array(out)
    end

    # ────────────────────────────────────────────────────────────────────────────
    # Vanna: dΔ/dσ = dVega/dS  (cross partial spot × vol)
    # 2D cross finite difference:
    #   [P(S+h,σ+v) - P(S+h,σ-v) - P(S-h,σ+v) + P(S-h,σ-v)] / (4hv)
    # ────────────────────────────────────────────────────────────────────────────

    function _vanna_kernel!(
        out,
        spots, strikes, tenors, sigmas, rights,
        tenor_ids, pv_divs_arr,
        r_by_tenor, df_by_tenor,
        da, dtimes, n_divs::Int32,
        d_spot::Float32, d_sigma::Float32,
        N_t::Int32, N_s::Int32
    )
        idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        idx > Int32(length(spots)) && return nothing

        S, K, T, sigma = spots[idx], strikes[idx], tenors[idx], sigmas[idx]
        is_call = rights[idx] != 0x00

        if !isfinite(S) || !isfinite(K) || !isfinite(T) || !isfinite(sigma)
            out[idx] = NaN32
            return nothing
        end
        if T <= 0f0 || S <= 0f0 || K <= 0f0 || sigma <= 0f0
            out[idx] = 0f0
            return nothing
        end

        tid = tenor_ids[idx]
        pv_divs = pv_divs_arr[tid]
        r_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)
        r_view  = view(r_by_tenor,  r_start:r_start + RATE_SCHED_STRIDE - Int32(1))
        df_view = view(df_by_tenor, r_start:r_start + RATE_SCHED_STRIDE - Int32(1))

        S_up   = S + d_spot
        S_down = max(S - d_spot, 1f-6)
        sig_up   = sigma + d_sigma
        sig_down = max(sigma - d_sigma, 1f-6)

        pp = _d_fd_price!(S_up,   K, T, sig_up,   is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        pm = _d_fd_price!(S_up,   K, T, sig_down, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        mp = _d_fd_price!(S_down, K, T, sig_up,   is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        mm = _d_fd_price!(S_down, K, T, sig_down, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)

        dS  = S_up   - S_down
        dsig = sig_up - sig_down
        out[idx] = (pp - pm - mp + mm) / (dS * dsig)

        return nothing
    end

    function _v_vanna_gpu(
        spots, strikes, tenors, sigmas, rights,
        rates_curve, rates_times, div_amounts, div_times,
        d_spot, d_sigma, time_steps, space_steps
    )
        n = length(spots)
        N_t = Int32(clamp(time_steps, 10, 256))
        N_s = Int32(clamp(space_steps, 10, 256))

        tenor_ids, unique_tenors = _build_tenor_map(tenors)
        n_unique = length(unique_tenors)

        pv_by_tenor = Vector{Float32}(undef, n_unique)
        for tid in 1:n_unique
            T = unique_tenors[tid]
            acc = 0f0
            for j in eachindex(div_amounts)
                td = div_times[j]
                if 0f0 < td <= T
                    acc += div_amounts[j] * df_0_t(td, rates_curve, rates_times)
                end
            end
            pv_by_tenor[tid] = acc
        end

        cu_r_by_tenor  = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_df_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_unique_tenors = CuArray(unique_tenors)
        cu_rates_curve   = CuArray(rates_curve)
        cu_rates_times   = CuArray(rates_times)

        threads_2d = (min(n_unique, 32), min(Int(RATE_SCHED_STRIDE), 32))
        blocks_2d  = (cld(n_unique, threads_2d[1]), cld(Int(RATE_SCHED_STRIDE), threads_2d[2]))

        @cuda threads=threads_2d blocks=blocks_2d _build_rate_sched_kernel!(
            cu_unique_tenors, Int32(n_unique),
            cu_rates_curve, cu_rates_times, N_t,
            cu_r_by_tenor, cu_df_by_tenor)

        cu_spots = CuArray(spots); cu_strikes = CuArray(strikes)
        cu_tenors = CuArray(tenors); cu_sigmas = CuArray(sigmas)
        cu_rights = CuArray(rights); cu_tenor_ids = CuArray(tenor_ids)
        cu_pv = CuArray(pv_by_tenor)
        cu_da = CuArray(div_amounts); cu_dt = CuArray(div_times)
        out = CUDA.fill(NaN32, n)

        nthreads = 256
        @cuda threads=nthreads blocks=cld(n, nthreads) _vanna_kernel!(
            out,
            cu_spots, cu_strikes, cu_tenors, cu_sigmas, cu_rights,
            cu_tenor_ids, cu_pv,
            cu_r_by_tenor, cu_df_by_tenor,
            cu_da, cu_dt, Int32(length(div_amounts)),
            Float32(d_spot), Float32(d_sigma), N_t, N_s)

        CUDA.synchronize()
        Array(out)
    end

    # ────────────────────────────────────────────────────────────────────────────
    # Volga: d²P/dσ²  (also called Vomma)
    # Central second difference over vol: (P(σ+v) - 2P(σ) + P(σ-v)) / v²
    # ────────────────────────────────────────────────────────────────────────────

    function _volga_kernel!(
        out,
        spots, strikes, tenors, sigmas, rights,
        tenor_ids, pv_divs_arr,
        r_by_tenor, df_by_tenor,
        da, dtimes, n_divs::Int32,
        d_sigma::Float32,
        N_t::Int32, N_s::Int32
    )
        idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        idx > Int32(length(spots)) && return nothing

        S, K, T, sigma = spots[idx], strikes[idx], tenors[idx], sigmas[idx]
        is_call = rights[idx] != 0x00

        if !isfinite(S) || !isfinite(K) || !isfinite(T) || !isfinite(sigma)
            out[idx] = NaN32
            return nothing
        end
        if T <= 0f0 || S <= 0f0 || K <= 0f0 || sigma <= 0f0
            out[idx] = 0f0
            return nothing
        end

        tid = tenor_ids[idx]
        pv_divs = pv_divs_arr[tid]
        r_start = (tid - Int32(1)) * RATE_SCHED_STRIDE + Int32(1)
        r_view  = view(r_by_tenor,  r_start:r_start + RATE_SCHED_STRIDE - Int32(1))
        df_view = view(df_by_tenor, r_start:r_start + RATE_SCHED_STRIDE - Int32(1))

        sig_up   = sigma + d_sigma
        sig_down = max(sigma - d_sigma, 1f-6)

        p_up   = _d_fd_price!(S, K, T, sig_up,   is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        p_mid  = _d_fd_price!(S, K, T, sigma,    is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)
        p_down = _d_fd_price!(S, K, T, sig_down, is_call, pv_divs, r_view, df_view, da, dtimes, n_divs, N_t, N_s)

        out[idx] = (p_up - 2f0 * p_mid + p_down) / (d_sigma * d_sigma)

        return nothing
    end

    function _v_volga_gpu(
        spots, strikes, tenors, sigmas, rights,
        rates_curve, rates_times, div_amounts, div_times,
        d_sigma, time_steps, space_steps
    )
        n = length(spots)
        N_t = Int32(clamp(time_steps, 10, 256))
        N_s = Int32(clamp(space_steps, 10, 256))

        tenor_ids, unique_tenors = _build_tenor_map(tenors)
        n_unique = length(unique_tenors)

        pv_by_tenor = Vector{Float32}(undef, n_unique)
        for tid in 1:n_unique
            T = unique_tenors[tid]
            acc = 0f0
            for j in eachindex(div_amounts)
                td = div_times[j]
                if 0f0 < td <= T
                    acc += div_amounts[j] * df_0_t(td, rates_curve, rates_times)
                end
            end
            pv_by_tenor[tid] = acc
        end

        cu_r_by_tenor  = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_df_by_tenor = CUDA.zeros(Float32, n_unique * Int(RATE_SCHED_STRIDE))
        cu_unique_tenors = CuArray(unique_tenors)
        cu_rates_curve   = CuArray(rates_curve)
        cu_rates_times   = CuArray(rates_times)

        threads_2d = (min(n_unique, 32), min(Int(RATE_SCHED_STRIDE), 32))
        blocks_2d  = (cld(n_unique, threads_2d[1]), cld(Int(RATE_SCHED_STRIDE), threads_2d[2]))

        @cuda threads=threads_2d blocks=blocks_2d _build_rate_sched_kernel!(
            cu_unique_tenors, Int32(n_unique),
            cu_rates_curve, cu_rates_times, N_t,
            cu_r_by_tenor, cu_df_by_tenor)

        cu_spots = CuArray(spots); cu_strikes = CuArray(strikes)
        cu_tenors = CuArray(tenors); cu_sigmas = CuArray(sigmas)
        cu_rights = CuArray(rights); cu_tenor_ids = CuArray(tenor_ids)
        cu_pv = CuArray(pv_by_tenor)
        cu_da = CuArray(div_amounts); cu_dt = CuArray(div_times)
        out = CUDA.fill(NaN32, n)

        nthreads = 256
        @cuda threads=nthreads blocks=cld(n, nthreads) _volga_kernel!(
            out,
            cu_spots, cu_strikes, cu_tenors, cu_sigmas, cu_rights,
            cu_tenor_ids, cu_pv,
            cu_r_by_tenor, cu_df_by_tenor,
            cu_da, cu_dt, Int32(length(div_amounts)),
            Float32(d_sigma), N_t, N_s)

        CUDA.synchronize()
        Array(out)
    end
end

# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

"""
    get_delta_fd(...)
    Calculates Delta (dP/dS) for a single American option using a central finite difference on the CPU.
"""
function get_delta_fd(
    spot::Float32, strike::Float32, tenor::Float32, sigma::Float32, right::UInt8,
    rates_curve::Vector{Float32}, rates_times::Vector{Float32},
    div_amounts::Vector{Float32}, div_times::Vector{Float32};
    d_spot::Float32 = 1f-2, time_steps::Int32 = 200, space_steps::Int32 = 200
)::Float32
    is_call = right != 0x00
    if !isfinite(spot) || !isfinite(strike) || !isfinite(tenor) || !isfinite(sigma) || tenor <= 0f0 || spot <= 0f0 || strike <= 0f0 || sigma <= 0f0
        return is_call ? (spot > strike ? 1f0 : 0f0) : (spot < strike ? -1f0 : 0f0)
    end
    if isempty(div_amounts) || isempty(div_times)
       pv_divs = 0f0
    else
        pv_divs = _safe_div_pv(div_amounts, div_times, tenor, rates_curve, rates_times)
    end

    N_t = Int32(clamp(time_steps, 10, 256)); N_s = Int32(clamp(space_steps, 10, 256))
    r_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE); df_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE)
    _build_rate_sched_cpu!(r_sched, df_sched, tenor, N_t, rates_curve, rates_times)
    n_divs = Int32(length(div_amounts))
    S_up = spot + d_spot
    S_down = max(spot - d_spot, 1f-6)
    p_up = _d_fd_price!(S_up, strike, tenor, sigma, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
    p_down = _d_fd_price!(S_down, strike, tenor, sigma, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
    return (p_up - p_down) / (S_up - S_down)
end

 """
     get_gamma_fd(...)

 Calculates Gamma (d²P/dS²) for a single American option using a central finite difference on the CPU.
 """
 function get_gamma_fd(
     spot::Float32, strike::Float32, tenor::Float32, sigma::Float32, right::UInt8,
     rates_curve::Vector{Float32}, rates_times::Vector{Float32},
     div_amounts::Vector{Float32}, div_times::Vector{Float32};
     d_spot::Float32 = 1f-1, time_steps::Int32 = 200, space_steps::Int32 = 200
 )::Float32
     is_call = right != 0x00
     if !isfinite(spot) || !isfinite(strike) || !isfinite(tenor) || !isfinite(sigma) || tenor <= 0f0 || spot <= 0f0 || strike <= 0f0 || sigma <= 0f0
         return 0f0
     end
     if isempty(div_amounts) || isempty(div_times)
       pv_divs = 0f0
    else
        pv_divs = _safe_div_pv(div_amounts, div_times, tenor, rates_curve, rates_times)
    end

     N_t = Int32(clamp(time_steps, 10, 256)); N_s = Int32(clamp(space_steps, 10, 256))
     r_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE); df_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE)
     _build_rate_sched_cpu!(r_sched, df_sched, tenor, N_t, rates_curve, rates_times)
     n_divs = Int32(length(div_amounts))

     S_up = spot + d_spot
     S_down = max(spot - d_spot, 1f-6)

     p_up = _d_fd_price!(S_up, strike, tenor, sigma, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     p_mid = _d_fd_price!(spot, strike, tenor, sigma, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     p_down = _d_fd_price!(S_down, strike, tenor, sigma, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)

     h = 0.5f0 * (S_up - S_down)
     return (p_up - 2f0 * p_mid + p_down) / (h * h)
 end

 """
get_gamma_decay_fd(...)
 Calculates Gamma Decay (dΓ/dt) for a single American option on the CPU.
 """
 function get_gamma_decay_fd(
     spot::Float32, strike::Float32, tenor::Float32, sigma::Float32, right::UInt8,
     rates_curve::Vector{Float32}, rates_times::Vector{Float32},
     div_amounts::Vector{Float32}, div_times::Vector{Float32};
     d_spot::Float32 = 1f-1, d_time::Float32 = 1f0/365f0, time_steps::Int32 = 200, space_steps::Int32 = 200
 )::Float32
     is_call = right != 0x00
     if !isfinite(spot) || !isfinite(strike) || !isfinite(tenor) || !isfinite(sigma) || tenor <= 0f0 || spot <= 0f0 || strike <= 0f0 || sigma <= 0f0
         return 0f0
     end
     if isempty(div_amounts) || isempty(div_times)
        div_amounts = Float32[]
        div_times = Float32[]
    end

     N_t = Int32(clamp(time_steps, 10, 256)); N_s = Int32(clamp(space_steps, 10, 256))
     n_divs = Int32(length(div_amounts))
     S_up_bump = spot + d_spot; S_down_bump = max(spot - d_spot, 1f-6)
     h_gamma = 0.5f0 * (S_up_bump - S_down_bump)

     # --- Gamma at T_up ---
     T_up = tenor + d_time
     pv_divs_up = _safe_div_pv(div_amounts, div_times, T_up, rates_curve, rates_times)
     r_sched_up = Vector{Float32}(undef, RATE_SCHED_STRIDE); df_sched_up = Vector{Float32}(undef, RATE_SCHED_STRIDE)
     _build_rate_sched_cpu!(r_sched_up, df_sched_up, T_up, N_t, rates_curve, rates_times)

     p_up_up = _d_fd_price!(S_up_bump, strike, T_up, sigma, is_call, pv_divs_up, r_sched_up, df_sched_up, div_amounts, div_times, n_divs, N_t, N_s)
     p_up_mid = _d_fd_price!(spot, strike, T_up, sigma, is_call, pv_divs_up, r_sched_up, df_sched_up, div_amounts, div_times, n_divs, N_t, N_s)
     p_up_down = _d_fd_price!(S_down_bump, strike, T_up, sigma, is_call, pv_divs_up, r_sched_up, df_sched_up, div_amounts, div_times, n_divs, N_t, N_s)
     gamma_up = (p_up_up - 2f0 * p_up_mid + p_up_down) / (h_gamma * h_gamma)

     # --- Gamma at T_down ---
     T_down = max(tenor - d_time, 1f-6)
     pv_divs_down = _safe_div_pv(div_amounts, div_times, T_down, rates_curve, rates_times)
     r_sched_down = Vector{Float32}(undef, RATE_SCHED_STRIDE); df_sched_down = Vector{Float32}(undef, RATE_SCHED_STRIDE)
     _build_rate_sched_cpu!(r_sched_down, df_sched_down, T_down, N_t, rates_curve, rates_times)

     p_down_up = _d_fd_price!(S_up_bump, strike, T_down, sigma, is_call, pv_divs_down, r_sched_down, df_sched_down, div_amounts, div_times, n_divs, N_t, N_s)
     p_down_mid = _d_fd_price!(spot, strike, T_down, sigma, is_call, pv_divs_down, r_sched_down, df_sched_down, div_amounts, div_times, n_divs, N_t, N_s)
     p_down_down = _d_fd_price!(S_down_bump, strike, T_down, sigma, is_call, pv_divs_down, r_sched_down, df_sched_down, div_amounts, div_times, n_divs, N_t, N_s)
     gamma_down = (p_down_up - 2f0 * p_down_mid + p_down_down) / (h_gamma * h_gamma)

     return (gamma_down - gamma_up) / (T_up - T_down)
 end

 """
     get_gamma_vol_fd(...)
    Calculates GammaVol (dΓ/dσ) for a single American option on the CPU.
 """
 function get_gamma_vol_fd(
     spot::Float32, strike::Float32, tenor::Float32, sigma::Float32, right::UInt8,
     rates_curve::Vector{Float32}, rates_times::Vector{Float32},
     div_amounts::Vector{Float32}, div_times::Vector{Float32};
     d_spot::Float32 = 1f-1, d_sigma::Float32 = 1f-3, time_steps::Int32 = 200, space_steps::Int32 = 200
 )::Float32
     is_call = right != 0x00
     if !isfinite(spot) || !isfinite(strike) || !isfinite(tenor) || !isfinite(sigma) || tenor <= 0f0 || spot <= 0f0 || strike <= 0f0 || sigma <= 0f0
         return 0f0
     end
     if isempty(div_amounts) || isempty(div_times)
        div_amounts = Float32[]
        div_times = Float32[]
    end

     N_t = Int32(clamp(time_steps, 10, 256)); N_s = Int32(clamp(space_steps, 10, 256))
     pv_divs = _safe_div_pv(div_amounts, div_times, tenor, rates_curve, rates_times)
     r_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE); df_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE)
     _build_rate_sched_cpu!(r_sched, df_sched, tenor, N_t, rates_curve, rates_times)
     n_divs = Int32(length(div_amounts))

     S_up_bump = spot + d_spot; S_down_bump = max(spot - d_spot, 1f-6)
     h_gamma = 0.5f0 * (S_up_bump - S_down_bump)

     # --- Gamma at sigma_up ---
     sigma_up = sigma + d_sigma
     p_up_up = _d_fd_price!(S_up_bump, strike, tenor, sigma_up, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     p_up_mid = _d_fd_price!(spot, strike, tenor, sigma_up, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     p_up_down = _d_fd_price!(S_down_bump, strike, tenor, sigma_up, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     gamma_up = (p_up_up - 2f0 * p_up_mid + p_up_down) / (h_gamma * h_gamma)

     # --- Gamma at sigma_down ---
     sigma_down = max(sigma - d_sigma, 1f-6)
     p_down_up = _d_fd_price!(S_up_bump, strike, tenor, sigma_down, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     p_down_mid = _d_fd_price!(spot, strike, tenor, sigma_down, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     p_down_down = _d_fd_price!(S_down_bump, strike, tenor, sigma_down, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     gamma_down = (p_down_up - 2f0 * p_down_mid + p_down_down) / (h_gamma * h_gamma)

     return (gamma_up - gamma_down) / (sigma_up - sigma_down)
 end

 """
     get_iv_fd(price, spot, strike, tenor, right, rates_curve, rates_times, div_amounts, div_times; ...)

 Calculates the implied volatility for a single American option using a finite difference method on the CPU.

 This function is the scalar counterpart to `get_v_iv_fd`. It shares the core pricing
 logic with the GPU version for consistency.

 # Arguments
 - `price::Float32`: Market price of the option.
 - `spot::Float32`: Current price of the underlying asset.
 - `strike::Float32`: Strike price of the option.
 - `tenor::Float32`: Time to expiry in years.
 - `right::UInt8`: `0x01` for a call option, `0x00` for a put.
 - `rates_curve::Vector{Float32}`: Interest rates for the curve.
 - `rates_times::Vector{Float32}`: Tenors for the interest rates.
 - `div_amounts::Vector{Float32}`: Amounts of discrete dividends.
 - `div_times::Vector{Float32}`: Times of discrete dividends.

 # Keyword Arguments
 - `tol::Float32 = 1f-6`: Tolerance for the root-finding algorithm.
 - `max_iter::Int = 100`: Maximum number of iterations for the root-finder.
 - `v_min::Float32 = 1f-4`: Minimum volatility guess.
 - `v_max::Float32 = 5f0`: Maximum volatility guess.
 - `time_steps::Int32 = 200`: Number of time steps in the finite difference grid.
 - `space_steps::Int32 = 200`: Number of space (spot price) steps in the grid.

 # Returns
 - `Float32`: The calculated implied volatility.
 """
function get_iv_fd(
     price::Float32, spot::Float32, strike::Float32,
     tenor::Float32, right::UInt8,
     rates_curve::Vector{Float32}, rates_times::Vector{Float32},
     div_amounts::Vector{Float32}, div_times::Vector{Float32};
     tol::Float32 = 1f-6, max_iter::Int = 100,
     v_min::Float32 = 1f-4, v_max::Float32 = 5f0,
     time_steps::Int32 = Int32(200), space_steps::Int32 = Int32(200)
)::Float32
    if isempty(rates_times) || isempty(rates_curve)
        return NaN32
    end
    if isempty(div_amounts) || isempty(div_times)
        div_amounts = Float32[]
        div_times = Float32[]
    end
    N_t = Int32(clamp(time_steps, 10, 256))
    N_s = Int32(clamp(space_steps, 10, 256))

    is_call = right == 0x01

    if tenor <= 0f0
        @warn "get_iv_fd(): Received bad tenor: $tenor"
        return 0f0
    end
    if spot <= 0f0 || strike <= 0f0 || price <= 0f0
        @warn "get_iv_fd(): Received bad inputs: striek=$strike price=$price"
        return NaN32
    end

    intrinsic = is_call ? max(spot - strike, 0f0) : max(strike - spot, 0f0)
    if abs(price - intrinsic) < 1f-6 || price < intrinsic - 1f-6
        return v_min
    end

    pv_divs = _safe_div_pv(div_amounts, div_times, tenor, rates_curve, rates_times)

    r_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE)
    df_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE)
    _build_rate_sched_cpu!(r_sched, df_sched, tenor, N_t, rates_curve, rates_times)

     n_divs = Int32(length(div_amounts))

     p_low = _d_fd_price!(spot, strike, tenor, v_min, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     price <= p_low + 1f-9 && return v_min

     p_high = _d_fd_price!(spot, strike, tenor, v_max, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     price >= p_high - 1f-9 && return v_max

     # Bisection search
     lo, hi = v_min, v_max
     mid = 0.5f0 * (lo + hi)
     rel_tol = tol / max(price, 1f0)

     for _ in 1:max_iter
         mid = 0.5f0 * (lo + hi)
         p_mid = _d_fd_price!(spot, strike, tenor, mid, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
         diff = p_mid - price
         (abs(diff) < tol || abs(diff) / price < rel_tol) && break
         diff < 0f0 ? (lo = mid) : (hi = mid)
         (hi - lo < v_min * 0.01f0) && break
     end

    if !isfinite(mid)
        @warn "get_iv_fd(). inf mid=$mid, hi=$hi, lo=$lo. Inputs: $((spot, strike, tenor, is_call, n_divs, N_t, N_s))"
    end

    return mid < 0 ? 0 : mid
 end

"""
     get_speed_fd(...)

 Calculates Speed (dΓ/dS) for a single American option using a 5-point stencil on the CPU.
 """
 function get_speed_fd(
     spot::Float32, strike::Float32, tenor::Float32, sigma::Float32, right::UInt8,
     rates_curve::Vector{Float32}, rates_times::Vector{Float32},
     div_amounts::Vector{Float32}, div_times::Vector{Float32};
     d_spot::Float32 = 1f-1, time_steps::Int32 = 200, space_steps::Int32 = 200
 )::Float32
     is_call = right != 0x00
     if !isfinite(spot) || !isfinite(strike) || !isfinite(tenor) || !isfinite(sigma) || tenor <= 0f0 || spot <= 0f0 || strike <= 0f0 || sigma <= 0f0
         return 0f0
     end
     if isempty(div_amounts) || isempty(div_times)
        div_amounts = Float32[]
        div_times = Float32[]
    end

     N_t = Int32(clamp(time_steps, 10, 256)); N_s = Int32(clamp(space_steps, 10, 256))
     pv_divs = _safe_div_pv(div_amounts, div_times, tenor, rates_curve, rates_times)
     r_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE); df_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE)
     _build_rate_sched_cpu!(r_sched, df_sched, tenor, N_t, rates_curve, rates_times)
     n_divs = Int32(length(div_amounts))

     h = d_spot
     S_p2 = spot + 2f0 * h
     S_p1 = spot + h
     S_m1 = max(spot - h, 1f-6)
     S_m2 = max(spot - 2f0 * h, 1f-6)

     p_p2 = _d_fd_price!(S_p2, strike, tenor, sigma, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     p_p1 = _d_fd_price!(S_p1, strike, tenor, sigma, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     p_m1 = _d_fd_price!(S_m1, strike, tenor, sigma, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     p_m2 = _d_fd_price!(S_m2, strike, tenor, sigma, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)

     return (-p_p2 + 2f0 * p_p1 - 2f0 * p_m1 + p_m2) / (2f0 * h * h * h)
 end

 """
     get_theta_fd(...)

 Calculates Theta (dP/dt) for a single American option using a backward finite difference on the CPU.
 """
 function get_theta_fd(
     spot::Float32, strike::Float32, tenor::Float32, sigma::Float32, right::UInt8,
     rates_curve::Vector{Float32}, rates_times::Vector{Float32},
     div_amounts::Vector{Float32}, div_times::Vector{Float32};
     d_time::Float32 = 1f0/365f0, time_steps::Int32 = 200, space_steps::Int32 = 200
 )::Float32
     is_call = right != 0x00
     if !isfinite(spot) || !isfinite(strike) || !isfinite(tenor) || !isfinite(sigma) || tenor <= 0f0 || spot <= 0f0 || strike <= 0f0 || sigma <= 0f0
         return 0f0
     end

     N_t = Int32(clamp(time_steps, 10, 256)); N_s = Int32(clamp(space_steps, 10, 256))
     n_divs = Int32(length(div_amounts))

     # --- Price at T ---
     pv_divs_now = _safe_div_pv(div_amounts, div_times, tenor, rates_curve, rates_times)
     r_sched_now = Vector{Float32}(undef, RATE_SCHED_STRIDE); df_sched_now = Vector{Float32}(undef, RATE_SCHED_STRIDE)
     _build_rate_sched_cpu!(r_sched_now, df_sched_now, tenor, N_t, rates_curve, rates_times)
     p_now = _d_fd_price!(spot, strike, tenor, sigma, is_call, pv_divs_now, r_sched_now, df_sched_now, div_amounts, div_times, n_divs, N_t, N_s)

     # --- Price at T_down ---
     T_down = max(tenor - d_time, 1f-6)
     pv_divs_down = _safe_div_pv(div_amounts, div_times, T_down, rates_curve, rates_times)
     r_sched_down = Vector{Float32}(undef, RATE_SCHED_STRIDE); df_sched_down = Vector{Float32}(undef, RATE_SCHED_STRIDE)
     _build_rate_sched_cpu!(r_sched_down, df_sched_down, T_down, N_t, rates_curve, rates_times)
     p_down = _d_fd_price!(spot, strike, T_down, sigma, is_call, pv_divs_down, r_sched_down, df_sched_down, div_amounts, div_times, n_divs, N_t, N_s)

     return (p_down - p_now) / (T_down - tenor)
 end

 """
     get_theta_decay_fd(...)

 Calculates Theta Decay (d²P/dt²) for a single American option on the CPU.
 """
 function get_theta_decay_fd(
     spot::Float32, strike::Float32, tenor::Float32, sigma::Float32, right::UInt8,
     rates_curve::Vector{Float32}, rates_times::Vector{Float32},
     div_amounts::Vector{Float32}, div_times::Vector{Float32};
     d_time::Float32 = 1f0/365f0, time_steps::Int32 = 200, space_steps::Int32 = 200
 )::Float32
     is_call = right != 0x00
     if !isfinite(spot) || !isfinite(strike) || !isfinite(tenor) || !isfinite(sigma) || tenor <= 0f0 || spot <= 0f0 || strike <= 0f0 || sigma <= 0f0
         return 0f0
     end
     if isempty(div_amounts) || isempty(div_times)
        div_amounts = Float32[]
        div_times = Float32[]
    end

     N_t = Int32(clamp(time_steps, 10, 256)); N_s = Int32(clamp(space_steps, 10, 256))
     n_divs = Int32(length(div_amounts))

     # --- Price at T_up ---
     T_up = tenor + d_time
     pv_divs_up = _safe_div_pv(div_amounts, div_times, T_up, rates_curve, rates_times)
     r_sched_up = Vector{Float32}(undef, RATE_SCHED_STRIDE); df_sched_up = Vector{Float32}(undef, RATE_SCHED_STRIDE)
     _build_rate_sched_cpu!(r_sched_up, df_sched_up, T_up, N_t, rates_curve, rates_times)
     p_up = _d_fd_price!(spot, strike, T_up, sigma, is_call, pv_divs_up, r_sched_up, df_sched_up, div_amounts, div_times, n_divs, N_t, N_s)

     # --- Price at T ---
     pv_divs_mid = _safe_div_pv(div_amounts, div_times, tenor, rates_curve, rates_times)
     r_sched_mid = Vector{Float32}(undef, RATE_SCHED_STRIDE); df_sched_mid = Vector{Float32}(undef, RATE_SCHED_STRIDE)
     _build_rate_sched_cpu!(r_sched_mid, df_sched_mid, tenor, N_t, rates_curve, rates_times)
     p_mid = _d_fd_price!(spot, strike, tenor, sigma, is_call, pv_divs_mid, r_sched_mid, df_sched_mid, div_amounts, div_times, n_divs, N_t, N_s)

     # --- Price at T_down ---
     T_down = max(tenor - d_time, 1f-6)
     pv_divs_down = _safe_div_pv(div_amounts, div_times, T_down, rates_curve, rates_times)
     r_sched_down = Vector{Float32}(undef, RATE_SCHED_STRIDE); df_sched_down = Vector{Float32}(undef, RATE_SCHED_STRIDE)
     _build_rate_sched_cpu!(r_sched_down, df_sched_down, T_down, N_t, rates_curve, rates_times)
     p_down = _d_fd_price!(spot, strike, T_down, sigma, is_call, pv_divs_down, r_sched_down, df_sched_down, div_amounts, div_times, n_divs, N_t, N_s)

     return (p_up - 2f0 * p_mid + p_down) / (d_time * d_time)
 end

 """
     get_vanna_fd(...)

 Calculates Vanna (dΔ/dσ) for a single American option using a 2D finite difference on the CPU.
 """
 function get_vanna_fd(
     spot::Float32, strike::Float32, tenor::Float32, sigma::Float32, right::UInt8,
     rates_curve::Vector{Float32}, rates_times::Vector{Float32},
     div_amounts::Vector{Float32}, div_times::Vector{Float32};
     d_spot::Float32 = 1f-1, d_sigma::Float32 = 1f-3, time_steps::Int32 = 200, space_steps::Int32 = 200
 )::Float32
     is_call = right != 0x00
     if !isfinite(spot) || !isfinite(strike) || !isfinite(tenor) || !isfinite(sigma) || tenor <= 0f0 || spot <= 0f0 || strike <= 0f0 || sigma <= 0f0
         return 0f0
     end
     if isempty(div_amounts) || isempty(div_times)
        div_amounts = Float32[]
        div_times = Float32[]
    end

     N_t = Int32(clamp(time_steps, 10, 256)); N_s = Int32(clamp(space_steps, 10, 256))
     pv_divs = _safe_div_pv(div_amounts, div_times, tenor, rates_curve, rates_times)
     r_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE); df_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE)
     _build_rate_sched_cpu!(r_sched, df_sched, tenor, N_t, rates_curve, rates_times)
     n_divs = Int32(length(div_amounts))

     S_up = spot + d_spot; S_down = max(spot - d_spot, 1f-6)
     sig_up = sigma + d_sigma; sig_down = max(sigma - d_sigma, 1f-6)

     pp = _d_fd_price!(S_up, strike, tenor, sig_up, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     pm = _d_fd_price!(S_up, strike, tenor, sig_down, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     mp = _d_fd_price!(S_down, strike, tenor, sig_up, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     mm = _d_fd_price!(S_down, strike, tenor, sig_down, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)

     return (pp - pm - mp + mm) / ((S_up - S_down) * (sig_up - sig_down))
 end

 """
     get_vega_fd(...)

 Calculates Vega (dP/dσ) for a single American option using a central finite difference on the CPU.
 """
 function get_vega_fd(
     spot::Float32, strike::Float32, tenor::Float32, sigma::Float32, right::UInt8,
     rates_curve::Vector{Float32}, rates_times::Vector{Float32},
     div_amounts::Vector{Float32}, div_times::Vector{Float32};
     d_sigma::Float32 = 1f-3, time_steps::Int32 = 200, space_steps::Int32 = 200
 )::Float32
     is_call = right != 0x00
     if !isfinite(spot) || !isfinite(strike) || !isfinite(tenor) || !isfinite(sigma) || tenor <= 0f0 || spot <= 0f0 || strike <= 0f0 || sigma <= 0f0
         return 0f0
     end
     if isempty(div_amounts) || isempty(div_times)
        div_amounts = Float32[]
        div_times = Float32[]
    end

     N_t = Int32(clamp(time_steps, 10, 256)); N_s = Int32(clamp(space_steps, 10, 256))
     pv_divs = _safe_div_pv(div_amounts, div_times, tenor, rates_curve, rates_times)
     r_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE); df_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE)
     _build_rate_sched_cpu!(r_sched, df_sched, tenor, N_t, rates_curve, rates_times)
     n_divs = Int32(length(div_amounts))

     sigma_up = sigma + d_sigma
     sigma_down = max(sigma - d_sigma, 1f-6)

     p_up = _d_fd_price!(spot, strike, tenor, sigma_up, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     p_down = _d_fd_price!(spot, strike, tenor, sigma_down, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)

     return (p_up - p_down) / (sigma_up - sigma_down)
 end

 """
     get_vega_decay_fd(...)

 Calculates Vega Decay (dVega/dt) for a single American option on the CPU.
 """
 function get_vega_decay_fd(
     spot::Float32, strike::Float32, tenor::Float32, sigma::Float32, right::UInt8,
     rates_curve::Vector{Float32}, rates_times::Vector{Float32},
     div_amounts::Vector{Float32}, div_times::Vector{Float32};
     d_sigma::Float32 = 1f-3, d_time::Float32 = 1f0/365f0, time_steps::Int32 = 200, space_steps::Int32 = 200
 )::Float32
     is_call = right != 0x00
     if !isfinite(spot) || !isfinite(strike) || !isfinite(tenor) || !isfinite(sigma) || tenor <= 0f0 || spot <= 0f0 || strike <= 0f0 || sigma <= 0f0
         return 0f0
     end
     if isempty(div_amounts) || isempty(div_times)
        div_amounts = Float32[]
        div_times = Float32[]
    end

     N_t = Int32(clamp(time_steps, 10, 256)); N_s = Int32(clamp(space_steps, 10, 256))
     n_divs = Int32(length(div_amounts))
     sig_up_bump = sigma + d_sigma; sig_down_bump = max(sigma - d_sigma, 1f-6)
     h_vega = sig_up_bump - sig_down_bump

     # --- Vega at T_up ---
     T_up = tenor + d_time
     pv_divs_up = _safe_div_pv(div_amounts, div_times, T_up, rates_curve, rates_times)
     r_sched_up = Vector{Float32}(undef, RATE_SCHED_STRIDE); df_sched_up = Vector{Float32}(undef, RATE_SCHED_STRIDE)
     _build_rate_sched_cpu!(r_sched_up, df_sched_up, T_up, N_t, rates_curve, rates_times)

     p_up_up = _d_fd_price!(spot, strike, T_up, sig_up_bump, is_call, pv_divs_up, r_sched_up, df_sched_up, div_amounts, div_times, n_divs, N_t, N_s)
     p_up_down = _d_fd_price!(spot, strike, T_up, sig_down_bump, is_call, pv_divs_up, r_sched_up, df_sched_up, div_amounts, div_times, n_divs, N_t, N_s)
     vega_up = (p_up_up - p_up_down) / h_vega

     # --- Vega at T_down ---
     T_down = max(tenor - d_time, 1f-6)
     pv_divs_down = _safe_div_pv(div_amounts, div_times, T_down, rates_curve, rates_times)
     r_sched_down = Vector{Float32}(undef, RATE_SCHED_STRIDE); df_sched_down = Vector{Float32}(undef, RATE_SCHED_STRIDE)
     _build_rate_sched_cpu!(r_sched_down, df_sched_down, T_down, N_t, rates_curve, rates_times)

     p_down_up = _d_fd_price!(spot, strike, T_down, sig_up_bump, is_call, pv_divs_down, r_sched_down, df_sched_down, div_amounts, div_times, n_divs, N_t, N_s)
     p_down_down = _d_fd_price!(spot, strike, T_down, sig_down_bump, is_call, pv_divs_down, r_sched_down, df_sched_down, div_amounts, div_times, n_divs, N_t, N_s)
     vega_down = (p_down_up - p_down_down) / h_vega

     return (vega_down - vega_up) / (T_up - T_down)
 end

 """
     get_volga_fd(...)

 Calculates Volga (d²P/dσ²) for a single American option using a central finite difference on the CPU.
 """
 function get_volga_fd(
     spot::Float32, strike::Float32, tenor::Float32, sigma::Float32, right::UInt8,
     rates_curve::Vector{Float32}, rates_times::Vector{Float32},
     div_amounts::Vector{Float32}, div_times::Vector{Float32};
     d_sigma::Float32 = 1f-3, time_steps::Int32 = 200, space_steps::Int32 = 200
 )::Float32
     is_call = right != 0x00
     if !isfinite(spot) || !isfinite(strike) || !isfinite(tenor) || !isfinite(sigma) || tenor <= 0f0 || spot <= 0f0 || strike <= 0f0 || sigma <= 0f0
         return 0f0
     end

     N_t = Int32(clamp(time_steps, 10, 256)); N_s = Int32(clamp(space_steps, 10, 256))
     pv_divs = _safe_div_pv(div_amounts, div_times, tenor, rates_curve, rates_times)
     r_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE); df_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE)
     _build_rate_sched_cpu!(r_sched, df_sched, tenor, N_t, rates_curve, rates_times)
     n_divs = Int32(length(div_amounts))

     sigma_up = sigma + d_sigma
     sigma_down = max(sigma - d_sigma, 1f-6)

     p_up = _d_fd_price!(spot, strike, tenor, sigma_up, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     p_mid = _d_fd_price!(spot, strike, tenor, sigma, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)
     p_down = _d_fd_price!(spot, strike, tenor, sigma_down, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, n_divs, N_t, N_s)

     return (p_up - 2f0 * p_mid + p_down) / (d_sigma * d_sigma)
 end


"""
    get_price_fd(spot, strike, tenor, sigma, right, rates_curve, rates_times, div_amounts, div_times; ...)

Calculates the price of a single American option using a finite difference method on the CPU.

This function is the scalar counterpart to `get_v_price_fd`. It shares the core pricing
logic with the GPU version for consistency.

# Arguments
- `spot::Float32`: Current price of the underlying asset.
- `strike::Float32`: Strike price of the option.
- `tenor::Float32`: Time to expiry in years.
- `sigma::Float32`: Implied volatility.
- `right::UInt8`: `0x01` for a call option, `0x00` for a put.
- `rates_curve::Vector{Float32}`: Interest rates for the curve.
- `rates_times::Vector{Float32}`: Tenors for the interest rates.
- `div_amounts::Vector{Float32}`: Amounts of discrete dividends.
- `div_times::Vector{Float32}`: Times of discrete dividends.

# Keyword Arguments
- `time_steps::Int32 = 200`: Number of time steps in the finite difference grid.
- `space_steps::Int32 = 200`: Number of space (spot price) steps in the grid.

# Returns
- `Float32`: The calculated option price.
"""
function get_price_fd(
    spot::Float32, strike::Float32,
    tenor::Float32, sigma::Float32,
    right::UInt8,
    rates_curve::Vector{Float32}, rates_times::Vector{Float32},
    div_amounts::Vector{Float32}, div_times::Vector{Float32};
    time_steps::Int32 = 200, space_steps::Int32 = 200
)::Float32
    is_call = right != 0x00
    if tenor <= 0f0 || spot <= 0f0 || strike <= 0f0 || sigma <= 0f0
        return is_call ? max(spot - strike, 0f0) : max(strike - spot, 0f0)
    end
    if isempty(div_amounts) || isempty(div_times)
        div_amounts = Float32[]
        div_times = Float32[]
    end

    N_t = Int32(clamp(time_steps, 10, 256))
    N_s = Int32(clamp(space_steps, 10, 256))

    pv_divs = _safe_div_pv(div_amounts, div_times, tenor, rates_curve, rates_times)

    r_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE)
    df_sched = Vector{Float32}(undef, RATE_SCHED_STRIDE)

    _build_rate_sched_cpu!(r_sched, df_sched, tenor, N_t, rates_curve, rates_times)


    return _d_fd_price!(spot, strike, tenor, sigma, is_call, pv_divs, r_sched, df_sched, div_amounts, div_times, Int32(length(div_amounts)), N_t, N_s)
end

function get_v_price_fd(
    spots::Vector{Float32}, strikes::Vector{Float32},
    tenors::Vector{Float32}, sigmas::Vector{Float32},
    rights::Vector{UInt8},
    rates_curve::Vector{Float32}, rates_times::Vector{Float32},
    div_amounts::Vector{Float32}, div_times::Vector{Float32};
    time_steps::Int = 200, space_steps::Int = 200
)::Vector{Float32}
    CUDA_AVAILABLE || error("CUDA is required: CPU implementation was removed.")
    _v_price_gpu(spots, strikes, tenors, sigmas, rights,
        rates_curve, rates_times, div_amounts, div_times,
        time_steps, space_steps)
end

function get_v_iv_fd(
    prices::Vector{Float32}, spots::Vector{Float32},
    strikes::Vector{Float32}, tenors::Vector{Float32},
    rights::Vector{UInt8},
    rates_curve::Vector{Float32}, rates_times::Vector{Float32},
    div_amounts::Vector{Float32}, div_times::Vector{Float32};
    tol::Float32 = 1f-6, max_iter::Int = 100,
    v_min::Float32 = 1f-4, v_max::Float32 = 5f0,
    time_steps::Int = 200, space_steps::Int = 200
)::Vector{Float32}
    CUDA_AVAILABLE || error("CUDA is required: CPU implementation was removed.")
    _v_iv_gpu(prices, spots, strikes, tenors, rights,
        rates_curve, rates_times, div_amounts, div_times,
        tol, max_iter, v_min, v_max, time_steps, space_steps)
end

function get_v_vega_fd(
    spots::Vector{Float32}, strikes::Vector{Float32},
    tenors::Vector{Float32}, sigmas::Vector{Float32},
    v_is_call::Vector{UInt8},
    rates_curve::Vector{Float32}, rates_times::Vector{Float32},
    div_amounts::Vector{Float32}, div_times::Vector{Float32};
    d_sigma::Float32 = 1f-3,
    time_steps::Int = 200,
    space_steps::Int = 200
)::Vector{Float32}
    CUDA_AVAILABLE || error("CUDA is required: CPU implementation was removed.")
    _v_vega_gpu(
        spots, strikes, tenors, sigmas, v_is_call,
        rates_curve, rates_times, div_amounts, div_times,
        d_sigma, time_steps, space_steps
    )
end

function get_v_delta_fd(
    spots::Vector{Float32}, strikes::Vector{Float32},
    tenors::Vector{Float32}, sigmas::Vector{Float32},
    v_is_call::Vector{UInt8},
    rates_curve::Vector{Float32}, rates_times::Vector{Float32},
    div_amounts::Vector{Float32}, div_times::Vector{Float32};
    d_spot::Float32 = 1f-2,
    time_steps::Int = 200,
    space_steps::Int = 200
)::Vector{Float32}
    CUDA_AVAILABLE || error("CUDA is required: CPU implementation was removed.")
    _v_delta_gpu(
        spots, strikes, tenors, sigmas, v_is_call,
        rates_curve, rates_times, div_amounts, div_times,
        d_spot, time_steps, space_steps
    )
end

"""
get_v_theta_fd(spots, strikes, tenors, sigmas, v_is_call,
           rates_curve, rates_times, div_amounts, div_times;
           d_time=1/365, time_steps=200, space_steps=200)
Theta = dP/dt, quoted as value change per unit time (same units as T).
Backward difference: [P(T - d_time) - P(T)] / d_time  (negative for long options).
"""
function get_v_theta_fd(
    spots::Vector{Float32}, strikes::Vector{Float32},
    tenors::Vector{Float32}, sigmas::Vector{Float32},
    v_is_call::Vector{UInt8},
    rates_curve::Vector{Float32}, rates_times::Vector{Float32},
    div_amounts::Vector{Float32}, div_times::Vector{Float32};
    d_time::Float32 = 1f0 / 365f0,
    time_steps::Int = 200, space_steps::Int = 200
)::Vector{Float32}
    CUDA_AVAILABLE || error("CUDA is required: CPU implementation was removed.")
    _v_theta_gpu(spots, strikes, tenors, sigmas, v_is_call,
        rates_curve, rates_times, div_amounts, div_times,
        d_time, time_steps, space_steps)
end

"""
    get_v_gamma_fd(...)
Gamma = d²P/dS².  Central second finite difference over spot.
"""
function get_v_gamma_fd(
    spots::Vector{Float32}, strikes::Vector{Float32},
    tenors::Vector{Float32}, sigmas::Vector{Float32},
    v_is_call::Vector{UInt8},
    rates_curve::Vector{Float32}, rates_times::Vector{Float32},
    div_amounts::Vector{Float32}, div_times::Vector{Float32};
    d_spot::Float32 = 1f-1,
    time_steps::Int = 200, space_steps::Int = 200
)::Vector{Float32}
    CUDA_AVAILABLE || error("CUDA is required: CPU implementation was removed.")
    _v_gamma_gpu(spots, strikes, tenors, sigmas, v_is_call,
        rates_curve, rates_times, div_amounts, div_times,
        d_spot, time_steps, space_steps)
end

"""
    get_v_speed_fd(...)
Speed = dΓ/dS = d³P/dS³.  5-point antisymmetric stencil.
"""
function get_v_speed_fd(
    spots::Vector{Float32}, strikes::Vector{Float32},
    tenors::Vector{Float32}, sigmas::Vector{Float32},
    v_is_call::Vector{UInt8},
    rates_curve::Vector{Float32}, rates_times::Vector{Float32},
    div_amounts::Vector{Float32}, div_times::Vector{Float32};
    d_spot::Float32 = 1f-1,
    time_steps::Int = 200, space_steps::Int = 200
)::Vector{Float32}
    CUDA_AVAILABLE || error("CUDA is required: CPU implementation was removed.")
    _v_speed_gpu(spots, strikes, tenors, sigmas, v_is_call,
        rates_curve, rates_times, div_amounts, div_times,
        d_spot, time_steps, space_steps)
end

"""
get_v_gamma_decay_fd(...)
GammaDecay = dΓ/dt.  Central difference of gamma over a time bump.
Positive value means gamma increases as time to expiry grows.
"""
function get_v_gamma_decay_fd(
    spots::Vector{Float32}, strikes::Vector{Float32},
    tenors::Vector{Float32}, sigmas::Vector{Float32},
    v_is_call::Vector{UInt8},
    rates_curve::Vector{Float32}, rates_times::Vector{Float32},
    div_amounts::Vector{Float32}, div_times::Vector{Float32};
    d_spot::Float32 = 1f-1,
    d_time::Float32 = 1f0 / 365f0,
    time_steps::Int = 200, space_steps::Int = 200
)::Vector{Float32}
    CUDA_AVAILABLE || error("CUDA is required: CPU implementation was removed.")
    _v_gamma_decay_gpu(spots, strikes, tenors, sigmas, v_is_call,
        rates_curve, rates_times, div_amounts, div_times,
        d_spot, d_time, time_steps, space_steps)
end

"""
    get_v_gamma_vol_fd(...)
GammaVol = dΓ/dσ.  Central difference of gamma over a vol bump.
"""
function get_v_gamma_vol_fd(
    spots::Vector{Float32}, strikes::Vector{Float32},
    tenors::Vector{Float32}, sigmas::Vector{Float32},
    v_is_call::Vector{UInt8},
    rates_curve::Vector{Float32}, rates_times::Vector{Float32},
    div_amounts::Vector{Float32}, div_times::Vector{Float32};
    d_spot::Float32 = 1f-1,
    d_sigma::Float32 = 1f-3,
    time_steps::Int = 200, space_steps::Int = 200
)::Vector{Float32}
    CUDA_AVAILABLE || error("CUDA is required: CPU implementation was removed.")
    _v_gamma_vol_gpu(spots, strikes, tenors, sigmas, v_is_call,
        rates_curve, rates_times, div_amounts, div_times,
        d_spot, d_sigma, time_steps, space_steps)
end

"""
    get_v_theta_decay_fd(...)
ThetaDecay = dΘ/dt = d²P/dt².  Central second difference over time.
Measures how quickly theta itself accelerates near expiry.
"""
function get_v_theta_decay_fd(
    spots::Vector{Float32}, strikes::Vector{Float32},
    tenors::Vector{Float32}, sigmas::Vector{Float32},
    v_is_call::Vector{UInt8},
    rates_curve::Vector{Float32}, rates_times::Vector{Float32},
    div_amounts::Vector{Float32}, div_times::Vector{Float32};
    d_time::Float32 = 1f0 / 365f0,
    time_steps::Int = 200, space_steps::Int = 200
)::Vector{Float32}
    CUDA_AVAILABLE || error("CUDA is required: CPU implementation was removed.")
    _v_theta_decay_gpu(spots, strikes, tenors, sigmas, v_is_call,
        rates_curve, rates_times, div_amounts, div_times,
        d_time, time_steps, space_steps)
end

"""
   get_v_vega_decay_fd(...)
VegaDecay = dVega/dt.  Rate at which vega decays as expiry approaches.
"""
function get_v_vega_decay_fd(
    spots::Vector{Float32}, strikes::Vector{Float32},
    tenors::Vector{Float32}, sigmas::Vector{Float32},
    v_is_call::Vector{UInt8},
    rates_curve::Vector{Float32}, rates_times::Vector{Float32},
    div_amounts::Vector{Float32}, div_times::Vector{Float32};
    d_sigma::Float32 = 1f-3,
    d_time::Float32  = 1f0 / 365f0,
    time_steps::Int = 200, space_steps::Int = 200
)::Vector{Float32}
    CUDA_AVAILABLE || error("CUDA is required: CPU implementation was removed.")
    _v_vega_decay_gpu(spots, strikes, tenors, sigmas, v_is_call,
        rates_curve, rates_times, div_amounts, div_times,
        d_sigma, d_time, time_steps, space_steps)
end

"""
   get_v_vanna_fd(...)
Vanna = dΔ/dσ = dVega/dS.  Cross partial: 2D central finite difference.
"""
function get_v_vanna_fd(
    spots::Vector{Float32}, strikes::Vector{Float32},
    tenors::Vector{Float32}, sigmas::Vector{Float32},
    v_is_call::Vector{UInt8},
    rates_curve::Vector{Float32}, rates_times::Vector{Float32},
    div_amounts::Vector{Float32}, div_times::Vector{Float32};
    d_spot::Float32  = 1f-1,
    d_sigma::Float32 = 1f-3,
    time_steps::Int = 200, space_steps::Int = 200
)::Vector{Float32}
    CUDA_AVAILABLE || error("CUDA is required: CPU implementation was removed.")
    _v_vanna_gpu(spots, strikes, tenors, sigmas, v_is_call,
        rates_curve, rates_times, div_amounts, div_times,
        d_spot, d_sigma, time_steps, space_steps)
end

"""
    get_v_volga_fd(...)
Volga (Vomma) = d²P/dσ².  Central second difference over vol.
Measures the convexity of option price with respect to implied vol.
"""
function get_v_volga_fd(
    spots::Vector{Float32}, strikes::Vector{Float32},
    tenors::Vector{Float32}, sigmas::Vector{Float32},
    v_is_call::Vector{UInt8},
    rates_curve::Vector{Float32}, rates_times::Vector{Float32},
    div_amounts::Vector{Float32}, div_times::Vector{Float32};
    d_sigma::Float32 = 1f-3,
    time_steps::Int = 200, space_steps::Int = 200
)::Vector{Float32}
    CUDA_AVAILABLE || error("CUDA is required: CPU implementation was removed.")
    _v_volga_gpu(spots, strikes, tenors, sigmas, v_is_call,
        rates_curve, rates_times, div_amounts, div_times,
        d_sigma, time_steps, space_steps)
end

end # module PricingEngine
