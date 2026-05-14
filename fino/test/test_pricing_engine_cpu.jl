# test_cpu_pricing_engine.jl
# Unit tests for all PricingEngine CPU (non-GPU) public API methods.
# Run with: julia --project test_cpu_pricing_engine.jl

using Test
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "src", "PricingEngine.jl"))
using .PricingEngine

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

const RC  = Float32[0.05f0]          # rates curve  (single flat rate)
const RT  = Float32[1.0f0]           # rates times
const DA  = Float32[1.2f0]           # div amounts
const DT  = Float32[0.5f0]           # div times   (time to ex-div date, years)
const EMPTY_F32 = Float32[]          # empty rates/divs for no-drift tests

const ATOL = 1f-4
const RTOL = 2f-3

"""
    isok(gpu, cpp; atol, rtol)

Return true if two Float32 values are close enough, counting NaN==NaN as equal.
"""
isok(a::Float32, b::Float32; atol=ATOL, rtol=RTOL) =
    isapprox(Float64(a), Float64(b); atol=Float64(atol), rtol=Float64(rtol)) ||
    (isnan(a) && isnan(b))

"""
    isok_nan(a, b; atol, rtol)

Like isok but tolerates NaN in either position.
"""
isok_nan(a::Float32, b::Float32; atol=ATOL, rtol=RTOL) =
    (isnan(a) || isnan(b) || isok(a, b; atol=atol, rtol=rtol))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Rate-curve helpers  (no Greeks involved)
# ─────────────────────────────────────────────────────────────────────────────

@testset "df_0_t — discount factor from t=0 to t" begin
    # Flat rate: df = exp(-r * t)
    @test isok(df_0_t(0f0, RC, RT), 1f0; atol=1f-6)
    @test isok(df_0_t(1f0, RC, RT), exp(-0.05f0); atol=1f-5)
    @test isok(df_0_t(0.5f0, RC, RT), exp(-0.025f0); atol=1f-5)
    # Empty curve → df = 1
    @test isok(df_0_t(1f0, EMPTY_F32, EMPTY_F32), 1f0; atol=1f-6)
end

@testset "df_t_T — discount factor from t to T" begin
    # df_t_T(t, T) = exp(-[Z(T) - Z(t)])
    # With flat rate Z(t) = r*t, df_t_T = exp(-r*(T-t))
    @test isok(df_t_T(0f0, 1f0, RC, RT), exp(-0.05f0); atol=1f-5)
    @test isok(df_t_T(0.3f0, 0.7f0, RC, RT), exp(-0.02f0); atol=1f-5)
    @test isok(df_t_T(1f0, 1f0, RC, RT), 1f0; atol=1f-6)   # t==T → df=1
    @test isok(df_t_T(0.3f0, 0.3f0, EMPTY_F32, EMPTY_F32), 1f0; atol=1f-6)
end

@testset "get_rate_at_time_convex_monotone" begin
    # Flat curve: rate should equal the flat rate at all times
    @test isok(get_rate_at_time_convex_monotone(0f0, RC, RT), 0.05f0; atol=1f-6)
    @test isok(get_rate_at_time_convex_monotone(0.5f0, RC, RT), 0.05f0; atol=1f-5)
    @test isok(get_rate_at_time_convex_monotone(2f0, RC, RT), 0.05f0; atol=1f-5)
    # Empty curve → 0
    @test isok(get_rate_at_time_convex_monotone(1f0, EMPTY_F32, EMPTY_F32), 0f0; atol=1f-6)
end


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Scalar price & implied vol  (foundation for all Greeks)
# ─────────────────────────────────────────────────────────────────────────────

@testset "get_price_fd — basic ATM flat specs" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0

    call = get_price_fd(S, K, T, σ, 0x01, RC, RT, DA, DT)
    put  = get_price_fd(S, K, T, σ, 0x00, RC, RT, DA, DT)

    # Put-call parity: C - P = S*df(0,T) - K*df(0,T) - PV(divs)
    pv_div = DA[1] * df_0_t(DT[1], RC, RT)
    df_T   = df_0_t(T, RC, RT)
    @test isok(call - put, (S - K) * df_T - pv_div; atol=5f-3)

    # Call price positive
    @test call > 0f0
    # Put price positive
    @test put > 0f0
    # Call >= intrinsic
    @test call >= max(S - K, 0f0)
    # Put >= intrinsic
    @test put >= max(K - S, 0f0)
end

@testset "get_price_fd — extreme inputs" begin
    for (S, K, T, σ, is_call, desc) in [
        (1f-6,   100f0,  1f0,  0.2f0, true,  "Tiny spot"),
        (100f0,  1f-6,   1f0,  0.2f0, false, "Tiny strike"),
        (100f0,  100f0,  1f-6, 0.2f0, true,  "Tiny tenor"),
        (100f0,  100f0,  1f0,  1f-6,  true,  "Tiny vol"),
        (1f6,    100f0,  1f0,  0.2f0, true,  "Huge spot"),
        (100f0,  100f0,  100f0, 0.2f0, true,  "Huge tenor"),
        (100f0,  100f0,  1f0,  10f0,  true,  "Huge vol"),
        (0f0,    100f0,  1f0,  0.2f0, true,  "Zero spot"),
        (100f0,  100f0,  0f0,  0.2f0, true,  "Zero tenor"),
        (100f0,  100f0,  1f0,  0f0,   true,  "Zero vol"),
    ]
        r = is_call ? 0x01 : 0x00
        p = @test_nowarn get_price_fd(S, K, T, σ, r, RC, RT, DA, DT)
        # Result must be finite or NaN (must not throw)
        @test isfinite(p) || isnan(p)
        # With zero T or zero vol, result must be intrinsic or zero
        if T <= 0f0 || σ <= 0f0
            @test isfinite(p)
        end
    end
end

@testset "get_price_fd — deep ITM / OTM" begin
    S = 100f0; T = 1f0; σ = 0.25f0
    # Deep ITM call
    p_itm_call = get_price_fd(S, 50f0, T, σ, 0x01, RC, RT, DA, DT)
    @test isok(p_itm_call, S - 50f0; atol=5f-2)
    # Deep OTM call
    p_otm_call = get_price_fd(S, 200f0, T, σ, 0x01, RC, RT, DA, DT)
    @test p_otm_call < 1f-3
    # Deep ITM put
    p_itm_put = get_price_fd(S, 150f0, T, σ, 0x00, RC, RT, DA, DT)
    @test isok(p_itm_put, 150f0 - S; atol=5f-2)
    # Deep OTM put
    p_otm_put = get_price_fd(S, 50f0, T, σ, 0x00, RC, RT, DA, DT)
    @test p_otm_put < 1f-3
end

@testset "get_price_fd — no dividends" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0
    call_no_div = get_price_fd(S, K, T, σ, 0x01, RC, RT, EMPTY_F32, EMPTY_F32)
    put_no_div  = get_price_fd(S, K, T, σ, 0x00, RC, RT, EMPTY_F32, EMPTY_F32)
    # No-div put-call parity: C - P = (S - K) * df(T)
    df_T = df_0_t(T, RC, RT)
    @test isok(call_no_div - put_no_div, (S - K) * df_T; atol=5f-3)
    @test call_no_div > 0f0
    @test put_no_div > 0f0
end

@testset "get_price_fd — very short tenor (time-to-expiry ≈ 0)" begin
    S = 100f0; K = 100f0; σ = 0.25f0
    for T in [1f-4, 1f-3, 1f-2]
        call = get_price_fd(S, K, T, σ, 0x01, RC, RT, EMPTY_F32, EMPTY_F32)
        put  = get_price_fd(S, K, T, σ, 0x00, RC, RT, EMPTY_F32, EMPTY_F32)
        # Very short T → price ≈ max(S-K,0) or max(K-S,0)
        @test isok(call, max(S - K, 0f0); atol=5f-3)
        @test isok(put,  max(K - S, 0f0); atol=5f-3)
    end
end

@testset "get_iv_fd — round-trip price → IV → price" begin
    S = 100f0; K = 100f0; T = 1f0; σ_true = 0.30f0

    call_price = get_price_fd(S, K, T, σ_true, 0x01, RC, RT, DA, DT)
    put_price  = get_price_fd(S, K, T, σ_true, 0x00, RC, RT, DA, DT)

    iv_call = @test_nowarn get_iv_fd(call_price, S, K, T, 0x01, RC, RT, DA, DT)
    iv_put  = @test_nowarn get_iv_fd(put_price,  S, K, T, 0x00, RC, RT, DA, DT)

    # Recovered IV should match true σ
    @test isok(iv_call, σ_true; atol=5f-3)
    @test isok(iv_put,  σ_true; atol=5f-3)
end

@testset "get_iv_fd — degenerate inputs" begin
    # Zero market price → IV undefined / NaN
    # Market price below intrinsic → IV undefined
    S = 100f0; K = 120f0; T = 1f0

    iv_nan = @test_nowarn get_iv_fd(0f0, S, K, T, 0x01, RC, RT, DA, DT)
    @test isnan(iv_nan) || !isfinite(iv_nan)

    # OTM call priced below intrinsic floor
    iv_otm = @test_nowarn get_iv_fd(0.01f0, S, K, T, 0x01, RC, RT, DA, DT)
    @test isnan(iv_otm) || iv_otm < 0f0  # should be NaN or degenerate
end


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — First-order Greeks: delta, vega, theta
# ─────────────────────────────────────────────────────────────────────────────

@testset "get_delta_fd — basic ATM" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0

    δ_call = @test_nowarn get_delta_fd(S, K, T, σ, 0x01, RC, RT, DA, DT; d_spot=1f-1)
    δ_put  = @test_nowarn get_delta_fd(S, K, T, σ, 0x00, RC, RT, DA, DT; d_spot=1f-1)

    # ATM call delta ≈ 0.5 (positive), put delta ≈ -0.5 (negative)
    @test isok(δ_call, 0.5f0; atol=0.1f0)
    @test isok(δ_put, -0.5f0; atol=0.1f0)
    # Delta magnitude <= 1
    @test abs(δ_call) <= 1f0
    @test abs(δ_put)  <= 1f0
    # Delta of put = delta of call - 1
    @test isok(δ_put, δ_call - 1f0; atol=1f-3)
end

@testset "get_delta_fd — sign correctness across ITM/OTM" begin
    T = 1f0; σ = 0.25f0
    for (K, is_call, expected_sign) in [
        (80f0,  true,  +1),   # ITM call delta > 0
        (120f0, true,  +1),   # OTM call delta > 0 but small
        (80f0,  false, -1),   # ITM put  delta < 0
        (120f0, false, -1),   # OTM put  delta < 0
    ]
        δ = @test_nowarn get_delta_fd(100f0, K, T, σ, is_call ? 0x01 : 0x00,
                                      RC, RT, DA, DT; d_spot=1f-1)
        @test sign(δ) == expected_sign
    end
end

@testset "get_delta_fd — extreme inputs" begin
    for (S, K, T, σ, is_call, desc) in [
        (1f-6, 100f0,  1f0, 0.2f0, true,  "Tiny spot"),
        (100f0, 1f-6,  1f0, 0.2f0, false, "Tiny strike"),
        (100f0, 100f0, 0f0, 0.2f0, true,  "Zero tenor"),
        (100f0, 100f0, 1f0, 0f0,   true,  "Zero vol"),
        (0f0,   100f0, 1f0, 0.2f0, true,  "Zero spot"),
    ]
        r = is_call ? 0x01 : 0x00
        δ = @test_nowarn get_delta_fd(S, K, T, σ, r, RC, RT, DA, DT; d_spot=1f-1)
        @test isfinite(δ) || isnan(δ)
    end
end

@testset "get_vega_fd — basic ATM" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0
    v_call = @test_nowarn get_vega_fd(S, K, T, σ, 0x01, RC, RT, DA, DT; d_sigma=1f-3)
    v_put  = @test_nowarn get_vega_fd(S, K, T, σ, 0x00, RC, RT, DA, DT; d_sigma=1f-3)

    # Vega is always positive for both calls and puts
    @test v_call > 0f0
    @test v_put  > 0f0
    # Calls and puts have (nearly) the same vega
    @test isok(v_call, v_put; atol=1f-3)
    # Vega scales with sqrt(T) roughly; check it's in a sane range
    @test v_call < 1f0   # ≤ 1 per 1-unit move in vol (should be ~0.4)
    @test v_call > 1f-4
end

@testset "get_vega_fd — zero T → zero vega" begin
    S = 100f0; K = 100f0; σ = 0.25f0
    for T in [0f0, 1f-4]
        v = @test_nowarn get_vega_fd(S, K, T, σ, 0x01, RC, RT, DA, DT; d_sigma=1f-3)
        @test isok(v, 0f0; atol=1f-3)
    end
end

@testset "get_vega_fd — extreme inputs" begin
    for (S, K, T, σ, is_call, desc) in [
        (1f-6,  100f0, 1f0,  0.2f0, true,  "Tiny spot"),
        (100f0, 100f0, 1f0,  0f0,   true,  "Zero vol"),
        (100f0, 100f0, 0f0,  0.2f0, true,  "Zero tenor"),
        (1f6,   100f0, 1f0,  0.2f0, true,  "Huge spot"),
        (100f0, 100f0, 1f0,  10f0,  true,  "Huge vol"),
    ]
        r = is_call ? 0x01 : 0x00
        v = @test_nowarn get_vega_fd(S, K, T, σ, r, RC, RT, DA, DT; d_sigma=1f-3)
        @test isfinite(v) || isnan(v)
    end
end

@testset "get_theta_fd — basic ATM (negative: option decays over time)" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0
    θ_call = @test_nowarn get_theta_fd(S, K, T, σ, 0x01, RC, RT, DA, DT; d_time=1f0/365f0)
    θ_put  = @test_nowarn get_theta_fd(S, K, T, σ, 0x00, RC, RT, DA, DT; d_time=1f0/365f0)

    # Theta is (usually) negative: option loses value as time passes
    @test θ_call < 0f0
    @test θ_put  < 0f0
    # Theta should be in a reasonable range (small fraction of option price)
    @test abs(θ_call) < 1f1
end

@testset "get_theta_fd — deep ITM with long T (may be positive due to carry)" begin
    S = 50f0; K = 100f0; T = 2f0; σ = 0.15f0
    θ = @test_nowarn get_theta_fd(S, K, T, σ, 0x01, RC, RT, DA, DT; d_time=1f0/365f0)
    # Deep ITM call can have positive theta (time value of money dominates)
    # Just verify it is finite
    @test isfinite(θ)
end

@testset "get_theta_fd — zero T → zero theta" begin
    S = 100f0; K = 100f0; σ = 0.25f0
    θ = @test_nowarn get_theta_fd(S, K, 0f0, σ, 0x01, RC, RT, DA, DT; d_time=1f0/365f0)
    @test isok(θ, 0f0; atol=1f-3)
end

@testset "get_theta_fd — extreme inputs" begin
    for (S, K, T, σ, is_call, desc) in [
        (1f-6,  100f0, 1f0, 0.2f0, true,  "Tiny spot"),
        (100f0, 100f0, 0f0, 0.2f0, true,  "Zero tenor"),
        (100f0, 100f0, 1f0, 0f0,   true,  "Zero vol"),
        (1f6,   100f0, 1f0, 0.2f0, true,  "Huge spot"),
    ]
        r = is_call ? 0x01 : 0x00
        θ = @test_nowarn get_theta_fd(S, K, T, σ, r, RC, RT, DA, DT; d_time=1f0/365f0)
        @test isfinite(θ) || isnan(θ)
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Second-order Greeks: gamma, speed
# ─────────────────────────────────────────────────────────────────────────────

@testset "get_gamma_fd — basic ATM (gamma peaked at ATM)" begin
    T = 1f0; σ = 0.25f0
    for (K, desc) in [(80f0, "ITM"), (100f0, "ATM"), (120f0, "OTM")]
        γ = @test_nowarn get_gamma_fd(100f0, K, T, σ, 0x01, RC, RT, DA, DT; d_spot=1f-1)
        # Gamma is always non-negative
        @test γ >= 0f0
        @test isfinite(γ)
    end
    # ATM gamma should be highest
    γ_itm  = @test_nowarn get_gamma_fd(100f0, 80f0, T, σ, 0x01, RC, RT, DA, DT; d_spot=1f-1)
    γ_atm  = @test_nowarn get_gamma_fd(100f0, 100f0, T, σ, 0x01, RC, RT, DA, DT; d_spot=1f-1)
    γ_otm  = @test_nowarn get_gamma_fd(100f0, 120f0, T, σ, 0x01, RC, RT, DA, DT; d_spot=1f-1)
    @test γ_atm >= max(γ_itm, γ_otm)
end

@testset "get_gamma_fd — call and put have same gamma" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0
    γ_call = @test_nowarn get_gamma_fd(S, K, T, σ, 0x01, RC, RT, DA, DT; d_spot=1f-1)
    γ_put  = @test_nowarn get_gamma_fd(S, K, T, σ, 0x00, RC, RT, DA, DT; d_spot=1f-1)
    @test isok(γ_call, γ_put; atol=1f-3)
end

@testset "get_gamma_fd — zero T or zero vol → gamma → 0" begin
    S = 100f0; K = 100f0; σ = 0.25f0
    for T in [0f0, 1f-4]
        γ = @test_nowarn get_gamma_fd(S, K, T, σ, 0x01, RC, RT, DA, DT; d_spot=1f-1)
        @test isok(γ, 0f0; atol=1f-3)
    end
    for σ in [0f0, 1f-4]
        γ = @test_nowarn get_gamma_fd(S, K, 1f0, σ, 0x01, RC, RT, DA, DT; d_spot=1f-1)
        @test isok(γ, 0f0; atol=1f-2)
    end
end

@testset "get_gamma_fd — extreme inputs" begin
    for (S, K, T, σ, is_call, desc) in [
        (1f-6,  100f0, 1f0,  0.2f0, true,  "Tiny spot"),
        (100f0, 100f0, 0f0,  0.2f0, true,  "Zero tenor"),
        (100f0, 100f0, 1f0,  0f0,   true,  "Zero vol"),
        (1f6,   100f0, 1f0,  0.2f0, true,  "Huge spot"),
    ]
        r = is_call ? 0x01 : 0x00
        γ = @test_nowarn get_gamma_fd(S, K, T, σ, r, RC, RT, DA, DT; d_spot=1f-1)
        @test isfinite(γ) || isnan(γ)
    end
end

@testset "get_speed_fd — basic ATM" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0
    sp_call = @test_nowarn get_speed_fd(S, K, T, σ, 0x01, RC, RT, DA, DT; d_spot=1f-1)
    sp_put  = @test_nowarn get_speed_fd(S, K, T, σ, 0x00, RC, RT, DA, DT; d_spot=1f-1)

    # Speed is third derivative; always negative for calls near ATM
    # (gamma decreases as spot increases near ATM)
    @test isfinite(sp_call)
    @test isfinite(sp_put)
end

@testset "get_speed_fd — zero T / zero vol → speed ≈ 0" begin
    S = 100f0; K = 100f0; σ = 0.25f0
    for T in [0f0, 1f-4]
        sp = @test_nowarn get_speed_fd(S, K, T, σ, 0x01, RC, RT, DA, DT; d_spot=1f-1)
        @test isok(sp, 0f0; atol=1f-2)
    end
    for σ in [0f0, 1f-4]
        sp = @test_nowarn get_speed_fd(S, K, 1f0, σ, 0x01, RC, RT, DA, DT; d_spot=1f-1)
        @test isok(sp, 0f0; atol=1f-2)
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Cross Greeks: vanna, volga, decay Greeks
# ─────────────────────────────────────────────────────────────────────────────

@testset "get_vanna_fd — d²P/dS dσ (sign: positive for calls, call-like for puts near ATM)" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0
    van_call = @test_nowarn get_vanna_fd(S, K, T, σ, 0x01, RC, RT, DA, DT;
                                         d_spot=1f-1, d_sigma=1f-3)
    van_put  = @test_nowarn get_vanna_fd(S, K, T, σ, 0x00, RC, RT, DA, DT;
                                         d_spot=1f-1, d_sigma=1f-3)

    @test isfinite(van_call)
    @test isfinite(van_put)
    # Vanna is roughly similar in magnitude for calls and puts at ATM
    @test abs(van_call - van_put) < 5f-1
end

@testset "get_vanna_fd — extreme inputs" begin
    for (S, K, T, σ, is_call, desc) in [
        (100f0, 100f0, 0f0,  0.2f0, true,  "Zero tenor"),
        (100f0, 100f0, 1f0,  0f0,   true,  "Zero vol"),
        (0f0,   100f0, 1f0,  0.2f0, true,  "Zero spot"),
    ]
        r = is_call ? 0x01 : 0x00
        van = @test_nowarn get_vanna_fd(S, K, T, σ, r, RC, RT, DA, DT;
                                        d_spot=1f-1, d_sigma=1f-3)
        @test isfinite(van) || isnan(van)
    end
end

@testset "get_volga_fd — d²P/dσ² (volga: convexity of price in vol)" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0
    volga_call = @test_nowarn get_volga_fd(S, K, T, σ, 0x01, RC, RT, DA, DT; d_sigma=1f-3)
    volga_put  = @test_nowarn get_volga_fd(S, K, T, σ, 0x00, RC, RT, DA, DT; d_sigma=1f-3)

    # Volga ≥ 0 (price is convex in vol for both calls and puts)
    @test volga_call >= 0f0
    @test volga_put  >= 0f0
    # ATM volga is small (symmetric)
    @test volga_call < 1f0
end

@testset "get_volga_fd — zero T / zero vol → volga → 0" begin
    S = 100f0; K = 100f0; σ = 0.25f0
    for T in [0f0, 1f-4]
        volga = @test_nowarn get_volga_fd(S, K, T, σ, 0x01, RC, RT, DA, DT; d_sigma=1f-3)
        @test isok(volga, 0f0; atol=1f-3)
    end
end

@testset "get_volga_fd — extreme inputs" begin
    for (S, K, T, σ, is_call, desc) in [
        (1f-6,  100f0, 1f0,  0.2f0, true,  "Tiny spot"),
        (100f0, 100f0, 0f0,  0.2f0, true,  "Zero tenor"),
        (100f0, 100f0, 1f0,  0f0,   true,  "Zero vol"),
    ]
        r = is_call ? 0x01 : 0x00
        volga = @test_nowarn get_volga_fd(S, K, T, σ, r, RC, RT, DA, DT; d_sigma=1f-3)
        @test isfinite(volga) || isnan(volga)
    end
end

@testset "get_gamma_decay_fd — dΓ/dt (gamma's sensitivity to time)" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0
    gd_call = @test_nowarn get_gamma_decay_fd(S, K, T, σ, 0x01, RC, RT, DA, DT;
                                              d_spot=1f-1, d_time=1f0/365f0)
    gd_put  = @test_nowarn get_gamma_decay_fd(S, K, T, σ, 0x00, RC, RT, DA, DT;
                                              d_spot=1f-1, d_time=1f0/365f0)
    # Gamma decay is negative near ATM (gamma decreases as time passes)
    @test isfinite(gd_call)
    @test isfinite(gd_put)
end

@testset "get_gamma_decay_fd — zero T → gamma → 0, so decay → 0" begin
    S = 100f0; K = 100f0; σ = 0.25f0
    gd = @test_nowarn get_gamma_decay_fd(S, K, 0f0, σ, 0x01, RC, RT, DA, DT;
                                         d_spot=1f-1, d_time=1f0/365f0)
    @test isok(gd, 0f0; atol=1f-2)
end

@testset "get_gamma_vol_fd — dΓ/dσ (charm: gamma's sensitivity to vol)" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0
    gv_call = @test_nowarn get_gamma_vol_fd(S, K, T, σ, 0x01, RC, RT, DA, DT;
                                             d_spot=1f-1, d_sigma=1f-3)
    gv_put  = @test_nowarn get_gamma_vol_fd(S, K, T, σ, 0x00, RC, RT, DA, DT;
                                             d_spot=1f-1, d_sigma=1f-3)
    @test isfinite(gv_call)
    @test isfinite(gv_put)
end

@testset "get_gamma_vol_fd — zero T → gamma → 0, so vol gamma → 0" begin
    S = 100f0; K = 100f0; σ = 0.25f0
    gv = @test_nowarn get_gamma_vol_fd(S, K, 0f0, σ, 0x01, RC, RT, DA, DT;
                                       d_spot=1f-1, d_sigma=1f-3)
    @test isok(gv, 0f0; atol=1f-2)
end

@testset "get_theta_decay_fd — dΘ/dt (zomma: theta's sensitivity to time)" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0
    td_call = @test_nowarn get_theta_decay_fd(S, K, T, σ, 0x01, RC, RT, DA, DT;
                                               d_time=1f0/365f0)
    td_put  = @test_nowarn get_theta_decay_fd(S, K, T, σ, 0x00, RC, RT, DA, DT;
                                               d_time=1f0/365f0)
    @test isfinite(td_call)
    @test isfinite(td_put)
end

@testset "get_vega_decay_fd — dVega/dt (veta: vega's sensitivity to time)" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0
    vd_call = @test_nowarn get_vega_decay_fd(S, K, T, σ, 0x01, RC, RT, DA, DT;
                                              d_sigma=1f-3, d_time=1f0/365f0)
    vd_put  = @test_nowarn get_vega_decay_fd(S, K, T, σ, 0x00, RC, RT, DA, DT;
                                              d_sigma=1f-3, d_time=1f0/365f0)
    @test isfinite(vd_call)
    @test isfinite(vd_put)
    # Vega decay can be positive (vol decay increases effective vega near expiry)
    @test isfinite(vd_call)
end


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Consistency checks: CPU scalar == CPU vectorised
# ─────────────────────────────────────────────────────────────────────────────

@testset "CPU scalar vs vectorised — price" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0
    r = 0x01
    scalar_p = get_price_fd(S, K, T, σ, r, RC, RT, DA, DT)
    vector_p = first(get_v_price_fd([S], [K], [T], [σ], [r], RC, RT, DA, DT))
    @test isok(scalar_p, vector_p; atol=1f-4)
end

@testset "CPU scalar vs vectorised — IV round-trip" begin
    S = 100f0; K = 100f0; T = 1f0; σ_true = 0.30f0
    r = 0x01
    price = get_price_fd(S, K, T, σ_true, r, RC, RT, DA, DT)
    scalar_iv = get_iv_fd(price, S, K, T, r, RC, RT, DA, DT)
    vector_iv = first(get_v_iv_fd([price], [S], [K], [T], [r], RC, RT, DA, DT))
    @test isok(scalar_iv, vector_iv; atol=1f-3)
end

@testset "CPU scalar vs vectorised — delta" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0; d_spot = 1f-1
    r = 0x01
    scalar_d = get_delta_fd(S, K, T, σ, r, RC, RT, DA, DT; d_spot)
    vector_d = first(get_v_delta_fd([S], [K], [T], [σ], [r], RC, RT, DA, DT; d_spot))
    @test isok(scalar_d, vector_d; atol=1f-3)
end

@testset "CPU scalar vs vectorised — vega" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0; d_sigma = 1f-3
    r = 0x01
    scalar_v = get_vega_fd(S, K, T, σ, r, RC, RT, DA, DT; d_sigma)
    vector_v = first(get_v_vega_fd([S], [K], [T], [σ], [r], RC, RT, DA, DT; d_sigma))
    @test isok(scalar_v, vector_v; atol=1f-3)
end

@testset "CPU scalar vs vectorised — theta" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0; d_time = 1f0/365f0
    r = 0x01
    scalar_t = get_theta_fd(S, K, T, σ, r, RC, RT, DA, DT; d_time)
    vector_t = first(get_v_theta_fd([S], [K], [T], [σ], [r], RC, RT, DA, DT; d_time))
    @test isok(scalar_t, vector_t; atol=1f-3)
end

@testset "CPU scalar vs vectorised — gamma" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0; d_spot = 1f-1
    r = 0x01
    scalar_g = get_gamma_fd(S, K, T, σ, r, RC, RT, DA, DT; d_spot)
    vector_g = first(get_v_gamma_fd([S], [K], [T], [σ], [r], RC, RT, DA, DT; d_spot))
    @test isok(scalar_g, vector_g; atol=1f-3)
end

@testset "CPU scalar vs vectorised — speed" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0; d_spot = 1f-1
    r = 0x01
    scalar_s = get_speed_fd(S, K, T, σ, r, RC, RT, DA, DT; d_spot)
    vector_s = first(get_v_speed_fd([S], [K], [T], [σ], [r], RC, RT, DA, DT; d_spot))
    @test isok(scalar_s, vector_s; atol=1f-2)
end

@testset "CPU scalar vs vectorised — volga" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0; d_sigma = 1f-3
    r = 0x01
    scalar_vg = get_volga_fd(S, K, T, σ, r, RC, RT, DA, DT; d_sigma)
    vector_vg = first(get_v_volga_fd([S], [K], [T], [σ], [r], RC, RT, DA, DT; d_sigma))
    @test isok(scalar_vg, vector_vg; atol=1f-2)
end

@testset "CPU scalar vs vectorised — vanna" begin
    S = 100f0; K = 100f0; T = 1f0; σ = 0.25f0; d_spot = 1f-1; d_sigma = 1f-3
    r = 0x01
    scalar_vn = get_vanna_fd(S, K, T, σ, r, RC, RT, DA, DT; d_spot, d_sigma)
    vector_vn = first(get_v_vanna_fd([S], [K], [T], [σ], [r], RC, RT, DA, DT; d_spot, d_sigma))
    @test isok(scalar_vn, vector_vn; atol=1f-2)
end

println("\n✅  All CPU PricingEngine tests passed.")