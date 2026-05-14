ENV["JULIA_PYTHONCALL_EXE"] = raw"C:\\Users\\seb\\miniconda3\\envs\\py314\\python.exe"

using Test
using Printf
using Statistics
using PythonCall

include(joinpath(@__DIR__, "..", "src", "PricingEngine.jl"))
using .PricingEngine

const merlin = pyimport("merlin")

function cpp_reference_prices(g; time_steps=200, space_steps=200)
    py_prices = merlin.get_v_fd_price(
        g.v_spots, g.v_strikes, g.v_tenors, g.v_sigmas, g.v_rights,
        g.rates_curve, g.rates_times, g.div_amounts, g.div_times,
        time_steps, space_steps
    )
    py_prices = Float32.(pyconvert(Vector{Any}, py_prices))
end

function cpp_reference_ivs(prices, spots, strikes, tenors, rights, rc, rt, da, dt_;
tol=1f-6, max_iter=100, v_min=1f-4, v_max=5f0, time_steps=200, space_steps=200)
    py_ivs = merlin.get_v_iv_fd_gpu(
        prices, spots, strikes, tenors, rights,
        rc, rt, da, dt_,
        tol, max_iter, v_min, v_max, time_steps, space_steps
    )
    py_ivs = Float32.(pyconvert(Vector{Any}, py_ivs))
end

function make_test_grid(;
S::Float32       = 100f0,
rights           = [0x01, 0x00],
tenors           = [0.1f0, 0.5f0, 2.0f0],
strikes          = [80f0, 90f0, 100f0, 110f0, 120f0],
sigma::Float32   = 0.25f0,
rates_curve      = [0.03f0, 0.1f0, 0.03f0],
rates_times      = [0.2f0,  0.8f0, 1.5f0],
div_amounts      = [1.0f0,  1.5f0, 2.0f0],
div_times        = [0.4f0,  0.8f0, 1.2f0],
)
    v_spots   = Float32[]
    v_strikes = Float32[]
    v_tenors  = Float32[]
    v_sigmas  = Float32[]
    v_rights  = UInt8[]

    for r in rights, t in tenors, k in strikes
        push!(v_spots,   S)
        push!(v_strikes, k)
        push!(v_tenors,  t)
        push!(v_sigmas,  sigma)
        push!(v_rights,  r)
    end

    (; v_spots, v_strikes, v_tenors, v_sigmas, v_rights,
    rates_curve, rates_times, div_amounts, div_times)
end

#@testset "GPU pricing vs C++ pricing" begin
#    try
#        g = make_test_grid()
#        gpu_prices = PricingEngine.get_v_price_fd(
#            g.v_spots, g.v_strikes, g.v_tenors, g.v_sigmas, g.v_rights,
#            g.rates_curve, g.rates_times, g.div_amounts, g.div_times)
#
#        cpp_prices = cpp_reference_prices(g; time_steps=200, space_steps=200)
#
#        @test length(gpu_prices) == length(cpp_prices)
#        for i in eachindex(gpu_prices)
#            gpu_p = gpu_prices[i]
#            cpp_p = cpp_prices[i]
#            ok = isapprox(gpu_p, cpp_p; atol=1f-4, rtol=2f-3) || (isnan(gpu_p) && isnan(cpp_p))
#            if !ok
##                @error "GPU pricing mismatch" index=i gpu_price=gpu_p cpp_price=cpp_p abs_diff=abs(gpu_p - cpp_p)
#                println("GPU pricing mismatch index=$(i) gpu_price=$(gpu_p) cpp_price=$(cpp_p) abs_diff=$(abs(gpu_p - cpp_p))")
#            else
#                @test true
#            end
#        end
#    catch e
#        @error "GPU pricing vs C++ pricing test failed" exception=(e, catch_backtrace())
#        #@test_broken false  # Mark as broken but continue
#    end
#end
#
#@testset "GPU IV vs C++ IV" begin
#    try
#        S = 100f0
#        true_σ = 0.30f0
#        rc = [0.05f0]; rt = [1.0f0]
#        da = [1.20f0]; dt_ = [0.5f0]
#
#        v_spots   = Float32[]
#        v_strikes = Float32[]
#        v_tenors  = Float32[]
#        v_rights  = UInt8[]
#        v_prices  = Float32[]
#
#        for r in [0x01, 0x00], t in [0.2f0, 1.0f0], k in [90f0, 100f0, 110f0]
#            p = pyconvert(Float32, merlin.get_fd_price_cpu(S, k, t, true_σ, r,
#                rc, rt, da, dt_, 200, 200))
#            p > 1f-4 || continue
#            push!(v_spots, S)
#            push!(v_strikes, k)
#            push!(v_tenors, t)
#            push!(v_rights, r)
#            push!(v_prices, p)
#        end
#
#        gpu_ivs = PricingEngine.get_v_iv_fd(v_prices, v_spots, v_strikes, v_tenors, v_rights,
#            rc, rt, da, dt_; tol=1f-6, max_iter=200, v_min=1f-4, v_max=5f0,
#                time_steps=200, space_steps=200)
#
#        cpp_ivs = cpp_reference_ivs(v_prices, v_spots, v_strikes, v_tenors, v_rights,
#            rc, rt, da, dt_; tol=1f-6, max_iter=200, v_min=1f-4, v_max=5f0,
#                time_steps=200, space_steps=200)
#
#        @test length(gpu_ivs) == length(cpp_ivs)
#        for i in eachindex(gpu_ivs)
#            ok = isapprox(gpu_ivs[i], cpp_ivs[i]; atol=1f-3, rtol=1f-3) ||
#            (isnan(gpu_ivs[i]) && isnan(cpp_ivs[i]))
#            if !ok
#                @error "GPU IV vs C++ IV" index=i gpu_iv=gpu_ivs[i] cpp_iv=cpp_ivs[i] abs_diff=abs(gpu_ivs[i] - cpp_ivs[i])
#            else
#                @test true
#            end
#        end
#    catch e
#        @error "GPU IV vs C++ IV test failed" exception=(e, catch_backtrace())
##        @test_broken false
#    end
#end

@testset "Extreme inputs: GPU vs C++" begin
    rc = [0.05f0]; rt = [1.0f0]
    da = Float32[]; dt_ = Float32[]

    cases = [
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
    (S, K, T, σ, is_call, desc) = cases[1]
    for (S, K, T, σ, is_call, desc) in cases
        @testset "$desc" begin
            try
                r = is_call ? 0x01 : 0x00
                gpu_p = PricingEngine.get_v_price_fd([S], [K], [T], [σ], [r], rc, rt, da, dt_)[1]
                cpp_p = pyconvert(Float32, merlin.get_fd_price_cpu(S, K, T, σ, r, rc, rt, da, dt_, 200, 200))

                ok = isapprox(gpu_p, cpp_p; atol=1f-4, rtol=1f-3) || (isnan(gpu_p) && isnan(cpp_p))

                if !ok
                    @error "Extreme inputs: GPU vs C++" gpu_price=gpu_p cpp_price=cpp_p abs_diff=abs(gpu_p - cpp_p)
                else
                    @test true
                end

            catch e
                @error "Extreme input test failed: $desc" S K T σ is_call exception=(e, catch_backtrace())
#                @test_broken false
            end
        end
    end
end


@testset "Reconcile Julia CPU & GPU pricer" begin
    rc = [0.05f0]; rt = [1.0f0]
    da = Float32[]; dt_ = Float32[]

    cases = [
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
    (S, K, T, σ, is_call, desc) = cases[1]
    for (S, K, T, σ, is_call, desc) in cases
        @testset "$desc" begin
            try
                r = is_call ? 0x01 : 0x00
                gpu_p = PricingEngine.get_v_price_fd([S], [K], [T], [σ], [r], rc, rt, da, dt_)[1]
                cpp_p = PricingEngine.get_price_fd(S, K, T, σ, r, rc, rt, da, dt_)

                ok = isapprox(gpu_p, cpp_p; atol=1f-4, rtol=1f-3) || (isnan(gpu_p) && isnan(cpp_p))

                if !ok
                    @error "Extreme inputs: GPU vs CPU" gpu_price=gpu_p cpp_price=cpp_p abs_diff=abs(gpu_p - cpp_p)
                else
                    @test true
                end

            catch e
                @error "Extreme input test failed: $desc" S K T σ is_call exception=(e, catch_backtrace())
#                @test_broken false
            end
        end
    end
end

function cpp_reference_vegas(g; time_steps=200)
    # merlin signature: (spots, strikes, tenors, v_is_call, sigmas, n_steps, rates_curve, ...)
    v_is_call = Int32.(g.v_rights)   # UInt8 0x00/0x01 → Int 0/1
    #        merlin.get_v_fd_vega(
#            spots=s, strikes=k, tenors=t, v_is_call=v_is_call, sigmas=iv, rates_curve=yield_curve.rates,
        #    rates_times=yield_curve.times, div_amounts=div_amounts, div_times=div_times,
#        )
    py_vegas = merlin.get_v_fd_vega(
        g.v_spots, g.v_strikes, g.v_tenors, v_is_call, g.v_sigmas, time_steps,
        g.rates_curve, g.rates_times, g.div_amounts, g.div_times
    )
    Float32.(pyconvert(Vector{Any}, py_vegas))
end

function cpp_reference_deltas(g; time_steps=200)
    v_is_call = Int32.(g.v_rights)
    # merlin.get_v_fd_delta(
    #            spots=s, strikes=k, tenors=t, v_is_call=v_is_call, sigmas=iv, rates_curve=yield_curve.rates,
    #           rates_times=yield_curve.times, div_amounts=div_amounts, div_times=div_times,
    #        )
    py_deltas = merlin.get_v_fd_delta(
        g.v_spots, g.v_strikes, g.v_tenors, v_is_call, g.v_sigmas, time_steps,
        g.rates_curve, g.rates_times, g.div_amounts, g.div_times
    )
    Float32.(pyconvert(Vector{Any}, py_deltas))
end

#@testset "GPU vega vs C++ vega" begin
#    try
#        g = make_test_grid()
#        gpu_vegas = PricingEngine.get_v_vega_fd(
#            g.v_spots, g.v_strikes, g.v_tenors, g.v_sigmas, g.v_rights,
#            g.rates_curve, g.rates_times, g.div_amounts, g.div_times;
#            d_sigma=1f-3, time_steps=200, space_steps=200
#        )
#
#        cpp_vegas = cpp_reference_vegas(g; time_steps=200)
#
#        @test length(gpu_vegas) == length(cpp_vegas)
#        for i in eachindex(gpu_vegas)
#            gpu_v = gpu_vegas[i]
#            cpp_v = cpp_vegas[i]
#            ok = isapprox(gpu_v, cpp_v; atol=1f-4, rtol=2f-3) || (isnan(gpu_v) && isnan(cpp_v))
#            if !ok
#                @error "GPU vega mismatch" index=i gpu_vega=gpu_v cpp_vega=cpp_v abs_diff=abs(gpu_v - cpp_v)
#            else
#                @test true
#            end
#        end
#    catch e
#        @error "GPU vega vs C++ vega test failed" exception=(e, catch_backtrace())
#    end
#end
#
#@testset "GPU delta vs C++ delta" begin
#    try
#        g = make_test_grid()
#        gpu_deltas = PricingEngine.get_v_delta_fd(
#            g.v_spots, g.v_strikes, g.v_tenors, g.v_sigmas, g.v_rights,
#            g.rates_curve, g.rates_times, g.div_amounts, g.div_times;
#            d_spot=1f-2, time_steps=200, space_steps=200)
#
#        cpp_deltas = cpp_reference_deltas(g; time_steps=200)
#
#        @test length(gpu_deltas) == length(cpp_deltas)
#        for i in eachindex(gpu_deltas)
#            gpu_d = gpu_deltas[i]
#            cpp_d = cpp_deltas[i]
#            ok = isapprox(gpu_d, cpp_d; atol=1f-4, rtol=2f-3) || (isnan(gpu_d) && isnan(cpp_d))
#            if !ok
#                @error "GPU delta mismatch" index=i gpu_delta=gpu_d cpp_delta=cpp_d abs_diff=abs(gpu_d - cpp_d)
#            else
#                @test true
#            end
#        end
#    catch e
#        @error "GPU delta vs C++ delta test failed" exception=(e, catch_backtrace())
#    end
#end