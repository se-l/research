ENV["JULIA_PYTHONCALL_EXE"] = raw"C:\Users\seb\miniconda3\envs\py314\python.exe"

using Test
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "src", "PricingEngine.jl"))
using .PricingEngine

"""
Performance-only benchmark for GPU IV solver with varying tolerances/iterations.
Mirrors test_performance.cpp logic but only measures Julia GPU implementation.
"""

function generate_test_grid()
    # Match C++ test: diverse tenors and strikes
    tenors = Float32[0.05f0 + (2.0f0 - 0.05f0) * (i / 19.0f0) for i in 0:19]
    strikes = Float32[80.0f0 + (120.0f0 - 80.0f0) * (i / 500.0f0) for i in 0:500]
    rights = UInt8[0x01, 0x00]

    S = 100.0f0
    true_sigma = 0.30f0

    rates_curve = Float32[0.05f0]
    rates_times = Float32[1.0f0]
    div_amounts = Float32[1.20f0]
    div_times = Float32[0.5f0]

    return (; tenors, strikes, rights, S, true_sigma,
    rates_curve, rates_times, div_amounts, div_times)
end

function generate_target_prices(grid)
    spots   = Float32[]
    strikes = Float32[]
    tenors  = Float32[]
    rights  = UInt8[]

    for r in grid.rights, t in grid.tenors, k in grid.strikes
        push!(spots,   grid.S)
        push!(strikes, k)
        push!(tenors,  t)
        push!(rights,   r)
    end

    prices = get_v_price_fd(
        spots, strikes, tenors, fill(grid.true_sigma, length(spots)), rights,
        grid.rates_curve, grid.rates_times,
        grid.div_amounts, grid.div_times;
            time_steps=200, space_steps=200
    )

    v_spots   = Float32[]
    v_strikes = Float32[]
    v_tenors  = Float32[]
    v_rights  = UInt8[]
    v_prices  = Float32[]

    for i in eachindex(prices)
        price = prices[i]
        if price > 1f-4 && isfinite(price)
            push!(v_spots,   spots[i])
            push!(v_strikes, strikes[i])
            push!(v_tenors,  tenors[i])
            push!(v_rights,   rights[i])
            push!(v_prices,   price)
        end
    end

    return (; v_spots, v_strikes, v_tenors, v_rights, v_prices)
end

struct Config
    iv_tol::Float32
    max_iter::Int
    v_min::Float32
    v_max::Float32
    time_steps::Int
    space_steps::Int
    name::String
end

struct Result
    cfg::Config
    time_ms::Float64
    max_abs_err::Float32
    mean_abs_err::Float32
    non_finite::Int
    count::Int
end

function compute_metrics(ivs::Vector{Float32}, true_sigma::Float32)
    non_finite = count(!isfinite, ivs)
    finite_count = length(ivs) - non_finite

    if finite_count == 0
        return (
        max_abs_err = Inf32,
        mean_abs_err = Inf32,
        non_finite = non_finite,
        count = length(ivs)
        )
    end

    errors = [abs(iv - true_sigma) for iv in ivs if isfinite(iv)]

    return (
    max_abs_err = maximum(errors),
    mean_abs_err = mean(errors),
    non_finite = non_finite,
    count = length(ivs)
    )
end

function run_iv_benchmark(data, grid, cfg::Config)
    # Warmup run (force compilation)
    get_v_iv_fd(
        data.v_prices[1:min(100, length(data.v_prices))],
        data.v_spots[1:min(100, length(data.v_spots))],
        data.v_strikes[1:min(100, length(data.v_strikes))],
        data.v_tenors[1:min(100, length(data.v_tenors))],
        data.v_rights[1:min(100, length(data.v_rights))],
        grid.rates_curve, grid.rates_times,
        grid.div_amounts, grid.div_times;
        tol=cfg.iv_tol, max_iter=cfg.max_iter,
        v_min=cfg.v_min, v_max=cfg.v_max,
        time_steps=cfg.time_steps, space_steps=cfg.space_steps
    )

    # Timed run
    t0 = time_ns()
    ivs = get_v_iv_fd(
        data.v_prices, data.v_spots, data.v_strikes,
        data.v_tenors, data.v_rights,
        grid.rates_curve, grid.rates_times,
        grid.div_amounts, grid.div_times;
            tol=cfg.iv_tol, max_iter=cfg.max_iter,
            v_min=cfg.v_min, v_max=cfg.v_max,
            time_steps=cfg.time_steps, space_steps=cfg.space_steps
    )
    t1 = time_ns()

    time_ms = (t1 - t0) / 1e6

    metrics = compute_metrics(ivs, grid.true_sigma)

    return Result(
        cfg,
        time_ms,
        metrics.max_abs_err,
        metrics.mean_abs_err,
        metrics.non_finite,
        metrics.count
    )
end

@testset "Performance: GPU IV solver tolerance sweep" begin
    println("\nGenerating test grid...")
    grid = generate_test_grid()

    println("Generating target prices...")
    data = generate_target_prices(grid)
    println("Generated $(length(data.v_prices)) valid option prices")

    configs = [
#        Config(1f-4, 50,  1f-4, 5f0, 200, 200, "iv_tol=1e-4, it=50"),
#        Config(1f-4, 100, 1f-4, 5f0, 200, 200, "iv_tol=1e-4, it=100"),
#        Config(1f-5, 50,  1f-4, 5f0, 200, 200, "iv_tol=1e-5, it=50"),
#        Config(1f-5, 100, 1f-4, 5f0, 200, 200, "iv_tol=1e-5, it=100"),
#        Config(1f-6, 50,  1f-4, 5f0, 200, 200, "iv_tol=1e-6, it=50"),
        Config(1f-6, 100, 1f-4, 5f0, 200, 200, "iv_tol=1e-6, it=100"),
    ]

    results = Result[]
    for (i, cfg) in enumerate(configs)
        print("  [$i/$(length(configs))] $(cfg.name)... ")
        try
            result = run_iv_benchmark(data, grid, cfg)
            push!(results, result)
            println("✓ $(round(result.time_ms, digits=2)) ms")
        catch e
            println("✗ FAILED")
            @error "Benchmark failed" config=cfg.name exception=(e, catch_backtrace())
        end
    end

    @test !isempty(results)

    if !isempty(results)
        baseline_idx = argmax([r.time_ms for r in results])
        baseline = results[baseline_idx]
        allowed_worse = 1f-4
        viable = filter(r -> r.max_abs_err <= baseline.max_abs_err + allowed_worse, results)

        println("\n" * "="^80)
        println("Results Summary")
        println("="^80)

        println("\nBaseline (most expensive):")
        println("  Config: $(baseline.cfg.name)")
        println("  Time: $(round(baseline.time_ms, digits=2)) ms")
        println("  Max abs error: $(baseline.max_abs_err)")
        println("  Mean abs error: $(baseline.mean_abs_err)")
        println("  Non-finite: $(baseline.non_finite)/$(baseline.count)")

        println("\nAll configs:")
        for (i, r) in enumerate(results)
            is_baseline = (i == baseline_idx)
            is_viable = r in viable
            marker = is_baseline ? " [BASELINE]" : (is_viable ? " [VIABLE]" : "")
            println(@sprintf("  %2d. %-30s | %8.2f ms | n=%5d | max_err=%.6f | mean_err=%.6f | non_finite=%d%s",
            i, r.cfg.name, r.time_ms, r.count,
            r.max_abs_err, r.mean_abs_err, r.non_finite, marker))
        end

        println("\nViable configs (max_abs_err <= baseline + $(allowed_worse)):")
        if isempty(viable)
            println("  None found!")
        else
            for r in viable
                speedup = baseline.time_ms / r.time_ms
                println(@sprintf("  - %-30s | %8.2f ms (%.2fx) | max_err=%.6f | mean_err=%.6f",
                r.cfg.name, r.time_ms, speedup, r.max_abs_err, r.mean_abs_err))
            end
        end

        @test !isempty(viable)
    end

    println("\n" * "="^80)
end

run_performance_benchmark()

println("Finished test_performance.jl.")
