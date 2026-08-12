import time
from dataclasses import dataclass
from datetime import date, timedelta
import numpy as np

from options.types.option import get_price_jl, get_price_cuda
from shared.yield_curve import YieldCurve


# assume these already exist in your codebase
# from your_module import get_price_jl, get_price_cuda, YieldCurve, Dividend, OptionRight

@dataclass
class BenchmarkResult:
    name: str
    seconds: float
    per_contract_us: float

def build_test_data(n: int, calculation_date: date):
    rng = np.random.default_rng(42)

    # Spots around a broad range
    s = rng.uniform(20.0, 300.0, size=n).astype(np.float64)
    # Strikes near spots, but with some dispersion
    k = (s * rng.uniform(0.7, 1.3, size=n)).astype(np.float64)
    # Tenors in years
    t = rng.uniform(15 / 365.0, 730 / 365.0, size=n).astype(np.float64)
    # IVs
    iv = rng.uniform(0.10, 0.80, size=n).astype(np.float64)
    # Calls / puts
    v_is_call = rng.integers(0, 2, size=n, dtype=np.int32)

    return s, k, t, v_is_call, iv

def time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return out, elapsed

def benchmark_pricers(n=10_000):
    calculation_date = date(2024, 6, 27)
    yield_curve = YieldCurve().get_zero_curve(calculation_date)
    dividends = []

    s, k, t, v_is_call, iv = build_test_data(n, calculation_date)

    # -----------------------------
    # Warm-up
    # -----------------------------
    print("Warming up Julia...")
    _ = get_price_jl(
        s=s[:10], k=k[:10], t=t[:10], v_is_call=v_is_call[:10], iv=iv[:10],
        dividends=dividends, calculation_date=calculation_date,
        yield_curve=yield_curve, cpu=False
    )

    print("Warming up CUDA...")
    _ = get_price_cuda(
        s=s[:10], k=k[:10], t=t[:10], v_is_call=v_is_call[:10], iv=iv[:10],
        dividends=dividends, calculation_date=calculation_date,
        yield_curve=yield_curve, cpu=False
    )

    # -----------------------------
    # Benchmark Julia
    # -----------------------------
    jl_prices, jl_seconds = time_call(
        get_price_jl,
        s=s, k=k, t=t, v_is_call=v_is_call, iv=iv,
        dividends=dividends, calculation_date=calculation_date,
        yield_curve=yield_curve, cpu=False,
    )

    # -----------------------------
    # Benchmark CUDA
    # -----------------------------
    cuda_prices, cuda_seconds = time_call(
        get_price_cuda,
        s=s, k=k, t=t, v_is_call=v_is_call, iv=iv,
        dividends=dividends, calculation_date=calculation_date,
        yield_curve=yield_curve, cpu=False,
    )

    jl_prices = np.asarray(jl_prices, dtype=np.float64)
    cuda_prices = np.asarray(cuda_prices, dtype=np.float64)

    # -----------------------------
    # Compare outputs
    # -----------------------------
    abs_diff = np.abs(jl_prices - cuda_prices)

    print("\n=== Timing ===")
    print(f"Julia: {jl_seconds:.4f} s  ({jl_seconds / n * 1e6:.2f} us/contract)")
    print(f"CUDA : {cuda_seconds:.4f} s  ({cuda_seconds / n * 1e6:.2f} us/contract)")

    print("\n=== Accuracy ===")
    print(f"Mean abs diff: {abs_diff.mean():.6g}")
    print(f"Max abs diff : {abs_diff.max():.6g}")
    print(f"Median abs diff: {np.median(abs_diff):.6g}")

    return {
        "jl_seconds": jl_seconds,
        "cuda_seconds": cuda_seconds,
        "jl_us_per_contract": jl_seconds / n * 1e6,
        "cuda_us_per_contract": cuda_seconds / n * 1e6,
        "mean_abs_diff": float(abs_diff.mean()),
        "max_abs_diff": float(abs_diff.max()),
        "median_abs_diff": float(np.median(abs_diff)),
    }

def benchmark_many(n=10_000, repeats=5):
    results = []

    for i in range(repeats):
        print(f"\nRun {i+1}/{repeats}")
        result = benchmark_pricers(n)
        results.append(result)

    jl_times = np.array([r["jl_seconds"] for r in results])
    cuda_times = np.array([r["cuda_seconds"] for r in results])

    print("\n=== Summary ===")
    print(f"Julia median: {np.median(jl_times):.4f} s")
    print(f"CUDA  median: {np.median(cuda_times):.4f} s")

if __name__ == "__main__":
    benchmark_pricers(10_000)