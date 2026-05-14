module Fino
# Anchor all internal includes to this file's location
const ROOT = dirname(@__DIR__)

# ── Python imports (module-level, initialised once) ──────────────────────────
using PythonCall

const py_ers_mod = PythonCall.pynew()
const py_earnings_config_mod = PythonCall.pynew()
const pydate = PythonCall.pynew()

const _py_initialized = Ref{Bool}(false)

const PY_WORK_CHANNEL = Channel{Tuple{Function, Channel{Any}}}(32)

"""
    start_py_worker()

Must be called once on thread 1 before starting the WS server.
Drains PY_WORK_CHANNEL, executing each Python job on thread 1.
"""
function start_py_worker()
    # This @async runs on thread 1 (sticky to caller)
    t = Task() do
        @info "py_worker pinned to thread $(Threads.threadid())"
        for (f, result_ch) in PY_WORK_CHANNEL
            try
                put!(result_ch, f())
            catch e
                put!(result_ch, e)
            end
        end
    end
    t.sticky = true          # prevents migration to other threads
    schedule(t)
    return t
end

"""
    run_on_py_thread(f) -> result

Submit f() to be executed on thread 1 (where the GIL lives) and block until done.
"""
function run_on_py_thread(f::Function)
    result_ch = Channel{Any}(1)
    put!(PY_WORK_CHANNEL, (f, result_ch))
    result = take!(result_ch)
    if result isa Exception
        if result isa PythonCall.PyException
            py_tb_str = try
                sprint(showerror, result)
            catch
                "<unable to format Python traceback>"
            end
            @error "run_on_py_thread: PythonException" py_tb=py_tb_str
        end
#        println(stderr, "=== PYTHON ERROR ===\n$py_tb_str\n=====================")
        throw(result)
    end
    return result
end

function ensure_py_initialized_fino()
    _py_initialized[] && return

    # Must patch PATH before any pyomo imports
    os = pyimport("os")
    ipopt_dir = "C:\\Users\\seb\\miniconda3\\envs\\py314\\Library\\bin"
    current_path = pyconvert(String, os.environ["PATH"])
    if !contains(current_path, ipopt_dir)
        os.environ["PATH"] = ipopt_dir * ";" * current_path
    end

    sys = pyimport("sys")
    for repo_root in ["C:\\repos\\research", "C:\\repos\\trade\\src"]
        if !pyconvert(Bool, sys.path.__contains__(repo_root))
            sys.path.insert(0, repo_root)
        end
    end

    PythonCall.pycopy!(py_ers_mod, pyimport("derivatives.earnings_release_ssvi"))
    PythonCall.pycopy!(py_earnings_config_mod, pyimport("options.types.earnings_config"))
    PythonCall.pycopy!(pydate, pyimport("datetime").date)

    _py_initialized[] = true
end
export py_ers_mod, pydate

include(joinpath(@__DIR__, "shared", "Paths.jl"))
include(joinpath(@__DIR__, "shared", "constants.jl"))
include(joinpath(@__DIR__, "shared", "enums.jl"))
include(joinpath(@__DIR__, "shared", "helper.jl"))
export get_tenor, trade_days_between_dates

include(joinpath(@__DIR__, "PricingEngine.jl"))
include(joinpath(@__DIR__, "CalibrateIVS.jl"))
include(joinpath(@__DIR__, "DividendManager.jl"))
include(joinpath(@__DIR__, "YieldCurve.jl"))
include(joinpath(@__DIR__, "PricerZMQ.jl"))

include(joinpath(@__DIR__, "derivatives", "EarningsReleaseSSVI.jl"))

include(joinpath(@__DIR__, "types", "SymDate.jl"))
include(joinpath(@__DIR__, "types", "security.jl"))
include(joinpath(@__DIR__, "types", "Cash.jl"))
include(joinpath(@__DIR__, "types", "earnings_iv_drop_poly_regressor_v3.jl"))
include(joinpath(@__DIR__, "types", "equity.jl"))
include(joinpath(@__DIR__, "types", "holding.jl"))
export from_holding_pb, Holding
include(joinpath(@__DIR__, "types", "portfolio.jl"))
export Portfolio
include(joinpath(@__DIR__, "types", "earnings_config.jl"))
include(joinpath(@__DIR__, "types", "RawDataConfig.jl"))
include(joinpath(@__DIR__, "types", "SSVISurfParams.jl"))
include(joinpath(@__DIR__, "types", "SSVITenorParams.jl"))
include(joinpath(@__DIR__, "types", "StressTestDsResult.jl"))
include(joinpath(@__DIR__, "types", "options.jl"))
export Option, option_from_ib_symbol
include(joinpath(@__DIR__, "shared", "stress_test.jl"))
export StressTestDsResult, get_stress_test_ds,
           holdings2v_q, get_nlv_by_ds, get_weighted_dlnv,
           get_delta_total_across_ds, get_total_objective,
           get_density_for_bimodal_t_dist, get_assumed_fill_iv
include(joinpath(@__DIR__, "types", "earnings_release_simulation.jl"))
export pf_to_py_pf, PyEarningsReleaseSimulation, AbstractEarningsReleaseSimulation

using Logging
global_logger(ConsoleLogger(stderr, Logging.Info,
    meta_formatter = (level, _module, group, id, file, line) -> begin
        color = level == Logging.Error ? :red :
                level == Logging.Warn  ? :yellow : :cyan
        (color, "$(now()) [$(basename(string(file))):$line]", "")
    end
))

# Bring submodules into scope (relative imports with dot)
using .Paths

using .PricingEngine
using .CalibrateIVS
using .EarningsReleaseSSVI
using .YieldCurve
export YieldCurve, ZeroCurveData, get_zero_curve, get_last_zero_curve, store_calibrated_rates, load_calibrated_rates, has_calibrated_curve
using .DividendManager
export Dividend, get_dividends, fetch_dividends, load_from_disk, get_dividend_amount_times
include(joinpath(@__DIR__, "shared", "moneyness.jl"))
using .PricerZMQ

include(joinpath(@__DIR__, "types", "calibration_item.jl"))
include(joinpath(@__DIR__, "connector", "api", "WS.jl"))

# Optionally export key public APIs from submodules
export Paths, PricingEngine, CalibrateIVS, EarningsReleaseSSVI, EarningsPreSessionDates,
    start_pricer, get_dividend_amount_times, get_dividends, CalibrationItem, downsample, df2calibration_items, WS
export Security, Equity, security_type_equity, security_type_option, option_right_call, option_right_put, OptionRight,
    EarningsConfig

end # module Fino

# using ProtoBuf

## Deserialize from bytes
#message = ProtoBuf.read(io, YourMessageType)
# This approach skips the code generation step entirely and works directly with proto definitions.