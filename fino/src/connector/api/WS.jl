module WS

using PythonCall
using HTTP
using HTTP.WebSockets

const EARNINGS_IV_DROP_REGRESSOR_MODEL_NAME_VERSION="f_20260407-205754"

# ── Python imports (module-level, initialised once) ──────────────────────────
const py_surfaces            = PythonCall.pynew()
const py_kalman              = PythonCall.pynew()
const py_equity_mod          = PythonCall.pynew()
const py_enums               = PythonCall.pynew()
const py_pf_opt_minlp_pyomo  = PythonCall.pynew()
const py_pf_impact_by_option = PythonCall.pynew()
const py_pyomo               = PythonCall.pynew()

const _py_initialized = Ref{Bool}(false)

function ensure_py_initialized()
    _py_initialized[] && return

    sys = pyimport("sys")
    for repo_root in [dirname(dirname(dirname(dirname(dirname(@__DIR__))))), "C:\\repos\\trade\\src"]
        if !pyconvert(Bool, sys.path.__contains__(repo_root))
            sys.path.insert(0, repo_root)
        end
    end

    PythonCall.pycopy!(py_equity_mod, pyimport("options.types.equity"))
    PythonCall.pycopy!(py_enums,      pyimport("options.types.enums"))
    PythonCall.pycopy!(py_surfaces,   pyimport("options.surfaces.processors"))
    PythonCall.pycopy!(py_kalman,     pyimport("estimators.earnings_iv_drop_ssvi_surface.kalman_filter"))
    PythonCall.pycopy!(py_pf_opt_minlp_pyomo,      pyimport("options.volatility.pf_opt_minlp_pyomo"))
    PythonCall.pycopy!(py_pf_impact_by_option,     pyimport("derivatives.pf_impact_by_option"))
    PythonCall.pycopy!(py_pyomo,     pyimport("pyomo.environ"))

    _py_initialized[] = true
end

export EARNINGS_IV_DROP_REGRESSOR_MODEL_NAME_VERSION
export ensure_py_initialized, py_surfaces, py_kalman, py_equity_mod, py_enums, py_pf_opt_minlp_pyomo, py_pf_impact_by_option, py_pyomo

# protos begin
include(joinpath(@__DIR__, "protos", "Common_pb.jl"))
include(joinpath(@__DIR__, "protos", "RequestSSVICalibration_pb.jl")) # Needed by many
include(joinpath(@__DIR__, "protos", "Command_pb.jl"))
include(joinpath(@__DIR__, "protos", "PfRiskScenarios_pb.jl"))
include(joinpath(@__DIR__, "protos", "RequestKalmanInit_pb.jl"))
include(joinpath(@__DIR__, "protos", "StressTestDs_pb.jl"))
include(joinpath(@__DIR__, "protos", "RequestTargetPortfolios_pb.jl"))
include(joinpath(@__DIR__, "protos", "Websocket_pb.jl"))

using .Websocket_pb
using .Common_pb
using .Command_pb
using .PfRiskScenarios_pb
using .RequestKalmanInit_pb
using .RequestSSVICalibration_pb
using .RequestTargetPortfolios_pb
using .StressTestDs_pb

export SecurityTypePb, VectorDoublePb, OptionQuotePb, OptionRightPb, HoldingPb, TradePb
export QuotePb, MarketDataSnapByUnderlyingPb, TradesPb, QuotesPb, MarketDataHistoryPb
export CmdCancelOID, CmdCfgOverride, CmdFetchTargetPortfolio
export RequestPfRiskScenariosPb, PfRiskScenarioPb, ResponsePfRiskScenariosPb
export RequestKalmanInitPb, ResponseKalmanInitPb
export SSVIModelParamsPb, RequestSSVICalibrationPb, SSVIParamsPb, SSVIParamsByRightPb
export ResponseSSVICalibrationPb
export RequestTargetPortfoliosPb, TargetPortfolioPb, ResponseTargetPortfoliosPb
export ResultStressTestDsPb, RequestStressTestDsPb
export ActionPb, ChannelPb, MessagePb

# protos end

include(joinpath(@__DIR__, "common.jl"))
export parse_pb, holding2holding_pb, is_response_cached, cache_request, cache_result, send_response, load_response_from_cache,
    send_empty_response, get_density_for_bimodal_t_dist, holdings_pb2portfolio, get_mid_iv_from_cache, get_ordered_holdings, pb2bytes,
    get_cache_key_if_not_present, convert_py_sec2jl_sec2, StressTestDsResult2StressTestDsResult_pb, py2stress_inputs, py2jl_holdings,
    HoldingPb
include(joinpath(@__DIR__, "handlers", "AbstractHandler.jl"))
    export spawn_task, DT_FMT_PB

#include(joinpath(@__DIR__, "handlers", "echo.jl"))
#include(joinpath(@__DIR__, "handlers", "terminate.jl"))
include(joinpath(@__DIR__, "handlers", "greeting.jl"))
include(joinpath(@__DIR__, "handlers", "heartbeat.jl"))
include(joinpath(@__DIR__, "handlers", "KalmanHandler.jl"))
include(joinpath(@__DIR__, "handlers", "PfRiskScenarioHandler.jl"))
include(joinpath(@__DIR__, "handlers", "SSVICalibrationHandler.jl"))
include(joinpath(@__DIR__, "handlers", "StressTestDsHandler.jl"))
include(joinpath(@__DIR__, "handlers", "TargetPortfoliosHandler.jl"))
#include(joinpath(@__DIR__, "handlers", "steer_algo.jl"))

export handle_on_heartbeat, StressTestDsHandler, KalmanHandler, SSVICalibrationHandler, PfRiskScenarioHandler, TargetPortfoliosHandler

include(joinpath(@__DIR__, "handlers", "WsMsgBroker.jl"))

using .WsMsgBroker

# ============================================
# Start Function
# ============================================

"""
    start_ws(; host="127.0.0.1", port=8002)
Start the WebSocket server.
# Arguments
- `host::String`: Server host address (default: "127.0.0.1")
- `port::Int`: Server port (default: 8002)
"""
function start_ws(; host::String="127.0.0.1", port::Int=8002)
    @info "[WS] Starting server on http://$host:$port"
    HTTP.listen(host, port) do stream
        req = stream.message
        if req.target == "/ws" && HTTP.hasheader(req, "Upgrade", "websocket")
            HTTP.WebSockets.upgrade(stream) do ws
                @info "[WS] Client connected"
                try
                    WsMsgBroker.handler(ws)
                catch e
                    @warn "[WS] Error" exception = e
                end
                @info "[WS] Client disconnected"
            end
        elseif req.target == "/"
            resp = greeting_handler(req)
            HTTP.startwrite(stream)
            write(stream, resp.body)
        else
            HTTP.setstatus(stream, 404)
            HTTP.startwrite(stream)
            write(stream, "Not Found")
        end
    end
end

end # module
