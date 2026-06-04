# run_ws.jl — Launcher for the Fino WS server
include(joinpath(@__DIR__, "../init.jl"))

try
    Fino.ensure_py_initialized_fino()
    Fino.WS.ensure_py_initialized()
    Fino.start_py_worker()
    Fino.WS.start_ws(
        port=parse(Int, get(ENV, "WsPort", "8002")),
        host=get(ENV, "WsHost", "0.0.0.0")
    )
catch e
    if e isa InterruptException
        @info "Shutting down gracefully..."
    else
        @error "Server error" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

## Start Julia and run
#julia -e 'using Fino; Fino.WS.start_ws()'
#
## Or with custom port
#julia -e 'using Fino; Fino.WS.start_ws(port=9000)'