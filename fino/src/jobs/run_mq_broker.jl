include(joinpath(@__DIR__, "../init.jl"))

try
    Fino.ensure_py_initialized_fino()
    Fino.WS.ensure_py_initialized()
    Fino.start_py_worker()
    Fino.MQMsgBroker.start_broker(
#        port=parse(Int, get(ENV, "WsPort", "8002")),
#        host=get(ENV, "WsHost", "0.0.0.0")
    )
catch e
    if e isa InterruptException
        @info "Shutting down gracefully..."
    else
        @error "Server error" exception=(e, catch_backtrace())
        rethrow(e)
    end
end
