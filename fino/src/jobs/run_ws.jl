# run_ws.jl — Launcher for the Fino WS server
# Run from any directory: julia --project=C:/repos/research/fino run_ws.jl
import Pkg

# Ensure the Fino project is active (handles debugger CWD ≠ project dir)
if !isfile(joinpath(pwd(), "Project.toml")) ||
   !occursin("Fino", read(joinpath(pwd(), "Project.toml"), String))
    project_dir = "C:/repos/research/fino"
    @info "Activating project at $project_dir"
    Pkg.activate(project_dir)
end

try
    using Fino
    @info "Fino loaded successfully"
catch e
    @error "Failed to load Fino" exception=(e, catch_backtrace())
    rethrow(e)
end

using Base: exit

# Graceful shutdown on Ctrl+C / SIGINT
Base.exit_on_sigint(false)  # prevent immediate hard exit

@info "Starting Fino WS server. Press Ctrl+C to stop."

try
    Fino.ensure_py_initialized_fino()
    Fino.WS.ensure_py_initialized()
    Fino.start_py_worker()
    Fino.WS.start_ws()
catch e
    if e isa InterruptException
        @info "Shutting down gracefully..."
    else
        @error "Server error" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

# julia --project=C:/repos/research/fino C:/repos/research/fino/run_ws.jl
## Start Julia and run
#julia -e 'using Fino; Fino.WS.start_ws()'
#
## Or with custom port
#julia -e 'using Fino; Fino.WS.start_ws(port=9000)'