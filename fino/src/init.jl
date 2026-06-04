# Init.jl
using Revise
import Pkg

ENV["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

let
    current = abspath(@__DIR__)
    while current != dirname(current)
        project_file = joinpath(current, "Project.toml")
        if isfile(project_file) && occursin("name = \"Fino\"", read(project_file, String))
            @info "Activating Fino project at: $current"
            Pkg.activate(current)
            break
        end
        current = dirname(current)
    end
end

using Fino
Base.exit_on_sigint(false)
@info "Fino loaded successfully"