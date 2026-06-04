include(joinpath(@__DIR__, "../../init.jl"))

using .Fino

function clear_cache(symbol::String)
    cache_dir = Paths.PATH_API_CACHE
    for entry in readdir(cache_dir)
        if occursin(symbol, entry)
            fp = joinpath(cache_dir, entry)
            rm(fp)
            @info "Deleted cache file: $fp"
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    clear_cache("HPE")
end
