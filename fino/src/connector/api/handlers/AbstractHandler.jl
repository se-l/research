# handlers/AbstractHandler.jl

"""
Shared infrastructure for all WS message handlers.
Provides: background task management, logging, spawn+cleanup pattern.
"""

using Base.Threads
using Dates

const DT_FMT_PB = "yyyy-mm-ddTHH:MM:SS"

# ─── Background task management ───────────────────────────────────────────────

"""
    spawn_task(background_tasks::Set{Task}, label::String, f::Function)

Spawn `f` on a worker thread, register it in `background_tasks`,
and set up async cleanup + error logging on completion.
"""
function spawn_task(f::Function, background_tasks::Set{Task}, label::String)
    task = Threads.@spawn begin
        try
            f()
        catch e
            @error "[$label] Unhandled error" exception=(e, catch_backtrace())
        end
    end

    push!(background_tasks, task)
    @async begin
        try
            wait(task)
        catch e
            @error "[$label] Task wait error" exception=(e, catch_backtrace())
        finally
            delete!(background_tasks, task)
            @info "[$label] task completed. Active: $(length(background_tasks))"
        end
    end

    return task
end

"""
    @with_timeout seconds expr

Runs `expr` on a spawned task. If it doesn't complete within `seconds`,
"""
macro with_timeout(seconds, expr)
    quote
        result_ch = Channel{Any}(1)
        task = Threads.@spawn begin
            try
                put!(result_ch, $(esc(expr)))
            catch e
                put!(result_ch, e)
            end
        end

        if timedwait(() -> isready(result_ch), $(esc(seconds))) === :timed_out
            Base.throwto(task, InterruptException())
            error("Timed out after $($(esc(seconds)))s")
        end

        val = take!(result_ch)
        val isa Exception ? throw(val) : val
    end
end