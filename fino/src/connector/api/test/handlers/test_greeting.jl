
using Pkg
Pkg.activate("C:\\repos\\research\\fino")

# Mocking some dependencies if they are not in the environment
# But first let's try to just load the file and see if it compiles

try
    include("C:\\repos\\research\\fino\\src\\connector\\api\\WS.jl")
    println("WS.jl included successfully")
    
    # Try calling the handler
    # Note: greeting_handler is in WS module
    response = WS.greeting_handler()
    println("greeting_handler called successfully")
    println("Response type: ", typeof(response))
    # println("Response content: ", response.body)
catch e
    rethrow(e)
end
