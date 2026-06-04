include(joinpath(@__DIR__, "../init.jl"))

protocol = get(ENV, "PricerProtocol", "tcp")
host = get(ENV, "PricerHost", "0.0.0.0")
port = parse(Int, get(ENV, "PricerPort", "8102"))
Fino.PricerZMQ.start_pricer(protocol, host, port)