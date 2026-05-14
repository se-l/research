Base.@kwdef struct SSVITenorParams
    theta::Float64
    rho::Float64
    psi::Float64
end

# 1. Start iteration (returns first value and the next "state" index)
Base.iterate(p::SSVITenorParams) = (p.theta, 2)

# 2. Continue iteration based on the state index
Base.iterate(p::SSVITenorParams, state) = 
    state == 2 ? (p.rho, 3) :
    state == 3 ? (p.psi, 4) : nothing

# 3. Define the length (optional, but good practice)
Base.length(p::SSVITenorParams) = 3