using Dates

"""
    SSVISurfParams

A typed dictionary mapping Date keys to SSVITenorParams values.
Validates both keys and values on assignment.
"""
struct SSVISurfParams
    data::Dict{Date, Any}  # Holds SSVITenorParams (using Any to avoid circular deps)

    function SSVISurfParams(data::Dict{Date, Any} = Dict{Date, Any}())
        new(data)
    end
end

"""
    Base.setindex!(params::SSVISurfParams, value, key::Date)

Set a value with type validation.
"""
function Base.setindex!(params::SSVISurfParams, value, key::Date)
    if !isa(key, Date)
        throw(TypeError(:SSVISurfParams, "Key must be of type Date, got $(typeof(key))"))
    end
    # Validate value is SSVITenorParams (check if it has required fields)
    if !hasfield(typeof(value), :theta) || !hasfield(typeof(value), :rho) || !hasfield(typeof(value), :psi)
        throw(TypeError(:SSVISurfParams, "Value must be of type SSVITenorParams"))
    end
    params.data[key] = value
end

"""
    Base.getindex(params::SSVISurfParams, key::Date)

Get a value with type validation.
"""
function Base.getindex(params::SSVISurfParams, key::Date)
    if !isa(key, Date)
        throw(TypeError(:SSVISurfParams, "Key must be of type Date, got $(typeof(key))"))
    end
    return params.data[key]
end

"""
    Base.keys(params::SSVISurfParams)

Return keys (dates) in the params.
"""
Base.keys(params::SSVISurfParams) = keys(params.data)

"""
    Base.values(params::SSVISurfParams)

Return values (SSVITenorParams) in the params.
"""
Base.values(params::SSVISurfParams) = values(params.data)

"""
    Base.iterate(params::SSVISurfParams)

Iterate over (date, tenor_params) pairs.
"""
Base.iterate(params::SSVISurfParams) = iterate(params.data)
Base.iterate(params::SSVISurfParams, state) = iterate(params.data, state)