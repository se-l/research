using .Fino: Security

"""
    Holding{T <: Security}

Represents a position in a security with a given quantity.

Type Parameters:
- `T`: The security type (Equity, Option, etc.)

Fields:
- `symbol::T`: The security
- `quantity::Float64`: Number of shares/contracts
"""
struct Holding{T <: Security}
    symbol::T
    quantity::Float64
end

# Constructor with default Float64
Holding(symbol::T, quantity::Real) where {T <: Security} = Holding(symbol, Float64(quantity))

"""
    from_holding_pb(holding_pb, calculation_date=nothing, security_type=nothing)

Convert from protobuf holding to Holding type.

Arguments:
- `holding_pb`: The protobuf holding object
- `calculation_date::Union{Date, Nothing}`: Optional calculation date
- `security_type::Union{SecurityType, Nothing}`: Optional security type (inferred if not provided)

Returns:
- `Holding{E}` or `Holding{Option}` depending on the security type
"""
function from_holding_pb(holding_pb, calculation_date::Union{Date, Nothing}=nothing, security_type::Union{SecurityType, Nothing}=nothing)
    # Infer security type from symbol if not provided
    st = something(security_type, security_type_from_ib_symbol(holding_pb.symbol))
    
    if st === security_type_equity
        return Holding{Equity}(Equity(holding_pb.symbol), holding_pb.quantity)
    elseif st === security_type_option
        option = option_from_ib_symbol(holding_pb.symbol, calculation_date)
        return Holding{Option}(option, holding_pb.quantity)
    else
        throw(ArgumentError("Unknown security type: $st"))
    end
end

# Hash function - matches Python's __hash__
function Base.hash(h::Holding, hdf::UInt=zero(UInt))
    return hash((h.symbol, h.quantity), hdf)
end

# Equality
Base.:(==)(h1::Holding, h2::Holding) = h1.symbol == h2.symbol && h1.quantity == h2.quantity

# Show method for debugging
function Base.show(io::IO, h::Holding)
    print(io, "Holding($(h.symbol), $(h.quantity))")
end

## Create a holding
#eq = Equity("AAPL")
#h = Holding(eq, 100.0)
#
## Convert from protobuf (assuming you have a HoldingPb type)
#holding_pb = HoldingPb(symbol="AAPL", quantity=100.0)
#h = from_holding_pb(holding_pb)
#
## Hash
#hash(h)  # works with Dict/Set as key