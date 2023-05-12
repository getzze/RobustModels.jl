

soft_threshold(x::Real, λ::Real) = (abs(x) <= λ) ? zero(x) : x - sign(x) * λ


"""
    concrete!(p::PenaltyFunction, m::Integer, intercept_index::Union{Nothing,Integer}=nothing)

Modifies the penalty to make it compatible with the coefficients of size `m`.
For penalties that do not handle indices range, return the unmodified penalty,
otherwise (like for RangedPenalties), modify the ranges so they cover all
indices in the `1:m` range.
Throws an error if the penalty ranges are not compatible with `m` or if `intercept_index`
is defined and `p` is not a RangedPenalties.
"""
function concrete!(p::PenaltyFunction, n::Integer, intercept_index::Union{Nothing,Integer}=nothing)
    if !isnothing(intercept_index) && (1 <= intercept_index <= n)
        error(
            "intercept_index $(intercept_index) in 1:$n, cannot change the type " *
            "from $(typeof(p)) to RangedPenalties."
        )
    end
    p
end

"""
    concrete(p::PenaltyFunction, m::Integer)

Returns a penalty that is compatible with the coefficients of size `m`.
For penalties that do not handle indices range, if `intercept_index` is not defined
return a deep copy of the penalty, otherwise return a RangedPenalties excluding this index.
For RangedPenalties, returns a penalty with ranges that cover all the indices in the `1:m` range.
Throws an error if the penalty ranges are not compatible with `m`.
"""
function concrete(p::PenaltyFunction, n::Integer, intercept_index::Union{Nothing,Integer}=nothing)
    if !isnothing(intercept_index) && (1 <= intercept_index <= n)
        if n == 1
            ranges = []
            penalties = []
        else
            ranges = [excludeindex(1:n, intercept_index)]
            penalties = [p]
        end
        return RangedPenalties(ranges, penalties, n)
    else
        return deepcopy(p)
    end
end



function proximal!(p::PenaltyFunction, out::AbstractVector, index::Integer, x::AbstractVector, step::AbstractFloat)
    proximal!(p, view(out, index), x[[index]], step)
    return out
end

proximal(p::PenaltyFunction, x::AbstractVector, step::AbstractFloat) = proximal!(p, similar(x), x, step)
proximal(p::PenaltyFunction, x::T, step::AbstractFloat) where {T<:AbstractFloat} = proximal!(p, ones(T, 1), [x], step)[1]



########################################################################
##### Penalty functions
########################################################################

"""
    NoPenalty{T<:AbstractFloat}

No penalty (constant penalty), the proximal operator returns the same vector as input (identity map).
P(x) = 0
"""
struct NoPenalty{T<:AbstractFloat} <: PenaltyFunction{T} end
NoPenalty(args...) = NoPenalty{Float64}()
cost(::NoPenalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat} = zero(T)
proximal!(p::NoPenalty{T}, out, x::AbstractVector{T}, step::T=one(T)) where {T<:AbstractFloat} = copyto!(out, x)

"""
    SquaredL2Penalty{T<:AbstractFloat}

Squared L2 penalty, the proximal operator returns a scaled version.
P(x) = λ/2 Σi |xi|²
"""
struct SquaredL2Penalty{T<:AbstractFloat} <: PenaltyFunction{T}
    λ::T
    nonnegative::Bool

    function SquaredL2Penalty(
        λ::T,
        nonnegative::Bool=false,
    ) where {T<:AbstractFloat}
        λ >= 0 || throw(ArgumentError("penalty constant λ should be non-negative: $λ"))
        new{T}(λ, nonnegative)
    end
end
cost(p::SquaredL2Penalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat} = p.λ / 2 * sum(abs2, x)
function proximal!(p::SquaredL2Penalty{T}, out, x::AbstractVector{T}, step::T=one(T)) where {T<:AbstractFloat}
#    return broadcast!(/, out, x, 1 + p.λ * step)
    a = 1 / (1 + p.λ * step)
    @inbounds @simd for i in eachindex(out, x)
        out[i] = x[i] > 0 ? x[i] * a : zero(T)
    end
    return out
end

"""
    EuclideanPenalty{T<:AbstractFloat}

Euclidean norm penalty, the proximal operator returns a scaled version.
P(x) = λ √(Σi |xi|²)
"""
struct EuclideanPenalty{T<:AbstractFloat} <: PenaltyFunction{T}
    λ::T
    nonnegative::Bool

    function EuclideanPenalty(
        λ::T,
        nonnegative::Bool=false,
    ) where {T<:AbstractFloat}
        λ >= 0 || throw(ArgumentError("penalty constant λ should be non-negative: $λ"))
        new{T}(λ, nonnegative)
    end
end
cost(p::EuclideanPenalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat} = p.λ * norm(x, 2)
function proximal!(p::EuclideanPenalty{T}, out, x::AbstractVector{T}, step::T=one(T)) where {T<:AbstractFloat}
    nn = p.nonnegative ? norm(broadcast!(max, out, x, 0), 2) : norm(x, 2)
    return rmul!(copyto!(out, x), (1 - p.λ * step / max(p.λ * step, nn)))
end

"""
    L1Penalty{T<:AbstractFloat}

L1 penalty, the proximal operator returns a soft-thresholded value.
P(x) = λ Σi |xi|
"""
struct L1Penalty{T<:AbstractFloat} <: PenaltyFunction{T}
    λ::T
    nonnegative::Bool

    function L1Penalty(
        λ::T,
        nonnegative::Bool=false,
    ) where {T<:AbstractFloat}
        λ >= 0 || throw(ArgumentError("penalty constant λ should be non-negative: $λ"))
        new{T}(λ, nonnegative)
    end
end
cost(p::L1Penalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat} = p.λ * sum(abs, x)
function proximal!(p::L1Penalty{T}, out, x::AbstractVector{T}, step::T=one(T)) where {T<:AbstractFloat}
    @inbounds @simd for i in eachindex(out, x)
        out[i] = (p.nonnegative && x[i] <= 0) ? zero(T) : soft_threshold(x[i], p.λ * step)
    end
    return out
end

"""
    ElasticNetPenalty{T<:AbstractFloat}

ElasticNet penalty, a sum of SquaredL2Penalty and sparse L1Penalty.
P(x) = l1_ratio . λ Σi |xi| + (1 - l1_ratio) . λ/2 Σi |xi|²
"""
struct ElasticNetPenalty{T<:AbstractFloat} <: PenaltyFunction{T}
    λ::T
    nonnegative::Bool
    l1_ratio::T

    function ElasticNetPenalty(
        λ::T,
        nonnegative::Bool=false,
        l1_ratio::AbstractFloat=T(0.5),
    ) where {T<:AbstractFloat}
        λ >= 0 || throw(ArgumentError("penalty constant λ should be non-negative: $λ"))
        0 <= l1_ratio <= 1 || throw(ArgumentError("l1_ratio must be between 0 and 1: $(l1_ratio)"))
        new{T}(λ, nonnegative, l1_ratio)
    end
end
cost(p::ElasticNetPenalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat} =
    p.λ * (p.l1_ratio * sum(abs, x) + (1 - p.l1_ratio) * sum(abs2, x) / 2)
function proximal!(p::ElasticNetPenalty{T}, out, x::AbstractVector{T}, step::T=one(T)) where {T<:AbstractFloat}
    a = p.λ * step
    @inbounds @simd for i in eachindex(out, x)
        out[i] = (p.nonnegative && x[i] <= 0) ? zero(T) :
            (soft_threshold(x[i], p.l1_ratio * a) / (1 + (1 - p.l1_ratio) * a))
    end
    return out
end



#################################
### RangedPenalties
#################################

abstract type AbstractRangedPenalties{T} <: PenaltyFunction{T} end

struct End{T<:Integer}
    offset::T

    End{T}(offset::T) where {T<:Integer} = new{T}(offset)
    End{T}() where {T<:Integer} = new{T}(zero(T))
end
End(offset::T) where {T<:Integer} = End{T}(offset)
End() = End{Int}()

struct EndStepRange{T<:Integer}
    start::T
    step::T
    stop::T

    EndStepRange{T}(start::Integer, step::Integer, stop::Integer) where {T<:Integer} =
        new{T}(Base.convert.(T, (start, step, stop))...)
end
EndStepRange(start::I1, step::I2, stop::I3) where {I1<:Integer,I2<:Integer,I3<:Integer} =
    EndStepRange{promote_type(I1, I2, I3)}(start, step, stop)

(::Colon)(start::Integer, stop::End) = EndStepRange(start, one(start), stop.offset)
(::Colon)(start::Integer, step::Integer, stop::End) = EndStepRange(start, step, stop.offset)

torange(p::EndStepRange, n::Integer) = (p.start):(p.step):(n - p.stop)
torange(p::Vector, n::Integer) = all(1 .<= p .<= n) ? sort(p) : throw(BoundsError("attempt to access $n-element Vector at index [$(p)]"))
torange(p::AbstractRange, n::Integer) = (last(p) <= n) ? p : throw(BoundsError("attempt to access $n-element Vector at index [$(p)]"))

function excludeindex(p::Vector, i::Integer)
    i in p || return p

    filter(!=(i), p)
end
function excludeindex(p::AbstractRange, i::Integer)
    i in p || return p

    if first(p) == i
        ran = (first(p) + step(p)):step(p):last(p)
    elseif last(p) == i
        ran = first(p):step(p):(last(p) - step(p))
    else
        ran = filter(!=(i), p)
    end
    ran
end

RangeLikeType{T} = Union{UnitRange{<:T}, StepRange{<:T,<:T}, EndStepRange{<:T}, Vector{<:T}}


mutable struct RangedPenalties{T<:AbstractFloat, N<:Integer} <: AbstractRangedPenalties{T}
    ranges::Vector{RangeLikeType{N}}
    penalties::Vector{PenaltyFunction}
    notinrange::Vector{N}
    isconcrete::Bool

    function RangedPenalties{T, N}(
        ranges::AbstractVector,
        penalties::AbstractVector{<:PenaltyFunction{T}},
    ) where {T<:AbstractFloat, N<:Integer}
        any(p isa RangedPenalties for p in penalties) && throw(ArgumentError(
            "RangedPenalties penalties should not be of type RangedPenalties: $(typeof.(penalties))"))
        length(ranges) == length(penalties) || throw(ArgumentError(
            "ranges and penalties should have the same number of elements: $(length(ranges)) != $(length(penalties))"))

        return new{T, N}(ranges, penalties, zeros(N, 0), false)
    end
end

function RangedPenalties(
    ranges::AbstractVector,
    penalties::AbstractVector{<:PenaltyFunction{T}},
    n::Union{Nothing,Integer}=nothing,
) where {T<:AbstractFloat}
    length(penalties) > 0 || throw(ArgumentError("RangedPenalties should contain at least one PenaltyFunction"))
    p = RangedPenalties{T,typeof(length(ranges))}(ranges, penalties)
    if !isnothing(n)
        concrete!(p, n)
    end
    return p
end

function Base.:(==)(x::RangedPenalties, y::RangedPenalties)
    if !isconcrete(x) || !isconcrete(y)
        return x.ranges == y.ranges && x.penalties == y.penalties
    end

    # Concrete penalties
    x.penalties == y.penalties || return false
    for (xr, yr) in zip(x.ranges, y.ranges)
        Set(xr) == Set(yr) || return false
    end
    Set(x.notinrange) == Set(y.notinrange) || return false
    return true
end

isconcrete(p::RangedPenalties) = p.isconcrete

function concrete!(
    p::RangedPenalties,
    n::Integer,
    intercept_index::Union{Nothing,Integer}=nothing,
)
    # Check that ranges are non-overlapping
    notinrange = Set(1:n)
    @inbounds for j in eachindex(p.ranges, p.penalties)
        ran = torange(p.ranges[j], n)
        sran = Set(ran)
        if intersect(notinrange, sran) != sran
            error("Overlapping ranges are not allowed: $(p.ranges)")
        end
        if !isnothing(intercept_index)
            # Remove intercept_index from range
            ran = excludeindex(ran, intercept_index)
            sran = Set(ran)
        end
        ## Transform End-ranges to Base ranges
        p.ranges[j] = ran
        ## Remoev used ranges
        setdiff!(notinrange, sran)
    end
    ## Store missing indices in notinrange
    p.notinrange = sort!(collect(notinrange))
    p.isconcrete = true
    return p
end

function concrete(
    p::RangedPenalties,
    n::Integer,
    intercept_index::Union{Nothing,Integer}=nothing,
)
    new_p = deepcopy(p)
    return concrete!(new_p, n, intercept_index)
end

function cost(p::RangedPenalties{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    isconcrete(p) || error("RangedPenalties not concrete, call `concrete!` beforehand to make sure the ranges are well-defined.")

    n = length(x)
    r = zero(T)
    for (ran, pen) in zip(p.ranges, p.penalties)
        r += cost(pen, view(x, ran))
    end
    return r
end
function proximal!(p::RangedPenalties{T}, out, x::AbstractVector{T}, step::T=one(T)) where {T<:AbstractFloat}
    isconcrete(p) || error("RangedPenalties not concrete, call `concrete!` beforehand to make sure the ranges are well-defined.")

    n = length(x)
    for (ran, pen) in zip(p.ranges, p.penalties)
        proximal!(pen, view(out, ran), view(x, ran), step)
    end
    # Apply NoPenalty to indices not defined by ranges
    if !isempty(p.notinrange)
        ran = p.notinrange
        pen = NoPenalty{T}()
        proximal!(pen, view(out, ran), view(x, ran), step)
    end
    return out
end
function proximal!(p::RangedPenalties{T}, out::AbstractVector, index::Integer, x::AbstractVector, step::AbstractFloat)  where {T<:AbstractFloat}
    isconcrete(p) || error("RangedPenalties not concrete, call `concrete!` beforehand to make sure the ranges are well-defined.")

    done = false
    for (ran, pen) in zip(p.ranges, p.penalties)
        if in(index, ran)
            proximal!(pen, view(out, index), x[[index]], step)
            done = true
            break
        end
    end
    if !done && !isempty(p.notinrange)
        if !in(index, p.notinrange)
            n = length(x)
            error("index not found in range $(1:n): $index")
        end
        proximal!(NoPenalty{T}(), view(out, index), x[[index]], step)
    end
    return out
end
