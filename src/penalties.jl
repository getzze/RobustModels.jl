

soft_threshold(x::Real, λ::Real) = (abs(x) <= λ) ? zero(x) : x - sign(x) * λ



isconvex(::PenaltyFunction) = false
isseparable(::PenaltyFunction) = false

"""
    concrete!(p::PenaltyFunction, m::Integer, intercept_index::Union{Nothing,Integer}=nothing)

Modifies the penalty to make it compatible with the coefficients of size `m`.
For penalties that do not handle indices range, return the unmodified penalty,
otherwise (like for RangedPenalties), modify the ranges so they cover all
indices in the `1:m` range.
Throws an error if the penalty ranges are not compatible with `m` or if `intercept_index`
is defined and `p` is not a RangedPenalties.
"""
function concrete!(
    p::PenaltyFunction, n::Integer, intercept_index::Union{Nothing,Integer}=nothing
)
    if !isnothing(intercept_index) && (1 <= intercept_index <= n)
        error(
            "intercept_index $(intercept_index) in 1:$n, cannot change the type " *
            "from $(typeof(p)) to RangedPenalties.",
        )
    end
    return p
end

"""
    concrete(p::PenaltyFunction, m::Integer)

Returns a penalty that is compatible with the coefficients of size `m`.
For penalties that do not handle indices range, if `intercept_index` is not defined
return a deep copy of the penalty, otherwise return a RangedPenalties excluding this index.
For RangedPenalties, returns a penalty with ranges that cover all the indices in the `1:m` range.
Throws an error if the penalty ranges are not compatible with `m`.
"""
function concrete(
    p::PenaltyFunction, n::Integer, intercept_index::Union{Nothing,Integer}=nothing
)
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



function proximal!(
    p::PenaltyFunction,
    out::AbstractVector,
    index::Integer,
    x::AbstractVector,
    step::AbstractFloat,
)
    proximal!(p, view(out, index), view(x, index), step)
    return out
end

function proximal(p::PenaltyFunction, x::AbstractArray, step::AbstractFloat)
    return proximal!(p, similar(x), x, step)
end
function proximal(p::PenaltyFunction, x::T, step::AbstractFloat) where {T<:AbstractFloat}
    return proximal!(p, ones(T, 1), [x], step)[1]
end

StatsAPI.dof(p::PenaltyFunction, x::AbstractVector{T}) where {T<:AbstractFloat} = length(x)

########################################################################
##### Penalty functions
########################################################################

"""
    NoPenalty{T<:AbstractFloat}

No penalty (constant penalty), the proximal operator returns the same vector as input (identity map).
P(x) = 0
"""
struct NoPenalty{T<:AbstractFloat} <: PenaltyFunction{T} end
NoPenalty(args...; kwargs...) = NoPenalty{Float64}()
cost(::NoPenalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat} = zero(T)
function proximal!(
    p::NoPenalty{T}, out, x::AbstractArray{T}, step::T=one(T)
) where {T<:AbstractFloat}
    return copyto!(out, x)
end

isconvex(::NoPenalty) = true
isseparable(::NoPenalty) = true


"""
    SquaredL2Penalty{T<:AbstractFloat}

Squared L2 penalty, the proximal operator returns a scaled version.
P(x) = λ/2 Σi |xi|²
"""
struct SquaredL2Penalty{T<:AbstractFloat} <: PenaltyFunction{T}
    λ::T
    nonnegative::Bool

    function SquaredL2Penalty(λ::T; nonnegative::Bool=false) where {T<:AbstractFloat}
        λ >= 0 || throw(ArgumentError("penalty constant λ should be non-negative: $λ"))
        return new{T}(λ, nonnegative)
    end
end
function cost(p::SquaredL2Penalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    return p.λ / 2 * sum(abs2, x)
end
function proximal!(
    p::SquaredL2Penalty{T}, out, x::AbstractArray{T}, step::T=one(T)
) where {T<:AbstractFloat}
    #    return broadcast!(/, out, x, 1 + p.λ * step)
    nonnegative = p.nonnegative
    a = 1 / (1 + p.λ * step)
    @inbounds @simd for i in eachindex(out, x)
        out[i] = (nonnegative && x[i] <= 0) ? zero(T) : x[i] * a
    end
    return out
end

# Approximate shrinkage of coefficients
function StatsAPI.dof(p::SquaredL2Penalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    return length(x) / (1 + p.λ)
end

isconvex(::SquaredL2Penalty) = true
isseparable(::SquaredL2Penalty) = true


"""
    EuclideanPenalty{T<:AbstractFloat}

Euclidean norm penalty, the proximal operator returns a scaled version.
P(x) = λ √(Σi |xi|²)
"""
struct EuclideanPenalty{T<:AbstractFloat} <: PenaltyFunction{T}
    λ::T
    nonnegative::Bool

    function EuclideanPenalty(λ::T; nonnegative::Bool=false) where {T<:AbstractFloat}
        λ >= 0 || throw(ArgumentError("penalty constant λ should be non-negative: $λ"))
        return new{T}(λ, nonnegative)
    end
end
function cost(p::EuclideanPenalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    return p.λ * norm(x, 2)
end
function proximal!(
    p::EuclideanPenalty{T}, out, x::AbstractArray{T}, step::T=one(T)
) where {T<:AbstractFloat}
    # nn = p.nonnegative ? norm(broadcast!(max, out, x, 0), 2) : norm(x, 2)
    if p.nonnegative
        nn = zero(T)
        @inbounds @simd for xi in x
            nn += xi > 0 ? xi^2 : zero(T)
        end
        nn = sqrt(nn)
    else
        nn = norm(x, 2)
    end
    return rmul!(copyto!(out, x), (1 - p.λ * step / max(p.λ * step, nn)))
end

isconvex(::EuclideanPenalty) = true
isseparable(::EuclideanPenalty) = false


"""
    L1Penalty{T<:AbstractFloat}

L1 penalty, the proximal operator returns a soft-thresholded value.
P(x) = λ Σi |xi|
"""
struct L1Penalty{T<:AbstractFloat} <: PenaltyFunction{T}
    λ::T
    nonnegative::Bool

    function L1Penalty(λ::T; nonnegative::Bool=false) where {T<:AbstractFloat}
        λ >= 0 || throw(ArgumentError("penalty constant λ should be non-negative: $λ"))
        return new{T}(λ, nonnegative)
    end
end
cost(p::L1Penalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat} = p.λ * sum(abs, x)
function proximal!(
    p::L1Penalty{T}, out, x::AbstractArray{T}, step::T=one(T)
) where {T<:AbstractFloat}
    a = p.λ * step
    nonnegative = p.nonnegative

    @inbounds @simd for i in eachindex(out, x)
        out[i] = (nonnegative && x[i] <= 0) ? zero(T) : soft_threshold(x[i], a)
    end
    return out
end

function StatsAPI.dof(p::L1Penalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    return count(!iszero, x)
end

isconvex(::L1Penalty) = true
isseparable(::L1Penalty) = true


"""
    ElasticNetPenalty{T<:AbstractFloat}

ElasticNet penalty, a sum of SquaredL2Penalty and sparse L1Penalty.
P(x) = l1_ratio . λ Σi |xi| + (1 - l1_ratio) . λ/2 Σi |xi|²
"""
struct ElasticNetPenalty{T<:AbstractFloat} <: PenaltyFunction{T}
    λ::T
    l1_ratio::T
    nonnegative::Bool

    function ElasticNetPenalty(
        λ::T, l1_ratio::AbstractFloat=T(0.5); nonnegative::Bool=false
    ) where {T<:AbstractFloat}
        λ >= 0 || throw(ArgumentError("penalty constant λ should be non-negative: $λ"))
        0 <= l1_ratio <= 1 ||
            throw(ArgumentError("l1_ratio must be between 0 and 1: $(l1_ratio)"))
        return new{T}(λ, l1_ratio, nonnegative)
    end
end

function cost(p::ElasticNetPenalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    return p.λ * (p.l1_ratio * sum(abs, x) + (1 - p.l1_ratio) * sum(abs2, x) / 2)
end

function proximal!(
    p::ElasticNetPenalty{T}, out, x::AbstractArray{T}, step::T=one(T)
) where {T<:AbstractFloat}
    a = p.λ * step
    nonnegative = p.nonnegative
    l1_ratio = p.l1_ratio
    @inbounds @simd for i in eachindex(out, x)
        out[i] = if (nonnegative && x[i] <= 0)
            zero(T)
        else
            (soft_threshold(x[i], l1_ratio * a) / (1 + (1 - l1_ratio) * a))
        end
    end
    return out
end

# Approximate shrinkage of coefficients
function StatsAPI.dof(
    p::ElasticNetPenalty{T}, x::AbstractVector{T}
) where {T<:AbstractFloat}
    return count(!iszero, x) / (1 + p.λ)
end

isconvex(::ElasticNetPenalty) = true
isseparable(::ElasticNetPenalty) = true


"""
    BerhuPenalty{T<:AbstractFloat}

Berhu convex penalty, equivalent to L1Penalty at low values and SquaredL2Penalty at high values
(reverse of Huber loss).
Owen (2006) - A robust hybrid of lasso and ridge regression

P(x) = λl1_ratio . λ Σi |xi| + (1 - l1_ratio) . λ/2 Σi |xi|²
"""
struct BerhuPenalty{T<:AbstractFloat} <: PenaltyFunction{T}
    λ::T
    γ::T
    nonnegative::Bool

    function BerhuPenalty(
        λ::T, γ::AbstractFloat=T(1); nonnegative::Bool=false
    ) where {T<:AbstractFloat}
        λ >= 0 || throw(ArgumentError("penalty constant λ should be non-negative: $λ"))
        γ >= 0 || throw(ArgumentError("γ must be non-negative: $(γ)"))
        return new{T}(λ, γ, nonnegative)
    end
end

function cost(p::BerhuPenalty{T}, xi::T) where {T<:AbstractFloat}
    return abs(xi) <= p.γ ? p.λ * abs(xi) : p.λ * p.γ / 2 * (1 + (xi / p.γ)^2)
end
function cost(p::BerhuPenalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    return sum(xi -> cost(p, xi), x; init=zero(T))
end

function proximal!(
    p::BerhuPenalty{T}, out, x::AbstractArray{T}, step::T=one(T)
) where {T<:AbstractFloat}
    a = p.λ * step
    γ = p.γ
    nonnegative = p.nonnegative
    @inbounds @simd for i in eachindex(out, x)
        if p.nonnegative && x[i] <= 0
            out[i] = zero(T)
        elseif abs(x[i]) <= a + γ
            out[i] = soft_threshold(x[i], a)
        else
            out[i] = γ * x[i] / (a + γ)
        end
    end
    return out
end

# Approximate coefficient selection
function StatsAPI.dof(p::BerhuPenalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    return count(!iszero, x)
end

isconvex(::BerhuPenalty) = true
isseparable(::BerhuPenalty) = true


"""
    SCADPenalty{T<:AbstractFloat}

SCAD penalty, folded-concave penalty, generalization of the LASSO.
Fan & Li (2001) - Variable Selection via Nonconcave Penalized Likelihood and its Oracle Properties

P(x) = λ ∫_0^|x| dt I(t<=λ) + (γλ - t)_+ / (λ*(γ-1)) I (t>λ)
"""
struct SCADPenalty{T<:AbstractFloat} <: PenaltyFunction{T}
    λ::T
    γ::T
    nonnegative::Bool

    function SCADPenalty(
        λ::T, γ::AbstractFloat=T(3.7); nonnegative::Bool=false
    ) where {T<:AbstractFloat}
        λ >= 0 || throw(ArgumentError("penalty constant λ must be non-negative: $λ"))
        γ > 2 || throw(ArgumentError("γ must be greater than 2: $(γ)"))
        if nonnegative
            @warn "`nonnegative` argument is ignored for SCADPenalty."
        end
        return new{T}(λ, γ, false)
    end
end

function cost(p::SCADPenalty{T}, xi::T) where {T<:AbstractFloat}
    if abs(xi) <= p.λ
        return p.λ * abs(xi)
    elseif abs(x[i]) >= p.λ * p.γ
        return p.λ^2 * (p.γ + 1) / 2
    else
        return -(xi^2 - 2 * p.λ * p.γ * abs(xi) + p.λ^2) / (2 * (p.γ - 1))
    end
end
function cost(p::SCADPenalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    return sum(xi -> cost(p, xi), x; init=zero(T))
end

function convex_approx(p::SCADPenalty{T}, x::T, u::T, s::T) where {T}
    return 0.5 * (x - u)^2 + cost(p, x) / s
end

function proximal!(
    p::SCADPenalty{T}, out, x::AbstractArray{T}, step::T=one(T)
) where {T<:AbstractFloat}
    # Step is non-multiplicative
    # Gong et al. (2013) - A General Iterative Shrinkage and Thresholding Algorithm for
    # Non-convex Regularized Optimization Problems
    λ = p.λ
    γ = p.γ

    if step <= 0
        return copy!(out, x)
    end

    @inbounds @simd for i in eachindex(out, x)
        #        if p.nonnegative && x[i] <= 0
        #            out[i] = zero(T)
        #        else
        x1 = sign(x[i]) * min(λ, max(0, abs(x[i]) - λ / step))
        x2 = sign(x[i]) * min(λ * γ, max(λ, (abs(x[i]) * (γ - 1) - λ * γ / step) / (γ - 2)))
        x3 = sign(x[i]) * min(λ * γ, abs(x[i]))
        sols = [x1, x2, x3]
        ind = argmin(convex_approx.(p, sols, x[i], step))
        out[i] = sols[ind]
    end
    return out
end

# Approximate coefficient selection
function StatsAPI.dof(p::SCADPenalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    return count(!iszero, x)
end

isconvex(::SCADPenalty) = false
isseparable(::SCADPenalty) = true


"""
    CappedL1Penalty{T<:AbstractFloat}

Capped L1 penalty, folded-concave penalty, a bridge between the L1Penalty and the Hard thresholding penalty.
For γ -> Inf, the MC+ penalty becomes equivalent to the L1Penalty.
For γ -> 0+, the MC+ penalty tends to the L0 penalty or hard thresholding penalty.

P(x) = λ min(abs(x), γ)
"""
struct CappedL1Penalty{T<:AbstractFloat} <: PenaltyFunction{T}
    λ::T
    γ::T
    nonnegative::Bool

    function CappedL1Penalty(
        λ::T, γ::AbstractFloat=T(3); nonnegative::Bool=false
    ) where {T<:AbstractFloat}
        λ >= 0 || throw(ArgumentError("penalty constant λ must be non-negative: $λ"))
        γ > 0 || throw(ArgumentError("γ must be greater than 1: $(γ)"))
        if nonnegative
            @warn "`nonnegative` argument is ignored for CappedL1Penalty."
        end
        return new{T}(λ, γ, false)
    end
end

cost(p::CappedL1Penalty{T}, xi::T) where {T<:AbstractFloat} = p.λ * min(abs(xi), p.γ)
function cost(p::CappedL1Penalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    return sum(xi -> cost(p, xi), x; init=zero(T))
end

function convex_approx(p::CappedL1Penalty{T}, x::T, u::T, s::T) where {T}
    return 0.5 * (x - u)^2 + cost(p, x) / s
end

function proximal!(
    p::CappedL1Penalty{T}, out, x::AbstractArray{T}, step::T=one(T)
) where {T<:AbstractFloat}
    # Step is non-multiplicative
    # Gong et al. (2013) - A General Iterative Shrinkage and Thresholding Algorithm for
    # Non-convex Regularized Optimization Problems
    λ = p.λ
    γ = p.γ

    if step <= 0
        return copy!(out, x)
    end

    @inbounds @simd for i in eachindex(out, x)
        #        if p.nonnegative && x[i] <= 0
        #            out[i] = zero(T)
        #        else
        x1 = sign(x[i]) * max(γ, abs(x[i]))
        x2 = sign(x[i]) * min(γ, max(0, abs(x[i]) - λ / step))
        sols = [x1, x2]
        ind = argmin(convex_approx.(p, sols, x[i], step))
        out[i] = sols[ind]
    end
    return out
end

# Approximate coefficient selection
function StatsAPI.dof(p::CappedL1Penalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    return count(!iszero, x)
end

isconvex(::CappedL1Penalty) = false
isseparable(::CappedL1Penalty) = true


"""
    MCPPenalty{T<:AbstractFloat}

MC+ penalty, folded-concave penalty, a bridge between the L1Penalty and the Hard thresholding penalty.
For γ -> Inf, the MC+ penalty becomes equivalent to the L1Penalty.
For γ -> 1+, the MC+ penalty tends to the L0 penalty or hard thresholding penalty.

Zhang (2010) - Nearly unbiased variable selection under minimax concave penalty

P(x) = λ ∫_0^|x| dt (1 - t/γλ)_+
"""
struct MCPPenalty{T<:AbstractFloat} <: PenaltyFunction{T}
    λ::T
    γ::T
    nonnegative::Bool

    function MCPPenalty(
        λ::T, γ::AbstractFloat=T(3); nonnegative::Bool=false
    ) where {T<:AbstractFloat}
        λ >= 0 || throw(ArgumentError("penalty constant λ must be non-negative: $λ"))
        γ > 1 || throw(ArgumentError("γ must be greater than 1: $(γ)"))
        if nonnegative
            @warn "`nonnegative` argument is ignored for MCPPenalty."
        end
        return new{T}(λ, γ, false)
    end
end

function cost(p::MCPPenalty{T}, xi::T) where {T<:AbstractFloat}
    if abs(xi) < p.λ * p.γ
        return p.λ * abs(xi) - xi^2 / (2 * p.γ)
    else
        return p.λ^2 * p.γ / 2
    end
end
function cost(p::MCPPenalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    return sum(xi -> cost(p, xi), x; init=zero(T))
end

function convex_approx(p::MCPPenalty{T}, x::T, u::T, s::T) where {T}
    return 0.5 * (x - u)^2 + cost(p, x) / s
end
function aux_func2(p::MCPPenalty{T}, x::T, u::T, s::T) where {T}
    return (x - abs(u))^2 / 2 + p.λ * x / s - x^2 / (2 * p.γ)
end

function proximal!(
    p::MCPPenalty{T}, out, x::AbstractArray{T}, step::T=one(T)
) where {T<:AbstractFloat}
    # proximal(p, x) = abs(x) <= p.λ ? 0 :
    #                  abs(x) < p.λ * p.γ ? sign(x) * (abs(x) - p.λ) / (1 - 1/p.γ) : x
    #
    # Step is non-multiplicative
    # Gong et al. (2013) - A General Iterative Shrinkage and Thresholding Algorithm for
    # Non-convex Regularized Optimization Problems
    λ = p.λ
    γ = p.γ

    if step <= 0
        return copy!(out, x)
    end

    a = p.λ * p.γ
    @inbounds @simd for i in eachindex(out, x)
        #        if p.nonnegative && x[i] <= 0
        #            out[i] = zero(T)
        #        else
        z1 = 0
        z2 = a
        z3 = min(a, max(0, (γ * abs(x[i]) - a / step) / (γ - 1)))
        zsols = [z1, z2, z3]
        ind = argmin(aux_func2.(p, zsols, x[i], step))
        z = zsols[ind]

        x1 = sign(x[i]) * z
        x2 = sign(x[i]) * min(a, abs(x[i]))
        sols = [x1, x2]
        ind = argmin(convex_approx.(p, sols, x[i], step))
        out[i] = sols[ind]
    end
    return out
end

# Approximate coefficient selection
function StatsAPI.dof(p::MCPPenalty{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    return count(!iszero, x)
end

isconvex(::MCPPenalty) = false
isseparable(::MCPPenalty) = true


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

    function EndStepRange{T}(
        start::Integer, step::Integer, stop::Integer
    ) where {T<:Integer}
        return new{T}(Base.convert.(T, (start, step, stop))...)
    end
end
function EndStepRange(
    start::I1, step::I2, stop::I3
) where {I1<:Integer,I2<:Integer,I3<:Integer}
    return EndStepRange{promote_type(I1, I2, I3)}(start, step, stop)
end

(::Colon)(start::Integer, stop::End) = EndStepRange(start, one(start), stop.offset)
(::Colon)(start::Integer, step::Integer, stop::End) = EndStepRange(start, step, stop.offset)

torange(p::EndStepRange, n::Integer) = (p.start):(p.step):(n - p.stop)
function torange(p::Vector, n::Integer)
    return if all(1 .<= p .<= n)
        sort(p)
    else
        throw(BoundsError("attempt to access $n-element Vector at index [$(p)]"))
    end
end
function torange(p::AbstractRange, n::Integer)
    return if (last(p) <= n)
        p
    else
        throw(BoundsError("attempt to access $n-element Vector at index [$(p)]"))
    end
end

function excludeindex(p::Vector, i::Integer)
    i in p || return p

    return filter(!=(i), p)
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
    return ran
end

RangeLikeType{T} = Union{UnitRange{<:T},StepRange{<:T,<:T},EndStepRange{<:T},Vector{<:T}}


mutable struct RangedPenalties{T<:AbstractFloat,N<:Integer} <: AbstractRangedPenalties{T}
    ranges::Vector{RangeLikeType{N}}
    penalties::Vector{PenaltyFunction}
    notinrange::Vector{N}
    isconcrete::Bool

    function RangedPenalties{T,N}(
        ranges::AbstractVector, penalties::AbstractVector{<:PenaltyFunction{T}}
    ) where {T<:AbstractFloat,N<:Integer}
        any(p isa RangedPenalties for p in penalties) && throw(
            ArgumentError(
                "RangedPenalties penalties should not be of type RangedPenalties: $(typeof.(penalties))",
            ),
        )
        length(ranges) == length(penalties) || throw(
            ArgumentError(
                "ranges and penalties should have the same number of elements: $(length(ranges)) != $(length(penalties))",
            ),
        )

        return new{T,N}(ranges, penalties, zeros(N, 0), false)
    end
end

function RangedPenalties(
    ranges::AbstractVector,
    penalties::AbstractVector{<:PenaltyFunction{T}},
    n::Union{Nothing,Integer}=nothing,
) where {T<:AbstractFloat}
    length(penalties) > 0 ||
        throw(ArgumentError("RangedPenalties should contain at least one PenaltyFunction"))
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
    p::RangedPenalties, n::Integer, intercept_index::Union{Nothing,Integer}=nothing
)
    ranges = p.ranges
    penalties = p.penalties
    # Check that ranges are non-overlapping
    notinrange = Set(1:n)
    @inbounds for j in eachindex(ranges, penalties)
        ran = torange(ranges[j], n)
        sran = Set(ran)
        if intersect(notinrange, sran) != sran
            error("Overlapping ranges are not allowed: $(ranges)")
        end
        if !isnothing(intercept_index)
            # Remove intercept_index from range
            ran = excludeindex(ran, intercept_index)
            sran = Set(ran)
        end
        ## Transform End-ranges to Base ranges
        ranges[j] = ran
        ## Remoev used ranges
        setdiff!(notinrange, sran)
    end
    ## Store missing indices in notinrange
    p.notinrange = sort!(collect(notinrange))
    p.isconcrete = true
    return p
end

function concrete(
    p::RangedPenalties, n::Integer, intercept_index::Union{Nothing,Integer}=nothing
)
    new_p = deepcopy(p)
    return concrete!(new_p, n, intercept_index)
end

function cost(p::RangedPenalties{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    isconcrete(p) || error(
        "RangedPenalties not concrete, call `concrete!` beforehand to make sure the ranges are well-defined.",
    )
    ranges = p.ranges
    penalties = p.penalties

    n = length(x)
    r = zero(T)
    for (ran, pen) in zip(ranges, penalties)
        r += cost(pen, view(x, ran))
    end
    return r
end

function proximal!(
    p::RangedPenalties{T}, out, x::AbstractArray{T}, step::T=one(T)
) where {T<:AbstractFloat}
    isconcrete(p) || error(
        "RangedPenalties not concrete, call `concrete!` beforehand to make sure the ranges are well-defined.",
    )
    ranges = p.ranges
    penalties = p.penalties

    n = length(x)
    for (ran, pen) in zip(ranges, penalties)
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

function proximal!(
    p::RangedPenalties{T},
    out::AbstractVector,
    index::Integer,
    x::AbstractVector,
    step::AbstractFloat,
) where {T<:AbstractFloat}
    isconcrete(p) || error(
        "RangedPenalties not concrete, call `concrete!` beforehand to make sure the ranges are well-defined.",
    )
    ranges = p.ranges
    penalties = p.penalties

    done = false
    for (ran, pen) in zip(ranges, penalties)
        if in(index, ran)
            proximal!(pen, view(out, index), view(x, index), step)
            done = true
            break
        end
    end
    if !done && !isempty(p.notinrange)
        if !in(index, p.notinrange)
            n = length(x)
            error("index not found in range $(1:n): $index")
        end
        proximal!(NoPenalty{T}(), view(out, index), view(x, index), step)
    end
    return out
end

function StatsAPI.dof(p::RangedPenalties{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    isconcrete(p) || error(
        "RangedPenalties not concrete, call `concrete!` beforehand to make sure the ranges are well-defined.",
    )
    ranges = p.ranges
    penalties = p.penalties

    # For ranges without penalty, count 1 dof per index
    r = convert(T, length(p.notinrange))
    @inbounds for (ran, pen) in zip(ranges, penalties)
        r += dof(pen, view(x, ran))
    end
    return r
end

isconvex(p::RangedPenalties) = all(isconvex.(p.penalties))
function isseparable(p::RangedPenalties)
    ranges = p.ranges
    penalties = p.penalties

    if !isconcrete(p)
        error("isseparable is only defined on concrete RangedPenalties.")
    end
    @inbounds for (ran, pen) in zip(ranges, penalties)
        if !isseparable(pen) && length(ran) > 1
            return false
        end
    end
    return true
end
