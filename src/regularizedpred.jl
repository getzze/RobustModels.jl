using SparseArrays: sparse
using LinearAlgebra: cholesky


####################################
### AbstractRegularizedPred methods
####################################

StatsAPI.modelmatrix(p::AbstractRegularizedPred) = p.X

StatsAPI.coef(p::AbstractRegularizedPred) = p.beta0

function StatsAPI.vcov(p::AbstractRegularizedPred, wt::AbstractVector)
    wXt = isempty(wt) ? modelmatrix(p)' : (modelmatrix(p) .* wt)'
    return inv(Hermitian(float(Matrix(wXt * modelmatrix(p)))))
end

function projectionmatrix(p::AbstractRegularizedPred, wt::AbstractVector)
    wXt = isempty(wt) ? modelmatrix(p)' : (modelmatrix(p) .* wt)'
    return Hermitian(modelmatrix(p) * vcov(p, wt) * wXt)
end

function StatsAPI.leverage(p::AbstractRegularizedPred, wt::AbstractVector)
    return diag(projectionmatrix(p, wt))
end

function leverage_weights(p::AbstractRegularizedPred, wt::AbstractVector)
    return sqrt.(1 .- leverage(p, wt))
end

"""
    linpred!(out, p::RidgePred{T}, f::Real=1.0)
Overwrite `out` with the linear predictor from `p` with factor `f`
The effective coefficient vector, `p.scratchbeta`, is evaluated as `p.beta0 .+ f * p.delbeta`,
and `out` is updated to `p.X * p.scratchbeta`
"""
function linpred!(out, p::AbstractRegularizedPred, f::Real=1.0)
    return mul!(
        out,
        p.X,
        iszero(f) ? p.beta0 : broadcast!(muladd, p.scratchbeta, f, p.delbeta, p.beta0),
    )
end

"""
   linpred(p::RidgePred, f::Read=1.0)
Return the linear predictor `p.X * (p.beta0 .+ f * p.delbeta)`
"""
function linpred(p::AbstractRegularizedPred, f::Real=1.0)
    return linpred!(Vector{eltype(p.X)}(undef, size(p.X, 1)), p, f)
end

"""
   installbeta!(p::LinPred, f::Real=1.0)
Install `pbeta0 .+= f * p.delbeta` and zero out `p.delbeta`.  Return the updated `p.beta0`.
"""
function installbeta!(p::AbstractRegularizedPred, f::Real=1.0)
    beta0 = p.beta0
    delbeta = p.delbeta
    @inbounds for i in eachindex(beta0, delbeta)
        beta0[i] += delbeta[i] * f
        delbeta[i] = 0
    end
    return beta0
end



##############################
###  Ridge predictor
##############################

"""
    cat_ridge_matrix(X::AbstractMatrix{T}, λ::T, G::AbstractMatrix{T}) where {T}
        = vcat(X,  λ * G)

Construct the extended model matrix by vertically concatenating the regularizer to X.
"""
function cat_ridge_matrix(X::AbstractMatrix{T}, λ::T, G::AbstractMatrix{T}) where {T}
    return vcat(X, λ * G)
end

"""
    RidgePred

Regularized predictor using ridge regression on the `p` features.

# Members

- `X`: model matrix
- `λ`: shrinkage parameter of the regularizer
- `G`: regularizer matrix of size p×p.
- `βprior`: regularizer prior of the coefficient values. Default to `zeros(p)`.
- `pred`: the non-regularized predictor using an extended model matrix.
- `pivot`: for `DensePredChol`, if the decomposition was pivoted.
- `scratchbeta`: scratch vector of length `p`, used in [`linpred!`](@ref) method
"""
mutable struct RidgePred{T<:BlasReal,M<:AbstractMatrix,P<:LinPred} <: LinPred
    X::M                    # model matrix
    sqrtλ::T                # sqrt of the shrinkage parameter λ
    G::M                    # regularizer matrix
    βprior::Vector{T}       # coefficients prior
    pred::P                 # predictor
    pivot::Bool             # pivoted decomposition
    scratchbeta::Vector{T}
    scratchy::Vector{T}
    scratchwt::Vector{T}
end

function RidgePred(
    ::Type{P}, X::M, λ::T, G::M, βprior::Vector{T}, pivot::Bool=false
) where {M<:Union{SparseMatrixCSC{T},Matrix{T}},P<:LinPred} where {T<:BlasReal}
    λ >= 0 || throw(DomainError(λ, "the shrinkage parameter should be non-negative"))

    m1, m2 = size(G)
    m1 == m2 || throw(DimensionMismatch("the regularization matrix should be square"))
    n, m = size(X)
    m1 == m || throw(
        DimensionMismatch(
            "the regularization matrix should of size p×p," *
            " with p the number of predictor in matrix X:" *
            " size(G)=$(size(G)) != size(X, 2)=$(m)",
        ),
    )
    ll = size(βprior, 1)
    ll in (0, m) || throw(DimensionMismatch("length of βprior is $ll, must be $m or 0"))

    sqrtλ = √λ
    pred = if P in (SparsePredCG, DensePredCG)
        cgpred(cat_ridge_matrix(X, sqrtλ, G), pivot)
    elseif P in (DensePredQR,)
        qrpred(cat_ridge_matrix(X, sqrtλ, G), pivot)
    elseif P in (SparsePredChol, DensePredChol)
        cholpred(cat_ridge_matrix(X, sqrtλ, G), pivot)
    else
        error("Undefined RidgePred with underlying Linpred type: $(P)")
    end

    return RidgePred{T,typeof(X),typeof(pred)}(
        X,
        sqrtλ,
        G,
        (ll == 0) ? zeros(T, m) : βprior,
        pred,
        pivot,
        zeros(T, m),
        zeros(T, n + m),
        ones(T, n + m),
    )
end

function postupdate_λ!(r::RidgePred)
    n, m = size(r.X)
    # Update the extended model matrix with the new value
    GG = r.sqrtλ * r.G
    @views r.pred.X[(n + 1):(n + m), :] .= GG

    # Update other fields
    # TODO: update DensePredQR
    if isa(r.pred, DensePredChol)
        # Recompute the cholesky decomposition
        X = r.pred.X
        F = Hermitian(float(X'X))
        T = eltype(F)
        r.pred.chol =
            r.pivot ? pivoted_cholesky!(F; tol=-one(T), check=false) : cholesky!(F)
    elseif isa(r.pred, SparsePredChol)
        # Update Xt
        @views r.pred.Xt[:, (n + 1):(n + m)] .= GG'
    end
end

function Base.getproperty(r::RidgePred, s::Symbol)
    if s ∈ (:λ, :lambda)
        (r.sqrtλ)^2
    elseif s ∈ (:beta0, :delbeta)
        getproperty(r.pred, s)
    else
        getfield(r, s)
    end
end

function Base.setproperty!(r::RidgePred, s::Symbol, v)
    if s ∈ (:λ, :lambda)
        v >= 0 || throw(DomainError(v, "the shrinkage parameter should be non-negative"))
        # Update the square root value
        setfield!(r, :sqrtλ, √v)
        postupdate_λ!(r)
    else
        error(
            "cannot set any property of RidgePred except" *
            " the shrinkage parameter λ (lambda): $(s)",
        )
        # setfield!(r, s, v)
    end
end

function Base.propertynames(r::RidgePred, private::Bool=false)
    if private
        (
            :X,
            :λ,
            :lambda,
            :G,
            :βprior,
            :pred,
            :beta0,
            :delbeta,
            :pivot,
            :sqrtλ,
            :scratchbeta,
            :scratchy,
            :scratchwt,
        )
    else
        (:X, :λ, :lambda, :G, :βprior, :pred, :beta0, :delbeta)
    end
end

function cgpred(
    X::StridedMatrix{T},
    λ::Real,
    G::AbstractMatrix{<:Real},
    βprior::AbstractVector{<:Real}=zeros(T, size(X, 2)),
    pivot::Bool=false,  # placeholder
) where {T<:AbstractFloat}
    return RidgePred(
        DensePredCG, X, Base.convert(T, λ), Matrix{T}(G), Vector{T}(βprior), pivot
    )
end

function GLM.cholpred(
    X::StridedMatrix{T},
    λ::Real,
    G::AbstractMatrix{<:Real},
    βprior::AbstractVector{<:Real}=zeros(T, size(X, 2)),
    pivot::Bool=false,
) where {T<:AbstractFloat}
    return RidgePred(
        DensePredChol, X, Base.convert(T, λ), Matrix{T}(G), Vector{T}(βprior), pivot
    )
end

function qrpred(
    X::AbstractMatrix{T},
    λ::Real,
    G::AbstractMatrix{<:Real},
    βprior::AbstractVector{<:Real}=zeros(T, size(X, 2)),
    pivot::Bool=false,  # placeholder
) where {T<:AbstractFloat}
    # No sparse version exists, force both matrices to be denses
    return RidgePred(
        DensePredQR,
        Matrix{T}(X),
        Base.convert(T, λ),
        Matrix{T}(G),
        Vector{T}(βprior),
        pivot,
    )
end

function cgpred(
    X::SparseMatrixCSC{T},
    λ::Real,
    G::AbstractMatrix{<:Real},
    βprior::AbstractVector{<:Real}=zeros(T, size(X, 2)),
    pivot::Bool=false,  # placeholder
) where {T<:AbstractFloat}
    return RidgePred(
        SparsePredCG, X, Base.convert(T, λ), SparseMatrixCSC{T}(G), Vector{T}(βprior), pivot
    )
end

function cholpred(
    X::SparseMatrixCSC{T},
    λ::Real,
    G::AbstractMatrix{<:Real},
    βprior::AbstractVector{<:Real}=zeros(T, size(X, 2)),
    pivot::Bool=false,
) where {T<:AbstractFloat}
    return RidgePred(
        SparsePredChol,
        X,
        Base.convert(T, λ),
        SparseMatrixCSC{T}(G),
        Vector{T}(βprior),
        pivot,
    )
end


function cgpred(X::SparseMatrixCSC, λ, G::AbstractVector, βprior::AbstractVector, args...)
    return cgpred(X, λ, spdiagm(0 => G), βprior, args...)
end

function cholpred(X::SparseMatrixCSC, λ, G::AbstractVector, βprior::AbstractVector, args...)
    return cholpred(X, λ, spdiagm(0 => G), βprior, args...)
end

function cgpred(X::StridedMatrix, λ, G::AbstractVector, βprior::AbstractVector, args...)
    return cgpred(X, λ, diagm(0 => G), βprior, args...)
end

function cholpred(X::StridedMatrix, λ, G::AbstractVector, βprior::AbstractVector, args...)
    return cholpred(X, λ, diagm(0 => G), βprior, args...)
end

function qrpred(X::AbstractMatrix, λ, G::AbstractVector, βprior::AbstractVector, args...)
    return qrpred(X, λ, diagm(0 => G), βprior, args...)
end

function delbeta!(
    p::RidgePred{T}, r::AbstractVector{T}, wt::AbstractVector{T}
) where {T<:BlasReal}
    n, m = size(p.X)
    # Fill response
    copyto!(p.scratchy, r)
    # yprior = sqrt(p.λ/2) * p.G * (p.βprior - p.pred.beta0)
    broadcast!(-, p.scratchbeta, p.βprior, p.pred.beta0)
    @views mul!(p.scratchy[(n + 1):end], p.G, p.scratchbeta, p.sqrtλ, 0)

    # Fill weights
    copyto!(p.scratchwt, wt)

    # Compute Δβₙ from (XᵀWₙX + λGᵀG)⁻¹ (XᵀWₙrₙ + λGᵀG(βprior - βₙ))
    delbeta!(p.pred, p.scratchy, p.scratchwt)

    return p
end

"""
    extendedmatrix(p::RidgePred{T})

Returns the extended model matrix for a Ridge predictor, of size `(n+m) x m`, where
`n` is the number of observations and `m` the number of coefficients.
"""
extendedmodelmatrix(p::RidgePred) = p.pred.X

"""
    extendedweights(p::RidgePred{T}, wt::AbstractVector) where {T<:BlasReal}

Returns a weights vector to multiply with `extendedmodelmatrix(p::RidgePred)`.
It has a size of 0 or `n + m`, where `n` is the number of observations and
`m` the number of coefficients.
"""
function extendedweights(p::RidgePred{T}, wt::AbstractVector) where {T<:BlasReal}
    n, m = size(p.X)
    k = length(wt)
    if k == 0
        return similar(wt, 0)
    elseif k == n
        return vcat(Vector{T}(wt), ones(T, m))
    elseif k == n + m
        return wt
    else
        throw(DimensionMismatch("weights for RidgePred should be of size 0, n or n+m"))
    end
end

function StatsAPI.vcov(p::RidgePred, wt::AbstractVector)
    # returns (XᵀWX + λGᵀG)⁻¹  = (extXᵀ . extW . extX)⁻¹
    wwt = extendedweights(p, wt)
    wXt = isempty(wwt) ? extendedmodelmatrix(p)' : (extendedmodelmatrix(p) .* wwt)'
    return inv(Hermitian(float(Matrix(wXt * extendedmodelmatrix(p)))))
end
