
using LinearAlgebra: BlasReal, QRCompactWY, Diagonal, Hermitian, LowerTriangular, UpperTriangular, transpose, cholesky, cholesky!, qr, mul!, inv, diag, diagm, ldiv!


#################
modelmatrix(p::LinPred) = p.X
vcov(p::LinPred, wt::AbstractVector) = inv(Hermitian(float(Matrix(modelmatrix(p)' * (wt .* modelmatrix(p))))))
projectionmatrix(p::LinPred, wt::AbstractVector) = Hermitian(modelmatrix(p) * vcov(p, wt) * modelmatrix(p)' .* wt)


modelmatrix(m::RobustLinearModel) = modelmatrix(m.pred)

vcov(m::RobustLinearModel) = vcov(m.pred, workingweights(m.resp))

"""
    projectionmatrix(m::RobustLinearModel)

The robust projection matrix from the predictor: X (X' W X)⁻¹ X' W
"""
projectionmatrix(m::RobustLinearModel) = projectionmatrix(m.pred, workingweights(m.resp))

function leverage_weights(m::RobustLinearModel)
    w = weights(m.resp)
    v = inv(Hermitian(float(modelmatrix(m)' * (w .* modelmatrix(m)))))
    h = diag(Hermitian(modelmatrix(m) * v * modelmatrix(m)' .* w))
    sqrt.(1 .- h)
end




###
### From GLM, for information
###
#"""
#    linpred!(out, p::LinPred, f::Real=1.0)
#Overwrite `out` with the linear predictor from `p` with factor `f`
#The effective coefficient vector, `p.scratchbeta`, is evaluated as `p.beta0 .+ f * p.delbeta`,
#and `out` is updated to `p.X * p.scratchbeta`
#"""
#function linpred!(out, p::LinPred, f::Real=1.)
#    mul!(out, p.X, iszero(f) ? p.beta0 : broadcast!(muladd, p.scratchbeta, f, p.delbeta, p.beta0))
#end

#"""
#    linpred(p::LinPred, f::Read=1.0)
#Return the linear predictor `p.X * (p.beta0 .+ f * p.delbeta)`
#"""
#linpred(p::LinPred, f::Real=1.) = linpred!(Vector{eltype(p.X)}(undef, size(p.X, 1)), p, f)

#"""
#    installbeta!(p::LinPred, f::Real=1.0)
#Install `pbeta0 .+= f * p.delbeta` and zero out `p.delbeta`.  Return the updated `p.beta0`.
#"""
#function installbeta!(p::LinPred, f::Real=1.)
#    beta0 = p.beta0
#    delbeta = p.delbeta
#    @inbounds for i = eachindex(beta0,delbeta)
#        beta0[i] += delbeta[i]*f
#        delbeta[i] = 0
#    end
#    beta0
#end

"""
    SparsePredChol{T<:BlasReal} <: LinPred
    
A LinPred type with a sparse Cholesky factorization of X'X

# Members

- `X`: model matrix of size n×p with n ≥ p. Should be full column rank.
- `beta0`: base coefficient vector of length p
- `delbeta`: increment to coefficient vector, also of length p
- `scratchbeta`: scratch vector of length p, used in [`linpred!`](@ref) method
- `chol`: a Cholesky object created from X'X, possibly using row weights.
"""
SparsePredChol



"""
    DensePredCG

A `LinPred` type with Conjugate Gradient and a dense `X`

# Members

- `X`: Model matrix of size `n` × `p` with `n ≥ p`.  Should be full column rank.
- `beta0`: base coefficient vector of length `p`
- `delbeta`: increment to coefficient vector, also of length `p`
- `scratchbeta`: scratch vector of length `p`, used in [`linpred!`](@ref) method
"""
mutable struct DensePredCG{T<:BlasReal} <: DensePred
    X::Matrix{T}                  # model matrix
    beta0::Vector{T}              # base coefficient vector
    delbeta::Vector{T}            # coefficient increment
    scratchbeta::Vector{T}
    scratchm1::Matrix{T}
    scratchr1::Vector{T}
    function DensePredCG{T}(X::Matrix{T}, beta0::Vector{T}) where T
        n, p = size(X)
        length(beta0) == p || throw(DimensionMismatch("length(β0) ≠ size(X,2)"))
        new{T}(X, beta0, zeros(T,p), zeros(T,p), zeros(T, (n,p)), zeros(T, n))
    end
    function DensePredCG{T}(X::Matrix{T}) where T
        n, p = size(X)
        new{T}(X, zeros(T, p), zeros(T,p), zeros(T,p), zeros(T, (n,p)), zeros(T, n))
    end
end
DensePredCG(X::Matrix, beta0::Vector) = DensePredCG{eltype(X)}(X, beta0)
DensePredCG(X::Matrix{T}) where T = DensePredCG{T}(X, zeros(T, size(X,2)))
convert(::Type{DensePredCG{T}}, X::Matrix{T}) where {T} = DensePredCG{T}(X, zeros(T, size(X, 2)))

"""
    delbeta!(p::LinPred, r::Vector)

Evaluate and return `p.delbeta` the increment to the coefficient vector from residual `r`
"""
function delbeta!(p::DensePredCG{T}, r::AbstractVector{T}) where T<:BlasReal
    lsqr!(p.delbeta, p.X, r; log=false)
    return p
end

cgpred(X::StridedMatrix) = DensePredCG(X)

function delbeta!(p::DensePredCG{T}, r::AbstractVector{T}, wt::AbstractVector{T}) where T<:BlasReal
    ## Use views, do not create new objects
#    scr = transpose(broadcast!(*, p.scratchm1, wt, p.X))
#    cg!(p.delbeta, Hermitian(mul!(p.scratchm2, scr, p.X), :U), mul!(p.scratchbeta, scr, r))
    sqwt = copyto!(p.scratchr1, sqrt.(wt))
    scr = broadcast!(*, p.scratchm1, sqwt, p.X)
    lsqr!(p.delbeta, scr, broadcast!(*, p.scratchr1, sqwt, r); log=false)
    p
end


"""
    SparsePredCG

A `LinPred` type with Conjugate Gradient and a sparse `X`

# Members

- `X`: Model matrix of size `n` × `p` with `n ≥ p`.  Should be full column rank.
- `beta0`: base coefficient vector of length `p`
- `delbeta`: increment to coefficient vector, also of length `p`
- `scratchbeta`: scratch vector of length `p`, used in [`linpred!`](@ref) method
"""
mutable struct SparsePredCG{T,M<:SparseMatrixCSC} <: LinPred
    X::M                           # model matrix
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    scratchm1::M
    scratchm2::M
    scratchr1::Vector{T}
end
function SparsePredCG(X::SparseMatrixCSC{T}) where T
    n, p = size(X)
    return SparsePredCG{eltype(X),typeof(X)}(
        X,
        zeros(T, p),
        zeros(T, p),
        zeros(T, p),
        similar(X),
        zeros(T, (p,p)),
        zeros(T, n),
    )
end

cgpred(X::SparseMatrixCSC) = SparsePredCG(X)

function delbeta!(p::SparsePredCG{T}, r::AbstractVector{T}, wt::AbstractVector{T}) where T
#    scr = mul!(p.scratchm1, Diagonal(wt), p.X)
#    cg!(p.delbeta, Hermitian(mul!(transpose(scr), p.X), :U), mul!(p.scratchr1, transpose(scr), r))
    scr = transpose(broadcast!(*, p.scratchm1, wt, p.X))
    cg!(p.delbeta, Hermitian(mul!(p.scratchm2, scr, p.X), :U), mul!(p.scratchbeta, scr, r))
    p
end


########
###     Ridge predictor
########

#function Base.float(::Type{S}, x::AbstractArray{T}) where {S<:AbstractFloat, T}
#    if !isconcretetype(T)
#        error("`float` not defined on abstractly-typed arrays; please convert to a more specific type")
#    end
#    convert(AbstractArray{typeof(convert(S, zero(T)))}, A)
#end

"""
    cat_ridge_matrix(X::AbstractMatrix{T}, λ::T, G::AbstractMatrix{T}) where T = vcat(X,  λ * G)
Construct the extended model matrix by vertically concatenating the regularizer to X.
"""
cat_ridge_matrix(X::AbstractMatrix{T}, λ::T, G::AbstractMatrix{T}) where T = vcat(X,  λ * G)

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
mutable struct RidgePred{T<:BlasReal, M<:AbstractMatrix, P<:LinPred} <: AbstractRegularizedPred{T}
    X::M                        # model matrix
    sqrthalfλ::T                # sqrt of half of the shrinkage parameter λ
    G::M                        # regularizer matrix
    βprior::Vector{T}           # coefficients prior
    pred::P
    pivot::Bool
    scratchbeta::Vector{T}
    scratchy::Vector{T}
    scratchwt::Vector{T}
end
function RidgePred(::Type{P}, X::M, λ::T, G::M, βprior::Vector{T}, pivot::Bool=false) where
                    {M<:Union{SparseMatrixCSC{T}, Matrix{T}}, P<:LinPred} where {T<:BlasReal}
    λ >= 0 || throw(DomainError(λ, "the shrinkage parameter should be non-negative"))

    m1, m2 = size(G)
    m1 == m2 || throw(DimensionMismatch("the regularization matrix should be square"))
    n, m = size(X)
    m1 == m || throw(DimensionMismatch("the regularization matrix should of size p×p, with p the number of predictor in matrix X: size(G)=$(size(G)) != size(X, 2)=$(m)"))
    ll = size(βprior, 1)
    ll == 0 || ll == m || throw(DimensionMismatch("length of βprior is $ll, must be $m or 0"))

    sqrthalfλ = √(λ/2)
    pred = if P == DensePredChol
        P(cat_ridge_matrix(X, sqrthalfλ, G), pivot)
    else
        P(cat_ridge_matrix(X, sqrthalfλ, G))
    end
    RidgePred{T, typeof(X), typeof(pred)}(X, sqrthalfλ, G, if ll==0; zeros(T, m) else βprior end, pred, pivot, zeros(T, m), zeros(T, n+m), ones(T, n+m))
end

function postupdate_λ!(r::RidgePred)
    n,m = size(r.X)
    # Update the extended model matrix with the new value
    GG = r.sqrthalfλ * r.G
    @views r.pred.X[n+1:n+m, :] .= GG
    if isa(r.pred, DensePredChol)
        # Recompute the cholesky decomposition
        X = r.pred.X
        F = Hermitian(float(X'X))
        T = eltype(F)
        r.pred.chol = r.pivot ? cholesky!(F, Val(true), tol = -one(T), check = false) : cholesky!(F)
    elseif isa(r.pred, SparsePredChol)
        # Update Xt
        @views r.pred.Xt[:, n+1:n+m] .= GG'
    end
end

function Base.getproperty(r::RidgePred, s::Symbol)
    if s ∈ (:λ, :lambda)
        2 * (r.sqrthalfλ)^2
    elseif s ∈ (:beta0, :delbeta)
        getproperty(r.pred, s)
    else
        getfield(r, s)
    end
end

function Base.setproperty!(r::RidgePred, s::Symbol, v)
    if s ∈ (:λ, :lambda)
        v >= 0 || throw(DomainError(λ, "the shrinkage parameter should be non-negative"))
        # Update the square root value
        r.sqrthalfλ = √(v/2)
        postupdate_λ!(r)
    elseif s ∈ (:sqrthalfλ, )
        setfield!(r, s, v)
    else
        error("cannot set any property of RidgePred except the shrinkage parameter λ (lambda): $(s)")
#        setfield!(r, s, v)
    end
end

function Base.propertynames(r::RidgePred, private=false)
    if private
        (:X, :λ, :lambda, :G, :βprior, :pred, :beta0, :delbeta, :pivot, :sqrthalfλ, :scratchbeta, :scratchy, :scratchwt)
    else
        (:X, :λ, :lambda, :G, :βprior, :pred, :beta0, :delbeta)
    end
end


function cgpred(X::StridedMatrix, λ::Real, G::AbstractMatrix, βprior::AbstractVector=zeros(eltype(X), size(X, 2)))
    T = eltype(X)
    RidgePred(DensePredCG, X, Base.convert(T, λ), Matrix{T}(G), Vector{T}(βprior))
end
function cholpred(X::StridedMatrix, λ::Real, G::AbstractMatrix, βprior::AbstractVector=zeros(eltype(X), size(X, 2)), pivot::Bool=false)
    T = eltype(X)
    RidgePred(DensePredChol, X, Base.convert(T, λ), Matrix{T}(G), Vector{T}(βprior), pivot)
end
function cgpred(X::SparseMatrixCSC, λ::Real, G::AbstractMatrix, βprior::AbstractVector=zeros(eltype(X), size(X, 2)))
    T = eltype(X)
    RidgePred(SparsePredCG, X, Base.convert(T, λ), SparseMatrixCSC{T}(G), Vector{T}(βprior))
end
function cholpred(X::SparseMatrixCSC, λ::Real, G::AbstractMatrix, βprior::AbstractVector=zeros(eltype(X), size(X, 2)))
    T = eltype(X)
    RidgePred(SparsePredChol, X, Base.convert(T, λ), SparseMatrixCSC{T}(G), Vector{T}(βprior))
end


cgpred(X::SparseMatrixCSC, λ, G::AbstractVector, βprior::AbstractVector) = cgpred(X, λ, spdiagm(0=>G), βprior)
cholpred(X::SparseMatrixCSC, λ, G::AbstractVector, βprior::AbstractVector) = cholpred(X, λ, spdiagm(0=>G), βprior)
cgpred(X::StridedMatrix, λ, G::AbstractVector, βprior::AbstractVector) = cgpred(X, λ, diagm(0=>G), βprior)
cholpred(X::StridedMatrix, λ, G::AbstractVector, βprior::AbstractVector, args...) = cholpred(X, λ, diagm(0=>G), βprior, args...)

function resetβ0!(p::RidgePred{T}) where {T<:BlasReal}
    beta0 = p.beta0
    delbeta = p.delbeta
    @inbounds for i = eachindex(beta0, delbeta)
        beta0[i] = 0
        delbeta[i] = 0
    end
    p
end

function delbeta!(p::RidgePred{T}, r::AbstractVector{T}, wt::AbstractVector{T}) where {T<:BlasReal}
    n, m = size(p.X)
    # Fill response
    copyto!(p.scratchy, r)
#    yprior = sqrt(p.λ/2) * p.G * (p.βprior - p.pred.beta0)
    broadcast!(-, p.scratchbeta, p.βprior, p.pred.beta0)
    @views mul!(p.scratchy[n+1:end], p.G, p.scratchbeta, p.sqrthalfλ, 0)
    
    # Fill weights
    copyto!(p.scratchwt, wt)
     
    # Compute Δβₙ from (XᵀWₙX + λGᵀG)⁻¹ (XᵀWₙrₙ + λGᵀG(βprior - βₙ))
    delbeta!(p.pred, p.scratchy, p.scratchwt)

    p
end

extendedmodelmatrix(p::RidgePred) = p.pred.X
function extendedweights(p::RidgePred{T}, wt::AbstractVector) where {T<:BlasReal}
    n,m = size(p.X)
    k = length(wt)
    if k == 0
        return ones(T, n+m)
    elseif k == n
        return vcat(Vector{T}(wt), ones(T, m))
    elseif k == n+m
        return wt
    else
        throw(DimensionMismatch("weights for RidgePred should be of size 0, n or n+m"))
    end
end
modelmatrix(p::RidgePred) = p.X
vcov(p::RidgePred, wt::AbstractVector) = inv(Hermitian(float(Matrix(extendedmodelmatrix(p)' * (extendedweights(p, wt) .* extendedmodelmatrix(p))))))
projectionmatrix(p::RidgePred, wt::AbstractVector) = (wwt = extendedweights(p, wt); Hermitian(extendedmodelmatrix(p) * vcov(p, wwt) * extendedmodelmatrix(p)' .* wwt))

dof(m::RobustLinearModel{T, R, L}) where {T, R, L<:AbstractRegularizedPred} = tr(projectionmatrix(m.pred, workingweights(m.resp)))
