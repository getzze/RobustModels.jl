
modelmatrix(p::AbstractRobustModel) = p.pred.X

vcov(p::AbstractRobustModel) = inv(Hermitian(float(modelmatrix(p)' * (p.resp.wrkwt .* modelmatrix(p)))))

projectionmatrix(p::AbstractRobustModel) = Hermitian(modelmatrix(p) * vcov(p) * modelmatrix(p)') .* p.resp.wrkwt

function cor(m::AbstractRobustModel)
    Σ = vcov(m)
    invstd = inv.(sqrt.(diag(Σ)))
    lmul!(Diagonal(invstd), rmul!(Σ, Diagonal(invstd)))
end

## TODO: specialize to make it faster
leverage(p::AbstractRobustModel) = diag(projectionmatrix(p))



"""
    DensePredCG

A `LinPred` type with Conjugate Gradient and a dense `X`

# Members

- `X`: Model matrix of size `n` × `p` with `n ≥ p`.  Should be full column rank.
- `beta0`: base coefficient vector of length `p`
- `delbeta`: increment to coefficient vector, also of length `p`
- `scratchbeta`: scratch vector of length `p`, used in [`linpred!`](@ref) method
- `qr`: a `QRCompactWY` object created from `X`, with optional row weights.
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
    sqwt = Diagonal(sqrt.(wt))
    lsqr!(p.delbeta, mul!(p.scratchm1, sqwt, p.X), mul!(p.scratchr1, sqwt, r); log=false)
#    lsqr!(p.delbeta, mul!(p.scratchm1, Diagonal(wt), p.X), mul!(p.scratchr1, Diagonal(wt), r); log=false)
    p
end


mutable struct SparsePredCG{T,M<:SparseMatrixCSC} <: LinPred
    X::M                           # model matrix
    Xt::M                          # X'
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    scratchm1::M
    scratchr1::Vector{T}
end
function SparsePredCG(X::SparseMatrixCSC{T}) where T
    return SparsePredCG{eltype(X),typeof(X)}(X,
        X',
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        zeros(T, size(X, 2)),
        similar(X),
        zeros(T, size(X, 1)),
    )
end

cgpred(X::SparseMatrixCSC) = SparsePredCG(X)

function delbeta!(p::SparsePredCG{T}, r::AbstractVector{T}, wt::AbstractVector{T}) where T
#    sqwt = Diagonal(sqrt.(wt))
#    lsqr!(p.delbeta, mul!(p.scratchm1, sqwt, p.X), mul!(p.scratchr1, sqwt, r); log=false, damp=1e-4)
#    p
    scr = mul!(p.scratchm1, Diagonal(wt), p.X)
    cg!(p.delbeta, Hermitian(mul!(transpose(scr), p.X), :U), mul!(p.scratchr1, transpose(scr), r))
    p
end

#cholesky(p::DensePredCG{T}) where {T<:FP} = Cholesky{T,typeof(p.X)}(copy(p.qr.R), 'U', 0)
#function cholesky(p::DensePredChol{T}) where T<:FP
#    c = p.chol
#    Cholesky(copy(cholfactors(c)), c.uplo, c.info)
#end
#cholesky!(p::DensePredQR{T}) where {T<:FP} = Cholesky{T,typeof(p.X)}(p.qr.R, 'U', 0)


#invchol(x::DensePred) = inv(cholesky!(x))
#function invchol(x::DensePredChol{T,<: CholeskyPivoted}) where T
#    ch = x.chol
#    rnk = rank(ch)
#    p = length(x.delbeta)
#    rnk == p && return inv(ch)
#    fac = ch.factors
#    res = fill(convert(T, NaN), size(fac))
#    for j in 1:rnk, i in 1:rnk
#        res[i, j] = fac[i, j]
#    end
#    copytri!(LAPACK.potri!(ch.uplo, view(res, 1:rnk, 1:rnk)), ch.uplo, true)
#    ipiv = invperm(ch.piv)
#    res[ipiv, ipiv]
#end
#invchol(x::SparsePredCG) = cholesky!(x) \ Matrix{Float64}(I, size(x.X, 2), size(x.X, 2))

