
using LinearAlgebra: BlasReal, QRCompactWY, Diagonal, Hermitian, LowerTriangular, UpperTriangular, transpose, cholesky, cholesky!, qr, mul!, inv, diag, diagm, ldiv!


#################
modelmatrix(m::RobustLinearModel) = m.pred.X

vcov(m::RobustLinearModel) = inv(Hermitian(float(Matrix(modelmatrix(m)' * (workingweights(m.resp) .* modelmatrix(m))))))

projectionmatrix(m::RobustLinearModel) = Hermitian(modelmatrix(m) * vcov(m) * modelmatrix(m)') .* workingweights(m.resp)

function leverage_weights(m::RobustLinearModel)
    r = m.resp

    w = if !isempty(r.wts); r.wts else ones(eltype(r.y), size(r.y)) end
    v = inv(Hermitian(float(modelmatrix(m)' * (w .* modelmatrix(m)))))
    h = diag(Hermitian(modelmatrix(m) * v * modelmatrix(m)') .* w)
    sqrt.(1 .- h)
end


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
    ## Use views, do not create new objects
#    sqwt = copy!(p.scratchr1, sqrt.(wt))
#    src = mul!(p.scratchm1, Diagonal(sqwt), p.X)
#    lsqr!(p.delbeta, src, mul!(p.scratchr1, Diagonal(sqwt), r); log=false)
    sqwt = copyto!(p.scratchr1, sqrt.(wt))
    scr = broadcast!(*, p.scratchm1, sqwt, p.X)
    lsqr!(p.delbeta, scr, broadcast!(*, p.scratchr1, sqwt, r); log=false)
    p
end


mutable struct SparsePredCG{T,M<:SparseMatrixCSC} <: LinPred
    X::M                           # model matrix
    Xt::M                          # X'
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
        X',
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


### The structure is different from the one used in GLM
#"""
#    DensePredQR
#A `LinPred` type with a dense, unpivoted QR decomposition of `X`
## Members
#- `X`: Model matrix of size `n` × `p` with `n ≥ p`.  Should be full column rank.
#- `beta0`: base coefficient vector of length `p`
#- `delbeta`: increment to coefficient vector, also of length `p`
#- `scratchbeta`: scratch vector of length `p`, used in [`linpred!`](@ref) method
#- `qr`: a `QRCompactWY` object created from `X`, with optional row weights.
#"""
#mutable struct DensePredQR{T<:BlasReal} <: DensePred
#    X::Matrix{T}                  # model matrix
#    s0::Vector{T}              # base coefficient vector
#    dels::Vector{T}            # coefficient increment
#    scratchbeta::Vector{T}
#    scratchr1::Vector{T}
#    scratchv1::Vector{Bool}
#    qr::QRCompactWY{T}
#    function DensePredQR{T}(X::Matrix{T}, beta0::Vector{T}) where T
#        n, p = size(X)
#        length(beta0) == p || throw(DimensionMismatch("length(β0) ≠ size(X,2)"))
#        F = qr(X)
#        new{T}(X, (Matrix(F.Q) * beta0)[1:p], zeros(T,p), zeros(T,p), zeros(T,n), ones(Bool,n), F)
#    end
#    function DensePredQR{T}(X::Matrix{T}) where T
#        n, p = size(X)
#        new{T}(X, zeros(T,p), zeros(T,p), zeros(T,p), zeros(T,n), ones(Bool,n), qr(X))
#    end
#end
#DensePredQR(X::Matrix, beta0::Vector) = DensePredQR{eltype(X)}(X, beta0)
#DensePredQR(X::Matrix{T}) where T = DensePredQR{T}(X, zeros(T, size(X,2)))
#convert(::Type{DensePredQR{T}}, X::Matrix{T}) where {T} = DensePredQR{T}(X, zeros(T, size(X, 2)))


#function Base.getproperty(p::DensePredQR, s::Symbol)
#    if s == :beta0
#        p.qr.R \ p.s0
#    else # fallback to getfield
#        return getfield(p, s)
#    end
#end

#function Base.setproperty!(p::DensePredQR, s::Symbol, v)
#    if s == :beta0
#        p.s0 .= p.qr.R * v
#    else # fallback to setfield!
#        return setfield!(p, s, v)
#    end
#end


#"""
#    delbeta!(p::LinPred, r::Vector)
#Evaluate and return `p.delbeta` the increment to the coefficient vector from residual `r`
#"""
#function delbeta!(p::DensePredQR{T}, r::AbstractVector{T}) where T<:BlasReal
#    p = length(p.dels)
#    p.dels .= (p.qr.Q' * r)[1:p]
#    return p
#end
    
#function delbeta!(p::DensePredQR{T}, r::AbstractVector{T}, wt::AbstractVector{T}) where T<:BlasReal
#    valid = copy!(p.scratchv1, wt .>= eps())
#    ## TODO: make it inplace, but the matrix size is undefined...
#    wQT = (wt[valid] .* p.qr.Q[valid, :])'
    
#    C  = cholesky(Hermitian(wQT * p.qr.Q[valid, :]))
#    ldiv!(C, mul!(p.scratchr1, wQT, r[valid]))
#    copy!(p.dels, p.scratchr1[1:size(p.qr.R, 1)])
#    return p
#end

#function linpred!(out, p::DensePredQR, f::Real=1.)
#    mul!(out, p.qr.Q, iszero(f) ? p.s0 : broadcast!(muladd, p.scratchbeta, f, p.dels, p.s0))
#end

#function installbeta!(p::DensePredQR, f::Real=1.)
#    s0 = p.s0
#    dels = p.dels
#    @inbounds for i = eachindex(s0,dels)
#        s0[i] += dels[i]*f
#        dels[i] = 0
#    end
#    p.beta0
#end


#qrpred(X::AbstractMatrix) = DensePredQR(Matrix(X))
