

#################
loss_criteria(p::LinPred) = 0

StatsAPI.modelmatrix(p::LinPred) = p.X

function StatsAPI.vcov(p::LinPred, wt::AbstractVector)
    wXt = isempty(wt) ? modelmatrix(p)' : (modelmatrix(p) .* wt)'
    return inv(Hermitian(float(Matrix(wXt * modelmatrix(p)))))
end

function projectionmatrix(p::LinPred, wt::AbstractVector)
    wXt = isempty(wt) ? modelmatrix(p)' : (modelmatrix(p) .* wt)'
    return Hermitian(modelmatrix(p) * vcov(p, wt) * wXt)
end

StatsAPI.leverage(p::LinPred, wt::AbstractVector) = diag(projectionmatrix(p, wt))

leverage_weights(p::LinPred, wt::AbstractVector) = sqrt.(1 .- leverage(p, wt))

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
    DensePredQR

A `LinPred` type with a dense, unpivoted QR decomposition of `X`

# Members

- `X`: Model matrix of size `n` × `p` with `n ≥ p`.  Should be full column rank.
- `beta0`: base coefficient vector of length `p`
- `delbeta`: increment to coefficient vector, also of length `p`
- `scratchbeta`: scratch vector of length `p`, used in `linpred!` method
- `qr`: a `QRCompactWY` object created from `X`, with optional row weights.
"""
DensePredQR

PRED_QR_WARNING_ISSUED = false

function qrpred(X::AbstractMatrix, pivot::Bool=false)
    p = try
        DensePredCG(Matrix(X), pivot)
    catch e
        if e isa MethodError
            # GLM.DensePredCG(X::AbstractMatrix, pivot::Bool) is not defined
            global PRED_QR_WARNING_ISSUED
            if !PRED_QR_WARNING_ISSUED
                @warn(
                    "GLM.DensePredCG(X::AbstractMatrix, pivot::Bool) is not defined, " *
                    "fallback to unpivoted QR. GLM version should be >= 1.9."
                )
                PRED_QR_WARNING_ISSUED = true
            end
            DensePredCG(Matrix(X))
        else
            rethrow()
        end
    end
end


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
    Σ::Matrix{T}                  # Gram matrix and temporary matrix.
    scratchbeta::Vector{T}
    scratchm1::Matrix{T}
    scratchr1::Vector{T}
    function DensePredCG{T}(X::Matrix{T}, beta0::Vector{T}) where {T}
        n, p = size(X)
        length(beta0) == p || throw(DimensionMismatch("length(β0) ≠ size(X,2)"))
        new{T}(X, beta0, zeros(T, p), zeros(T, (p, p)), zeros(T, p), zeros(T, (n, p)), zeros(T, n))
    end
    function DensePredCG{T}(X::Matrix{T}) where {T}
        n, p = size(X)
        new{T}(X, zeros(T, p), zeros(T, p), zeros(T, (p, p)), zeros(T, p), zeros(T, (n, p)), zeros(T, n))
    end
end
DensePredCG(X::Matrix, beta0::Vector) = DensePredCG{eltype(X)}(X, beta0)
DensePredCG(X::Matrix{T}) where {T} = DensePredCG{T}(X, zeros(T, size(X, 2)))
Base.convert(::Type{DensePredCG{T}}, X::Matrix{T}) where {T} =
    DensePredCG{T}(X, zeros(T, size(X, 2)))

# Compatibility with cholpred(X, pivot)
cgpred(X, pivot::Bool) = cgpred(X)
cgpred(X::StridedMatrix) = DensePredCG(X)

function delbeta!(
    p::DensePredCG{T},
    r::AbstractVector{T},
    wt::AbstractVector{T},
) where {T<:BlasReal}
    scr = transpose(broadcast!(*, p.scratchm1, wt, p.X))
    mul!(p.Σ, scr, p.X)
    mul!(p.scratchbeta, transpose(p.scratchm1), r)
    # Solve the linear system
    cg!(p.delbeta, Hermitian(p.Σ, :U), p.scratchbeta)
    p
end

function delbeta!(p::DensePredCG{T}, r::AbstractVector{T}) where {T<:BlasReal}
    cg!(p.delbeta, Hermitian(mul!(p.Σ, p.X', p.X), :U), mul!(p.scratchbeta, p.X', r))
    return p
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
    X::M                    # model matrix
    beta0::Vector{T}        # base vector for coefficients
    delbeta::Vector{T}      # coefficient increment
    Σ::M                    # Gram matrix and temporary matrix.
    scratchbeta::Vector{T}
    scratchm1::M
    scratchr1::Vector{T}
end
function SparsePredCG(X::SparseMatrixCSC{T}) where {T}
    n, p = size(X)
    return SparsePredCG{eltype(X),typeof(X)}(
        X,
        zeros(T, p),
        zeros(T, p),
        zeros(T, (p, p)),
        zeros(T, p),
        similar(X),
        zeros(T, n),
    )
end

cgpred(X::SparseMatrixCSC) = SparsePredCG(X)

function delbeta!(
    p::SparsePredCG{T},
    r::AbstractVector{T},
    wt::AbstractVector{T},
) where {T<:BlasReal}
    scr = transpose(broadcast!(*, p.scratchm1, wt, p.X))
    mul!(p.Σ, scr, p.X)
    mul!(p.scratchbeta, transpose(p.scratchm1), r)
    # Solve the linear system
    cg!(p.delbeta, Hermitian(p.Σ, :U), p.scratchbeta)
    p
end

function delbeta!(p::SparsePredCG{T}, r::AbstractVector{T}) where {T<:BlasReal}
    cg!(p.delbeta, Hermitian(mul!(p.Σ, p.X', p.X), :U), mul!(p.scratchbeta, p.X', r))
    return p
end
