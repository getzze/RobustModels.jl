using SparseArrays: sparse
using LinearAlgebra: cholesky


####################################
### AbstractRegularizedPred methods
####################################

StatsAPI.modelmatrix(p::AbstractRegularizedPred) = p.X

StatsAPI.coef(p::AbstractRegularizedPred) = p.beta0

penalty(p::AbstractRegularizedPred) = p.penalty

penalized_coef(p::AbstractRegularizedPred) = coef(p)

dev_criteria(p::AbstractRegularizedPred) = 2 * cost(penalty(p), penalized_coef(p))

initpred!(p::AbstractRegularizedPred, args...; kwargs...) = p

updatepred!(p::AbstractRegularizedPred, args...; kwargs...) = p

update_beta!(p::AbstractRegularizedPred, args...; kwargs...) = p

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
- `scratchbeta`: scratch vector of length `p`, used in [`GLM.linpred!`](@ref) method
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

penalty(p::RidgePred) = SquaredL2Penalty(p.λ)

penalized_coef(p::RidgePred) = p.G * (p.pred.beta0 - p.βprior)

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


##############################
###  Coordinate Gradient Descent predictor
##############################

mutable struct CGDParams
    "`outerloop`: use extra outer loop"
    outerloop::Bool
    "`update_all`: update all coefficients, that are zero or not."
    update_all::Bool

    function CGDParams(outerloop::Bool=true, update_all::Bool=true)
        return new(outerloop, update_all)
    end
end

function initparams!(p::CGDParams)
    p.update_all = true
    return p
end

"""
    CGDRegPred

Regularization using Coordinate Gradient Descent.

# Members

- `X`: model matrix
- `beta0`: base vector for coefficients
- `delbeta`: coefficients increment
- `Σ`: Gram matrix and temporary matrix
- `penalty`: the penalty function.
- `invvar`: precision vector.
- `scratchbeta`: scratch vector of length `p`, used in [`linpred!`](@ref) method.
"""
struct CGDRegPred{T<:BlasReal,M<:AbstractMatrix{T},V<:Vector{T},P<:PenaltyFunction} <:
       AbstractRegularizedPred{T}
    "`X`: model matrix"
    X::M
    "`beta0`: base vector for coefficients"
    beta0::V
    "`delbeta`: coefficients increment"
    delbeta::V
    "`Σ`: Gram matrix, X'WX."
    Σ::M
    "`penalty`: penalty function"
    penalty::P
    "`invvar`: inverse variance/precision vector"
    invvar::V
    "`params`: parameters for CGD"
    params::CGDParams
    "`scratchbeta`: scratch vector of length `p`, used in [`linpred!`](@ref) method"
    scratchbeta::Vector{T}

    function CGDRegPred(
        X::M,
        Σ::AbstractMatrix{<:Real},
        invvar::AbstractVector{<:Real},
        penalty::P,
        outerloop::Bool=true,
    ) where {M<:AbstractMatrix{T},P<:PenaltyFunction} where {T<:BlasReal}
        m = size(Σ, 1)

        if !isseparable(penalty)
            error(
                "Coordinate Gradient Descent is only allowed with a separable penalty: " *
                "$(penalty)",
            )
        end

        params = CGDParams(outerloop)

        return new{T,M,typeof(invvar),P}(
            X, zeros(T, m), zeros(T, m), Σ, penalty, invvar, params, zeros(T, m)
        )
    end
end

function CGDRegPred(
    X::M, penalty::P, wts::V, outerloop::Bool=true
) where {M<:AbstractMatrix{T},V<:AbstractVector,P<:PenaltyFunction} where {T<:BlasReal}
    n, m = size(X)
    ll = size(wts, 1)
    ll in (0, n) || throw(DimensionMismatch("length of wts is $ll, must be 0 or $n."))

    # invvar = precision_vector(X, wts)
    if isempty(wts)
        Σ = Hermitian(float(X' * X))
    else
        Σ = Hermitian(float((wts .* X)' * X))
    end
    invvar = Vector(inv.(diag(Σ)))

    return CGDRegPred(X, Σ, invvar, penalty, outerloop)
end

function precision_vector(
    X::AbstractMatrix{T}, wts::AbstractVector
) where {T<:AbstractFloat}
    invvar = Vector{T}(undef, size(X, 2))
    if isempty(wts)
        @inbounds @simd for j in eachindex(invvar, axes(X, 2))
            invvar[j] = 1 / sum(abs2, view(X, :, j))
        end
    else
        @inbounds @simd for j in eachindex(invvar, axes(X, 2))
            invvar[j] = 1 / sum(i -> wts[i] * abs2(X[i, j]), eachindex(wts, axes(X, 1)))
        end
    end
    return invvar
end

function initpred!(p::CGDRegPred, wts::AbstractVector=[], σ::Real=1; verbose::Bool=false)
    initparams!(p.params)
    return p
end

function update_βμ!(
    p::CGDRegPred{T},
    y::AbstractVector{T},
    μ::AbstractVector{T},
    res::AbstractVector{T},
    σ2::T=one(T),
    wts::AbstractVector{T}=T[];
    tol::Real=1e-4,
    verbose::Bool=false,
) where {T<:BlasReal}
    X = p.X
    β = p.beta0
    newβ = p.delbeta
    invvar = p.invvar
    pen = penalty(p)

    m = size(X, 2)
    outerM = p.params.outerloop ? m : 1
    update_all = p.params.update_all

    #    copyto!(newβ, β)
    # outer loop
    @inbounds for _ in 1:outerM

        Δdev = zero(T)
        # iterate over indices
        @inbounds for j in eachindex(axes(X, 2), β, newβ, invvar)
            βj = β[j]
            !update_all && iszero(βj) && continue

            Xj = view(X, :, j)

            # remove component due to index j in μ
            # µ -= X[:, j] * β[j]
            #            if !iszero(βj)
            #                @inbounds @simd for i in eachindex(μ, Xj)
            #                    μ[i] -= Xj[i] * β[j]
            #                end
            ##                broadcast!(muladd, μ, -β[j], Xj, μ)
            #            end

            newβ[j] = βj
            if isempty(wts)
                @inbounds @simd for i in eachindex(axes(X, 1), res, Xj)
                    newβ[j] += Xj[i] * res[i]
                    #                @inbounds @simd for i in eachindex(axes(X, 1), y, μ, Xj)
                    #                    newβ[j] += Xj[i] * (y[i] - μ[i] + Xj[i] * βj)
                end
            else
                @inbounds @simd for i in eachindex(axes(X, 1), res, Xj, wts)
                    newβ[j] += wts[i] * Xj[i] * res[i]
                end
            end
            newβ[j] *= invvar[j]


            #            verbose && println("∇βj before prox: $(gradβ)")
            # Proximal operator on a single coordinate
            proximal!(pen, newβ, j, newβ, invvar[j] * σ2)
            #            proximal!(pen, β, j, newβ, invvar[j] * σ2)
            #            verbose && println("β after prox: $(p.β)")

            # re-add component due to index j in μ
            #            μj = p.X[:, j] * p.β[j]
            #            p.µ .+= μj
            ##            if !iszero(β[j])
            ##                @inbounds @simd for i in eachindex(μ, Xj)
            ##                    μ[i] += Xj[i] * β[j]
            ##                end
            ###                broadcast!(muladd, μ, β[j], view(X, :, j), μ)
            ##            end
            Δβ = newβ[j] - βj
            if !iszero(Δβ)
                β[j] = newβ[j]
                @inbounds @simd for i in eachindex(μ, res, Xj)
                    dμi = Xj[i] * Δβ
                    μ[i] += dμi
                    res[i] -= dμi
                end
                Δdev = max(Δdev, abs2(Δβ) / invvar[j])
            end
        end

        if Δdev < tol
            if update_all
                # Converged
                break
            else
                # Run a last iteration, computing all the coefficients again
                update_all = true
            end
        else
            # Check for convergence
            update_all = false
        end
    end

    p.params.update_all = true
    return p
end


##############################
###  FISTA predictor
##############################

mutable struct FISTAParams{T<:AbstractFloat}
    "`restart`: allow restart"
    restart::Bool
    "`use_backtracking`: use backtracking for adjusting the step size"
    use_backtracking::Bool
    "`bt_maxiter`: backtracking loop, maximum number of iteration"
    bt_maxiter::Int
    "`bt_stepfac`: backtracking loop, reducing factor"
    bt_delta::T
    "`sk`: FISTA step parameter"
    sk::T
    "`tk`: FISTA acceleration step"
    tk::T
    "`kk`: ADMM number of iterations since last restart"
    kk::Int

    function FISTAParams{T}(
        restart::Bool, use_backtracking::Bool, bt_maxiter::Integer=20, bt_delta::Real=0.5
    ) where {T<:AbstractFloat}
        return new{T}(restart, use_backtracking, bt_maxiter, bt_delta, one(T), one(T), 1)
    end
end

function initparams!(p::FISTAParams, sk::Real=1)
    p.sk = sk
    p.tk = 1
    p.kk = 1
    return p
end

"""
    update_accelstep!(p::FISTAParams, restart::Bool=false; verbose::Bool=false)

Update the step for acceleration and return the acceleration factor.
If `restart`, reset to 1.
"""
function update_accelstep!(p::FISTAParams, restart::Bool=false; verbose::Bool=false)
    if restart
        verbose && println("Restart after $(p.kk) FISTA iterations")
        # reset step and iteration number
        p.tk = 1
        p.kk = 1
        return zero(p.tk)
    end

    tk = p.tk
    # tkp1 = (1 + √(1 + 4 * p.tk^2)) / 2
    # mk = (p.tk - 1) / tkp1

    # mk = (k - 1) / (k + 2)

    # Tseng (2008) - On accelerated proximal gradient methods for convex-concave optimization
    tkp1 = (√(tk^4 + 4 * tk^2) - tk^2) / 2
    mk = tkp1 * (1 - tk) / tk
    # Update the acceleration step
    p.tk = tkp1
    # Update the iteration number
    p.kk += 1
    return mk
end


"""
    FISTARegPred

Regularization using Fast Iterative Shrinkage-Thresholding Algorithm.

# Members

- `X`: model matrix
- `beta0`: base vector for coefficients
- `delbeta`: coefficients increment
- `Σ`: Gram matrix and temporary matrix
- `penalty`: the penalty function.
- `scratchbeta`: scratch vector of length `p`, used in [`linpred!`](@ref) method.
"""
struct FISTARegPred{T<:BlasReal,M<:AbstractMatrix{T},V<:Vector{T},P<:PenaltyFunction} <:
       AbstractRegularizedPred{T}
    "`X`: model matrix"
    X::M
    "`beta0`: base vector for coefficients"
    beta0::V
    "`delbeta`: coefficients increment"
    delbeta::V
    "`Σ`: Corresponds to X'WX."
    Σ::M
    "`penalty`: penalty function"
    penalty::P
    "`scratchbeta`: scratch vector of length `p`, used in [`linpred!`](@ref) method"
    scratchbeta::V
    "`wXt`: transpose of the (weighted) model matrix"
    wXt::M
    "`params`: parameters for FISTA"
    params::FISTAParams
    "pre-allocated temporary vectors"
    gradβ::V
    proxβ::V
    stepβ::V

    function FISTARegPred(
        X::M,
        wXt::AbstractMatrix{<:Real},
        Σ::AbstractMatrix{<:Real},
        penalty::P,
        restart::Bool=true,
        use_backtracking::Bool=false,
    ) where {M<:AbstractMatrix{T},P<:PenaltyFunction} where {T<:BlasReal}
        m = size(Σ, 1)

        params = FISTAParams{T}(restart, use_backtracking)

        beta0 = zeros(T, m)
        return new{T,M,typeof(beta0),P}(
            X,
            beta0,
            zeros(T, m),
            Σ,
            penalty,
            zeros(T, m),
            wXt,
            params,
            zeros(T, m),
            zeros(T, m),
            zeros(T, m),
        )
    end
end

function FISTARegPred(
    X::M, penalty::P, wts::V, restart::Bool=true, use_backtracking::Bool=false
) where {M<:AbstractMatrix{T},V<:AbstractVector,P<:PenaltyFunction} where {T<:BlasReal}
    n, m = size(X)
    ll = size(wts, 1)
    ll in (0, n) || throw(DimensionMismatch("length of wts is $ll, must be 0 or $n."))

    wXt = isempty(wts) ? X' : (X .* wts)'
    Σ = Hermitian(float(wXt * X))

    return FISTARegPred(X, wXt, Σ, penalty, restart, use_backtracking)
end

function initpred!(p::FISTARegPred, wts::AbstractVector=[], σ::Real=1; verbose::Bool=false)
    copyto!(p.proxβ, p.beta0)
    copyto!(p.stepβ, p.beta0)

    # for Normal errors
    #        sk = 1 / eigmax(Σ)
    # TODO: `eigen` methods not defined for sparse arrays, use Arpack.jl?
    #        sk = 1 / eigmax(Matrix(Σ))
    sk = 1 / eigmax(Matrix(p.Σ))
    initparams!(p.params, sk)
    return p
end

function update_beta!(
    p::FISTARegPred{T},
    y::AbstractVector{T},
    wts::AbstractVector{T}=T[],
    σ2::T=one(T);
    verbose::Bool=false,
) where {T<:BlasReal}
    β = p.beta0
    dβ = p.delbeta
    Σ = p.Σ
    wXt = p.wXt
    gradβ = p.gradβ
    stepβ = p.stepβ
    proxβ = p.proxβ
    scratchbeta = p.scratchbeta

    sk = p.params.sk
    restart = p.params.restart


    # ∇_β̂(f) = (-1/σ^2) * ((W .* X)' * wrky - Σ * β̂)
    # ∇_β̂(f) = Σ * β̂ - ((W .* X)' * wrky)
    # For Normal errors
    mul!(gradβ, wXt, y)
    mul!(scratchbeta, Σ, stepβ)  # dumb variable scratchbeta
    # rdiv!(broadcast!(-, gradβ, gradβ, scratchbeta), -σ2)
    broadcast!(-, gradβ, scratchbeta, gradβ)

    # broadcast!(-, yres, broadcast!(-, yres, p.y, mul!(yres, p.X, stepβ)), p.outliers)
    # rdiv!(mul!(gradβ, wXt, yres), -p.σ)
    @. dβ = stepβ - sk * gradβ
    proximal!(penalty(p), proxβ, dβ, sk * σ2)
    verbose && println("current coefs:\n\t$(β)")
    verbose && println("gradient:\n\t$(sk * gradβ)")
    verbose && println("new coefs:\n\t$(proxβ)")

    # find step-size
    if p.params.use_backtracking
        Im = I(size(β, 1))
        bt_delta = p.params.bt_delta
        bt_maxiter = p.params.bt_maxiter
        # backtracking
        # x = βstep
        # f(β) = 1/2 ||y - Xβ - γ||²
        # Δ = F(pr) - Q(pr, x, Li)
        # Δ = f(pr) - f(x) - dot(pr - x, df_x) - Lk/2 * sum(abs2, pr - x)
        broadcast!(-, scratchbeta, proxβ, stepβ)
        # for Normal errors
        # Needs julia > v1.4
        Δ = dot(scratchbeta, sk * Σ - Im, scratchbeta)
        verbose && println("backtracking: decrease gradient step if 0 < $(Δ)")
        jj = 0
        while Δ > 0 && jj < bt_maxiter
            # Update the gradient step
            p.params.sk = sk = bt_delta * sk
            @. dβ = stepβ - sk * gradβ
            proximal!(penalty(m), proxβ, dβ, si)
            broadcast!(-, scratchbeta, proxβ, stepβ)
            # for Normal errors
            Δ = dot(scratchbeta, sk * Σ - Im, scratchbeta)
            verbose && println("backtracking step $jj: decrease FISTA step $sk")
            jj += 1
        end
        Δ > 0 && throw(ConvergenceException(bt_maxiter))
    end


    # FISTA step
    c = dot(gradβ, proxβ) - dot(gradβ, β)
    if !restart || c <= 0
        # Accelerated step
        mk = update_accelstep!(p.params, false; verbose=verbose)
        @. stepβ = proxβ + mk * (proxβ - β)
    else
        # Restart
        update_accelstep!(p.params, true; verbose=verbose)
        @. stepβ = β
    end

    # update variables
    copyto!(β, proxβ)

    return p
end


##############################
###  AMA predictor
##############################

mutable struct AMAParams{T<:AbstractFloat}
    "`restart`: allow restart"
    restart::Bool
    "`η`: η for restart criteria"
    η::T
    "`ρ`: AMA step parameter"
    ρ::T
    "`tk`: AMA acceleration step"
    tk::T
    "`ck`: AMA restart criterium"
    ck::T
    "`kk`: AMA number of iterations since last restart"
    kk::Int

    function AMAParams{T}(restart::Bool, η::Real=1.0) where {T<:AbstractFloat}
        return new{T}(restart, η, one(T), one(T), zero(T), 1)
    end
end

function initparams!(p::AMAParams, ρ::Real=1)
    p.ρ = ρ
    p.tk = 1
    p.ck = 0
    p.kk = 1
    return p
end

"""
    update_accelstep!(p::AMAParams, restart::Bool=false; verbose::Bool=false)

Update the step for acceleration and return the acceleration factor.
If `restart`, reset to 1.
"""
function update_accelstep!(p::AMAParams, restart::Bool=false; verbose::Bool=false)
    if restart
        verbose && println("Restart after $(p.kk) AMA iterations")
        # reset step and iteration number
        p.tk = 1
        p.kk = 1
        return zero(p.tk)
    end

    tk = p.tk
    tkp1 = (1 + √(1 + 4 * p.tk^2)) / 2
    mk = (p.tk - 1) / tkp1
    p.tk = tkp1
    # update iteration number
    p.kk += 1
    return mk
end


struct AMARegPred{
    T<:BlasReal,
    M<:AbstractMatrix{T},
    V<:Vector{T},
    #    C,
    P<:PenaltyFunction,
    M2<:AbstractMatrix{T},
    V2<:AbstractVector{T},
} <: AbstractRegularizedPred{T}
    "`X`: model matrix"
    X::M
    "`beta0`: base vector for coefficients"
    beta0::V
    "`delbeta`: coefficients increment"
    delbeta::V
    "`Σ`: Corresponds to X'WX."
    Σ::M
    "`penalty`: penalty function"
    penalty::P
    "`scratchbeta`: scratch vector of length `p`, used in [`linpred!`](@ref) method"
    scratchbeta::V
    "`penbeta0`: vector of length `p`, used in [`penalized_coef`](@ref) method"
    penbeta0::V
    #    "`chol`: cholesky factorization"
    #    chol::C
    "`wXt`: transpose of the (weighted) model matrix"
    wXt::M
    "`A`: matrix of the constraint equation `A . u  - b - v = 0`"
    A::M2
    "`b`: vector of the constraint equation `A . u  - b - v = 0`"
    b::V2
    "`params`: parameters for AMA"
    params::AMAParams
    "pre-allocated temporary vectors"
    vk::V
    vkp1::V
    wk::V
    wkp1::V
    whatk::V
    whatkp1::V

    function AMARegPred(
        X::M,
        wXt::AbstractMatrix{<:Real},
        Σ::AbstractMatrix{<:Real},
        penalty::P,
        A::AbstractMatrix{<:Real}=float(I(size(Σ, 1))),
        b::AbstractVector{<:Real}=zeros(T, size(Σ, 1)),
        restart::Bool=true,
    ) where {M<:AbstractMatrix{T},P<:PenaltyFunction} where {T<:BlasReal}
        m = size(Σ, 1)

        #        chol = cholesky(Σ)
        bhat = A * b
        params = AMAParams{T}(restart)

        beta0 = zeros(T, m)
        #        return new{T,M,typeof(beta0),typeof(chol),P,typeof(A),typeof(b)}(
        return new{T,M,typeof(beta0),P,typeof(A),typeof(b)}(
            X,
            beta0,
            zeros(T, m),
            Σ,
            penalty,
            zeros(T, m),
            zeros(T, m),
            #            chol,
            wXt,
            A,
            bhat,
            params,
            zeros(T, m),
            zeros(T, m),
            zeros(T, m),
            zeros(T, m),
            zeros(T, m),
            zeros(T, m),
        )
    end
end

function AMARegPred(
    X::M,
    penalty::P,
    wts::V,
    A::AbstractMatrix{<:Real}=zeros(T, 0, 0),
    b::AbstractVector{<:Real}=zeros(T, 0),
    restart::Bool=true,
) where {
    M<:AbstractMatrix{T},V<:AbstractVector{<:Real},P<:PenaltyFunction
} where {T<:BlasReal}
    n, m = size(X)
    ll = size(wts, 1)
    ll in (0, n) || throw(DimensionMismatch("length of wts is $ll, must be 0 or $n."))

    wXt = isempty(wts) ? X' : (X .* wts)'
    Σ = Hermitian(float(wXt * X))

    if isempty(A)
        A = float(I(m))
    end
    size(A) == (m, m) ||
        throw(DimensionMismatch("size of A is $(size(A)), must be (0, 0) or $((m, m))."))
    if isempty(b)
        b = zeros(T, m)
    end
    size(b, 1) == m ||
        throw(DimensionMismatch("size of b is $(size(b, 1)), must be 0 or $(m)."))

    return AMARegPred(X, wXt, Σ, penalty, A, b, restart)
end

penalized_coef(p::AMARegPred) = mul!(copyto!(p.penbeta0, p.b), p.A, p.beta0, 1, -1)

function initpred!(p::AMARegPred, wts::AbstractVector=[], σ::Real=1; verbose::Bool=false)
    copyto!(p.delbeta, p.beta0)
    copyto!(p.scratchbeta, p.beta0)
    copyto!(p.vk, p.beta0)
    copyto!(p.vkp1, p.beta0)
    copyto!(p.wk, p.beta0)
    copyto!(p.wkp1, p.beta0)
    copyto!(p.whatk, p.beta0)
    copyto!(p.whatkp1, p.beta0)

    # Goldstein (2014) - Fast Alternating Direction Optimization Methods
    # Theorem 5: ρ <= σH / ρ(A'A)
    #        ρ = eigmin(Σ) / eigmax(A'A)
    # TODO: `eigen` methods not defined for sparse arrays, use Arpack.jl?
    #        ρ = eigmin(Matrix(Σ)) / eigmax(Matrix(A'A))
    ρ = eigmin(Matrix(p.Σ)) / eigmax(Matrix(p.A' * p.A))
    initparams!(p.params, abs(ρ))

    return p
end

function update_beta!(
    p::AMARegPred{T},
    y::AbstractVector{T},
    wts::AbstractVector{T}=T[],
    σ2::T=one(T);
    verbose::Bool=false,
) where {T<:BlasReal}
    β = p.beta0
    wXt = p.wXt
    βkp1 = p.delbeta
    scratch = p.scratchbeta
    #    chol = p.chol
    A = p.A
    b = p.b

    vk = p.vk
    vkp1 = p.vkp1
    wk = p.wk
    wkp1 = p.wkp1
    whatk = p.whatk
    whatkp1 = p.whatkp1

    ρ = p.params.ρ
    ck = p.params.ck
    restart = p.params.restart
    η = p.params.η

    verbose && println("AMA current coefs:\n\t$(β)")

    # Goldstein (2014) - Fast Alternating Minimization Algorithm
    # Lρ(u, v, w) = 1/2σ² |y - X u|² + P(v) + ρ w' * (A u - b - v) + ρ/2 |Au - b - v|²
    # uk+1 = (X'WX) \ (X'Wy - ρ σ² A' wk)     # argmin_u Lρ(u, vk, wk) without last term
    # vk+1 = prox_λ/ρ ( wk + A uk+1 - b )     # argmin_v Lρ(uk+1, v, wk)
    # wk+1 = wk + A uk+1 - b - vk+1
    #
    # primal residual: rk = A uk - b - vk
    # dual residual:   dk = ρ * (vk+1 - vk)
    #
    # βkp1 = chol \ (wXt * wrky - ρ * σ2 * A' * whatk)
    scratch = mul!(scratch, A', whatk)
    scratch = mul!(scratch, wXt, y, 1, -ρ * σ2)
    #    βkp1 = chol \ scratch
    cg!(βkp1, p.Σ, scratch)
    # vkp1 = proximal(penalty(p), A * βkp1 - b + whatk, 1/ρ)
    scratch = mul!(copyto!(scratch, b), A, βkp1, 1, -1)
    broadcast!(+, scratch, scratch, whatk)
    proximal!(penalty(p), vkp1, scratch, 1 / ρ)
    # wkp1 = whatk + A * βkp1 - b - vkp1
    broadcast!(-, wkp1, scratch, vkp1)

    ckp1 = sum(abs2, wkp1 - whatk)
    if !restart || ck == 0 || ckp1 < η * ck
        # Accelerated step
        mk = update_accelstep!(p.params, false; verbose=verbose)
        @. whatkp1 = wkp1 + mk * (wkp1 - wk)
    else
        # Restart
        update_accelstep!(p.params, true; verbose=verbose)
        @. whatkp1 = wk
        ckp1 = ck / η
    end

    # update variables
    p.params.ck = ckp1
    copyto!(β, βkp1)
    copyto!(vk, vkp1)
    copyto!(wk, wkp1)
    copyto!(whatk, whatkp1)

    return p
end



##############################
###  ADMM predictor
##############################

mutable struct ADMMParams{T<:AbstractFloat}
    "`restart`: allow restart"
    restart::Bool
    "`η`: η for restart criteria"
    η::T
    "`adapt`: allow adaptation of ρ"
    adapt::Bool
    "`adapt_τ`: parameter τ for ρ-adaptation"
    adapt_τ::T
    "`adapt_logμ`: (log of the) parameter μ for ρ-adaptation"
    adapt_logμ::T
    "`ρ`: ADMM step parameter"
    ρ::T
    "`tk`: ADMM acceleration step"
    tk::T
    "`ck`: ADMM restart criterium"
    ck::T
    "`kk`: ADMM number of iterations since last restart"
    kk::Int
    "`updatedρ`: flag for updated ρ"
    updatedρ::Bool

    function ADMMParams{T}(
        restart::Bool, adapt::Bool, η::Real=0.999, adapt_τ::Real=2.0, adapt_μ::Real=10.0
    ) where {T<:AbstractFloat}
        return new{T}(
            restart, η, adapt, adapt_τ, log(adapt_μ), one(T), one(T), zero(T), 1, false
        )
    end
end

function initparams!(p::ADMMParams, ρ::Real=1)
    p.ρ = ρ
    p.tk = 1
    p.ck = 0
    p.kk = 1
    p.updatedρ = false
    return p
end

"""
    update_accelstep!(p::ADMMParams, restart::Bool=false; verbose::Bool=false)

Update the step for acceleration and return the acceleration factor.
If `restart`, reset to 1.
"""
function update_accelstep!(p::ADMMParams, restart::Bool=false; verbose::Bool=false)
    if restart
        verbose && println("Restart after $(p.kk) ADMM iterations")
        # reset step and iteration number
        p.tk = 1
        p.kk = 1
        return zero(p.tk)
    end

    tk = p.tk
    tkp1 = (1 + √(1 + 4 * p.tk^2)) / 2
    mk = (p.tk - 1) / tkp1
    p.tk = tkp1
    # update iteration number
    p.kk += 1
    return mk
end

"""
    update_adapt!(p::ADMMParams, logr::Real; verbose::Bool=false)

If `adapt`, update the step ρ. `logr`= log(rrk / ssk), so the criteria become:
 - `rrk > ssk * adapt_μ` ==> `logr > adapt_logμ`
 - `rrk < ssk / adapt_μ` ==> `logr < -adapt_logμ`
Returns the step ρ.
"""
function update_adapt!(p::ADMMParams, logr::Real; verbose::Bool=false)
    if !p.adapt
        return p.ρ
    end
    ρ = p.ρ
    if logr > p.adapt_logμ
        ρ *= p.adapt_τ
    elseif logr < -p.adapt_logμ
        ρ /= p.adapt_τ
    end
    if p.ρ != ρ
        verbose && println("ADMM adapt ρ: $ρ")
        p.updatedρ = true
    end
    p.ρ = ρ
    return ρ
end


struct ADMMRegPred{
    T<:BlasReal,
    M<:AbstractMatrix{T},
    V<:Vector{T},
    #    C,
    P<:PenaltyFunction,
    M2<:AbstractMatrix{T},
    V2<:AbstractVector{T},
} <: AbstractRegularizedPred{T}
    "`X`: model matrix"
    X::M
    "`beta0`: base vector for coefficients"
    beta0::V
    "`delbeta`: coefficients increment"
    delbeta::V
    "`Σ`: Corresponds to X'WX."
    Σ::M
    "`penalty`: penalty function"
    penalty::P
    "`scratchbeta`: scratch vector of length `p`, used in [`linpred!`](@ref) method"
    scratchbeta::V
    "`penbeta0`: vector of length `p`, used in [`penalized_coef`](@ref) method"
    penbeta0::V
    #    "`chol`: cholesky factorization"
    #    chol::C
    "`wXt`: transpose of the (weighted) model matrix"
    wXt::M
    "`A`: matrix of the constraint equation `A . u  - b - v = 0`"
    A::M2
    "`b`: vector of the constraint equation `A . u  - b - v = 0`"
    b::V2
    "`params`: parameters for ADMM"
    params::ADMMParams
    "pre-allocated temporary vectors"
    vk::V
    vkp1::V
    vhatk::V
    vhatkp1::V
    wk::V
    wkp1::V
    whatk::V
    whatkp1::V
    wrkΣ::M

    function ADMMRegPred(
        X::M,
        wXt::AbstractMatrix{<:Real},
        Σ::AbstractMatrix{<:Real},
        penalty::P,
        A::AbstractMatrix{<:Real}=float(I(size(Σ, 1))),
        b::AbstractVector{<:Real}=zeros(T, size(Σ, 1)),
        restart::Bool=true,
        adapt::Bool=false,
    ) where {M<:AbstractMatrix{T},P<:PenaltyFunction} where {T<:BlasReal}
        m = size(Σ, 1)

        #        chol = cholesky(sparse(Σ))
        bhat = A * b
        params = ADMMParams{T}(restart, adapt)

        beta0 = zeros(T, m)
        #        return new{T,M,typeof(beta0),typeof(chol),P,typeof(A),typeof(b)}(
        return new{T,M,typeof(beta0),P,typeof(A),typeof(b)}(
            X,
            beta0,
            zeros(T, m),
            Σ,
            penalty,
            zeros(T, m),
            zeros(T, m),
            #            chol,
            wXt,
            A,
            bhat,
            params,
            zeros(T, m),
            zeros(T, m),
            zeros(T, m),
            zeros(T, m),
            zeros(T, m),
            zeros(T, m),
            zeros(T, m),
            zeros(T, m),
            similar(Σ),
        )
    end
end

function ADMMRegPred(
    X::M,
    penalty::P,
    wts::V,
    A::AbstractMatrix{<:Real}=zeros(T, 0, 0),
    b::AbstractVector{<:Real}=zeros(T, 0),
    restart::Bool=true,
    adapt::Bool=false,
) where {
    M<:AbstractMatrix{T},V<:AbstractVector{<:Real},P<:PenaltyFunction
} where {T<:BlasReal}
    n, m = size(X)
    ll = size(wts, 1)
    ll in (0, n) || throw(DimensionMismatch("length of wts is $ll, must be 0 or $n."))

    wXt = isempty(wts) ? X' : (X .* wts)'
    Σ = Hermitian(float(wXt * X))

    if isempty(A)
        A = float(I(m))
    end
    size(A) == (m, m) ||
        throw(DimensionMismatch("size of A is $(size(A)), must be (0, 0) or $((m, m))."))
    if isempty(b)
        b = zeros(T, m)
    end
    size(b, 1) == m ||
        throw(DimensionMismatch("size of b is $(size(b, 1)), must be 0 or $(m)."))

    return ADMMRegPred(X, wXt, Σ, penalty, A, b, restart, adapt)
end

penalized_coef(p::ADMMRegPred) = mul!(copyto!(p.penbeta0, p.b), p.A, p.beta0, 1, -1)

function initpred!(p::ADMMRegPred, wts::AbstractVector=[], σ::Real=1; verbose::Bool=false)
    ## Init temporary arrays
    copyto!(p.scratchbeta, p.beta0)
    copyto!(p.vk, p.beta0)
    copyto!(p.vkp1, p.beta0)
    copyto!(p.vhatk, p.beta0)
    copyto!(p.vhatkp1, p.beta0)
    copyto!(p.wk, p.beta0)
    copyto!(p.wkp1, p.beta0)
    copyto!(p.whatk, p.beta0)
    copyto!(p.whatkp1, p.beta0)

    ## Init parameters
    # Goldstein (2014) - Fast Alternating Direction Optimization Methods
    # Theorem 2: ρ^3 <= (σH σG^2) / (ρ(A'A) * ρ(B'B)^2)
    #        σK = eigmin(Σ) / eigmax(A'A)
    # TODO: `eigen` methods not defined for sparse arrays, use Arpack.jl?
    #        σK = eigmin(Matrix(Σ)) / eigmax(Matrix(A'A))
    #        ρ = σK^(1/3)
    σK = eigmin(Matrix(p.Σ)) / eigmax(Matrix(p.A' * p.A))
    initparams!(p.params, abs(σK)^(1 / 3))

    # Generate the extended Hessian matrix
    updatepred!(p, σ; verbose=verbose, force=true)

    return p
end


function update_beta!(
    p::ADMMRegPred{T},
    y::AbstractVector{T},
    wts::AbstractVector{T}=T[],
    σ2::T=one(T);
    verbose::Bool=false,
) where {T<:BlasReal}
    β = p.beta0
    wXt = p.wXt
    βkp1 = p.delbeta
    scratch = p.scratchbeta
    wrkΣ = p.wrkΣ
    A = p.A
    b = p.b

    vk = p.vk
    vkp1 = p.vkp1
    vhatk = p.vhatk
    vhatkp1 = p.vhatkp1
    wk = p.wk
    wkp1 = p.wkp1
    whatk = p.whatk
    whatkp1 = p.whatkp1

    ρ = p.params.ρ
    ck = p.params.ck
    restart = p.params.restart
    η = p.params.η

    verbose && println("ADMM current coefs:\n\t$(β)")

    # Goldstein (2014) - Fast Alternating Direction Optimization Methods
    # Lρ(u, v, w) = 1/2σ² |y - X u|² + P(v) + ρ/2 |A u - b - v + w|² - ρ/2 |w|²
    # uk+1 = (X'WX + ρ σ² A'A) \ (X'Wy + ρ σ² A' (b + vk - wk))
    # vk+1 = prox_λ/ρ ( A uk+1 - b + wk )
    # wk+1 = wk + A uk+1 - b - vk+1
    #
    # primal residual: rk = A uk - b - vk
    # dual residual:   dk = ρ * (vk+1 - vk)
    #
    # βkp1 = facΣρ \ (wXt * wrky + ρ * σ2 * A' * (b + vhatk - whatk))
    broadcast!(+, βkp1, broadcast!(-, βkp1, vhatk, whatk), b)  # dumb βkp1
    mul!(scratch, A', βkp1)  # dumb βkp1
    mul!(scratch, wXt, y, 1, ρ * σ2)
    #    βkp1 = facΣρ \ scratch
    cg!(βkp1, wrkΣ, scratch)
    # proximal!(penalty(p), vkp1, A * βkp1 - b + whatk, 1/ρ)
    scratch = mul!(copyto!(scratch, b), A, βkp1, 1, -1)
    broadcast!(+, scratch, scratch, whatk)
    proximal!(penalty(p), vkp1, scratch, 1 / ρ)
    # wkp1 = whatk + A * βkp1 - b - vkp1
    broadcast!(-, wkp1, scratch, vkp1)

    rrk = sum(abs2, wkp1 - whatk)
    ssk = ρ * sum(abs2, vkp1 - vhatk)
    ckp1 = rrk + ssk
    # verbose && println((; rrk, ssk))
    # verbose && println("criteria: $(ckp1) < $(η) * $(ck)")
    if !restart || ck == 0 || ckp1 < η * ck
        # Accelerated step
        mk = update_accelstep!(p.params, false; verbose=verbose)
        @. vhatkp1 = vkp1 + mk * (vkp1 - vk)
        @. whatkp1 = wkp1 + mk * (wkp1 - wk)
    else
        # Restart
        update_accelstep!(p.params, true; verbose=verbose)
        @. vhatkp1 = vk
        @. whatkp1 = wk
        ckp1 = ck / η
    end

    # update variables
    p.params.ck = ckp1
    copyto!(β, βkp1)
    copyto!(vk, vkp1)
    copyto!(vhatk, vhatkp1)
    copyto!(wk, wkp1)
    copyto!(whatk, whatkp1)

    # adapt ρ
    update_adapt!(p.params, log(rrk / ssk); verbose=verbose)

    return p
end

function updatepred!(p::ADMMRegPred, σ::Real; verbose::Bool=false, force::Bool=false)
    if (p.params.adapt && p.params.updatedρ) || force
        Σ = p.Σ
        A = p.A
        ρ = p.params.ρ

        # Refactorize
        # TODO: should be improved
        #        cholesky!(p.chol, sparse(Σ + ρ * σ^2 * Hermitian(A'A)))
        #        p.wrkΣ = Hermitian(p.Σ + σ2 * ρ * A'A)
        copyto!(p.wrkΣ, p.Σ)
        mul!(p.wrkΣ, A', A, ρ * σ^2, 1)
        copyto!(p.wrkΣ, Hermitian(p.wrkΣ))
        p.params.updatedρ = false
        verbose && println("Recompute cholesky factorization")
    end
    return p
end
