

using LinearAlgebra: dot
using JuMP: Model, @variable, @constraint, @objective, optimize!, value
import GLPK



"""
    QuantileRegression
Quantile regression representation
## Fields
* `τ`: the quantile value
* `X`: the model matrix
* `β`: the coefficients
* `y`: the response vector
* `wts`: the weights
* `wrkres`: the working residuals
* `fitdispersion`: if true, the dispersion is estimated otherwise it is kept fixed
* `fitted`: if true, the model was already fitted
"""
mutable struct QuantileRegression{T<:AbstractFloat, M<:AbstractMatrix{T}, V<:AbstractVector{T}} <: AbstractRobustModel{T}
    τ::T
    X::M
    β::V
    y::V
    wts::V
    wrkres::V
    fitdispersion::Bool
    fitted::Bool

    function QuantileRegression(τ::T, X::M, y::V, wts::V) where {V<:AbstractVector{T}, M<:AbstractMatrix{T}} where {T<:AbstractFloat}
        n = length(y)
        m, p = size(X)
        ll = length(wts)
        ll == 0 || ll == n || error("length of wts is $ll, must be $n or 0")
        m == n || error("X has $m rows, must be like y, $n")
        new{T, M, V}(τ, X, zeros(T, p), y, wts, similar(y), false, false)
    end
end


"""
    quantreg(X, y, args...; kwargs...)
An alias for `fit(QuantileRegression, X, y)`
The arguments `X` and `y` can be a `Matrix` and a `Vector` or a `Formula` and a `DataFrame`.
"""
quantreg(X, y, args...; kwargs...) = fit(QuantileRegression, X, y, args...; kwargs...)


function fit(::Type{M}, X::Union{AbstractMatrix{T},SparseMatrixCSC{T}},
             y::AbstractVector{T}; quantile::AbstractFloat=0.5,
             dofit::Bool          = true,
             weights::FPVector    = similar(y, 0),
             fitdispersion::Bool  = false,
             fitargs...) where {M<:QuantileRegression, T<:AbstractFloat}

    # Check that X and y have the same number of observations
    if size(X, 1) != size(y, 1)
        throw(DimensionMismatch("number of rows in X and y must match"))
    end

    # Check quantile is an allowed value
    (0 < quantile < 1) || error("quantile should be a number between 0 and 1 excluded: $(quantile)")

    m = QuantileRegression(quantile, X, y, weights)
    return if dofit; fit!(m; fitargs...) else m end
end

function fit(::Type{M}, X::Union{AbstractMatrix,SparseMatrixCSC},
             y::AbstractVector; kwargs...) where {M<:QuantileRegression}
    fit(M, float(X), float(y); kwargs...)
end



function fit!(m::QuantileRegression{T}, y::FPVector;
                quantile::Union{Nothing, AbstractFloat}=nothing,
                wts::Union{Nothing, FPVector}=nothing,
                kwargs...) where {T}

    # Update y and wts
    copy!(m.y, y)
    n = length(m.y)
    l = length(wts)
    if !isa(wts, Nothing) && (l==n || l==0)
        copy!(m.wts, wts)
    end
    if !isa(quantile, Nothing)
        (0 < quantile < 1) || error("quantile should be a number between 0 and 1 excluded: $(quantile)")
        m.τ = quantile
    end

    fit!(m; kwargs...)
end


"""
    fit!(m::QuantileRegression[; verbose::Bool=false, correct_leverage::Bool=false])
Optimize the objective of a `QuantileRegression`.  When `verbose` is `true` the values of the
objective and the parameters are printed on stdout at each function evaluation.
This function assumes that `m` was correctly initialized.
"""
function fit!(m::QuantileRegression{T}; verbose::Bool=false,
              correct_leverage::Bool=false, kwargs...) where {T}

    # Return early if model has the fit flag set
    m.fitted && return m

    if correct_leverage
        wts = m.wts
        copy!(wts, leverage_weights(m))
        ## TODO: maybe multiply by the old wts?
    end

    interiormethod!(m.β, m.wrkres, m.X, m.y, m.τ; wts=m.wts, verbose=verbose)

    m.fitted = true
    m
end


function interiormethod(X, y, τ; kwargs...)
    n, p = size(X)
    T = eltype(y)
    interiormethod!(zeros(T, p), zeros(T, n), X, y, τ; kwargs...)
end


function interiormethod!(βout, rout, X, y, τ; wts=[], verbose::Bool=false)
    model = Model(GLPK.Optimizer)

    n, p = size(X)

    @variable(model, β[1:p])
    @variable(model, u[1:n] >= 0)
    @variable(model, v[1:n] >= 0)

    e = ones(n)

    @objective(model, Min, τ*dot(e, u) + (1-τ)*dot(e, v) )

    Wy, WX = if isempty(wts)
        y, X
    else
        (wts .* y), (wts .* X)
    end

    @constraint(model, resid, Wy .== WX * β + u - v)

    optimize!(model)

    copyto!(βout, value.(β))
    copyto!(rout, value.(u) - value.(v))

    if verbose
        println("coef: ", βout)
        println("res: ", rout)
    end
    βout, rout
end


_objective(r::AbstractFloat, τ::AbstractFloat) = (τ - (r < 0)) * r

function dispersion(m::QuantileRegression, sqr::Bool=false)
    s = sum(abs2, m.wrkres)
    if sqr; s else sqrt(s) end
end

function location_variance(m::QuantileRegression, sqr::Bool=false)
    v = dispersion(m, true)
    v *= m.τ * (1 - m.τ) * 2*π
#    v *= (1/2 - m.τ * (1 - m.τ)) * 2*π
    v *= (nobs(m)/dof_residual(m))
    if sqr; v else sqrt(v) end
end


"""
    nobs(m::QuantileRegression)
For linear and generalized linear models, returns the number of elements of the response.
"""
nobs(m::QuantileRegression)::Int = if !isempty(m.wts); count(!iszero, m.wts) else size(m.y, 1) end

coef(m::QuantileRegression) = m.β

stderror(m::QuantileRegression) = location_variance(m, false) .* sqrt.(diag(vcov(m)))

weights(m::QuantileRegression{T}) where T<:AbstractFloat = if isempty(m.wts); weights(ones(T, length(m.y))) else weights(m.wts) end

response(m::QuantileRegression) = m.y

isfitted(m::QuantileRegression) = m.fitted

fitted(m::QuantileRegression) = m.y - m.wrkres

residuals(m::QuantileRegression) = m.wrkres

predict(m::QuantileRegression, newX::AbstractMatrix) = newX * coef(m)
predict(m::QuantileRegression) = fitted(m)

function nulldeviance(m::QuantileRegression)
    μ = quantile(m.y, m.τ)
    if isempty(m.wts)
        sum(_objective.(m.wrkres, Ref(m.τ)))
    else
        sum(m.wts .* _objective.(m.wrkres, Ref(m.τ)))
    end
end

deviance(m::QuantileRegression) = sum(_objective.(m.wrkres, Ref(m.τ)))

## TODO: define correctly the loglikelihood of the full model
fullloglikelihood(m::QuantileRegression) = 0
loglikelihood(m::QuantileRegression) = fullloglikelihood(m) - deviance(m)/2

modelmatrix(m::QuantileRegression) = m.X

vcov(m::QuantileRegression) = inv(Hermitian(float(Matrix(modelmatrix(m)' * (weights(m) .* modelmatrix(m))))))

projectionmatrix(m::QuantileRegression) = Hermitian(modelmatrix(m) * vcov(m) * modelmatrix(m)') .* weights(m)

leverage_weights(m::QuantileRegression) = sqrt.(1 .- leverage(m))
