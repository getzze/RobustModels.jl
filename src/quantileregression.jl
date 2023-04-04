

import Tulip


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
* `formula`: either a `FormulaTerm` object or `nothing`
* `fitdispersion`: if true, the dispersion is estimated otherwise it is kept fixed
* `fitted`: if true, the model was already fitted
"""
mutable struct QuantileRegression{
    T<:AbstractFloat,
    M<:AbstractMatrix{T},
    V<:AbstractVector{T},
} <: AbstractRobustModel{T}
    τ::T
    X::M
    β::V
    y::V
    wts::V
    wrkres::V
    formula::Union{FormulaTerm,Nothing}
    fitdispersion::Bool
    fitted::Bool

    function QuantileRegression(
        τ::T,
        X::M,
        y::V,
        wts::V,
        formula::Union{FormulaTerm,Nothing},
    ) where {V<:AbstractVector{T},M<:AbstractMatrix{T}} where {T<:AbstractFloat}
        n = length(y)
        m, p = size(X)
        ll = length(wts)
        ll == 0 || ll == n || error("length of wts is $ll, must be $n or 0")
        m == n || error("X has $m rows, must be like y, $n")
        new{T,M,V}(τ, X, zeros(T, p), y, wts, similar(y), formula, false, false)
    end
end


function Base.show(io::IO, obj::QuantileRegression)
    msg = "Quantile regression for quantile: $(obj.τ)\n\n"
    if hasformula(obj)
        msg *= "$(formula(obj))\n\n"
    end
    msg *= "Coefficients:\n"
    println(io, msg, coeftable(obj))
end

function Base.show(io::IO, obj::TableRegressionModel{M,T}) where {T,M<:QuantileRegression}
    msg = "Quantile regression for quantile: $(obj.τ)\n\n"
    if hasformula(obj)
        msg *= "$(formula(obj))\n\n"
    end
    msg *= "Coefficients:\n"
    println(io, msg, coeftable(obj))
end

function Base.getproperty(mm::TableRegressionModel{M}, f::Symbol) where {M<:QuantileRegression}
    if f == :τ
        return mm.model.τ
    else
        return getfield(mm, f)
    end
end

"""
    quantreg(X, y, args...; kwargs...)

An alias for `fit(QuantileRegression, X, y; kwargs...)`.

The arguments `X` and `y` can be a `Matrix` and a `Vector` or a `Formula` and a `DataFrame`.
"""
quantreg(X, y, args...; kwargs...) = fit(QuantileRegression, X, y, args...; kwargs...)


"""
    fit(::Type{M}, X::Union{AbstractMatrix{T},SparseMatrixCSC{T}},
        y::AbstractVector{T}; quantile::AbstractFloat=0.5,
        dofit::Bool          = true,
        wts::FPVector        = similar(y, 0),
        fitdispersion::Bool  = false,
        fitargs...) where {M<:QuantileRegression, T<:AbstractFloat}

Fit a quantile regression model with the model matrix (or formula) X and response vector (or dataframe) y.

It is solved using the exact interior method.

# Arguments

- `X`: the model matrix (it can be dense or sparse) or a formula
- `y`: the response vector or a dataframe.

# Keywords

- `quantile::AbstractFloat=0.5`: the quantile value for the regression, between 0 and 1.
- `dofit::Bool = true`: if `false`, return the model object without fitting;
- `wts::Vector = []`: a weight vector, should be empty if no weights are used;
- `fitdispersion::Bool = false`: reevaluate the dispersion;
- `fitargs...`: other keyword arguments like `verbose` to print iteration details.

# Output

the RobustLinearModel object.

"""
function StatsAPI.fit(
    ::Type{M},
    X::Union{AbstractMatrix{T},SparseMatrixCSC{T}},
    y::AbstractVector{T};
    quantile::AbstractFloat=0.5,
    dofit::Bool=true,
    wts::FPVector=similar(y, 0),
    fitdispersion::Bool=false,
    contrasts::AbstractDict{Symbol,Any}=Dict{Symbol,Any}(),  # placeholder
    __formula::Union{Nothing,FormulaTerm}=nothing,
    fitargs...,
) where {M<:QuantileRegression,T<:AbstractFloat}

    # Check that X and y have the same number of observations
    if size(X, 1) != size(y, 1)
        mess = "number of rows in X and y must match: $(size(X, 1)) != $(size(y, 1))"
        throw(DimensionMismatch(mess))
    end

    # Check quantile is an allowed value
    (0 < quantile < 1) ||
        error("quantile should be a number between 0 and 1 excluded: $(quantile)")

    m = QuantileRegression(quantile, X, y, wts, __formula)
    return dofit ? fit!(m; fitargs...) : m
end

function StatsAPI.fit(
    ::Type{M},
    f::FormulaTerm,
    data;
    wts::Union{Nothing,Symbol,FPVector}=nothing,
    contrasts::AbstractDict{Symbol,Any}=Dict{Symbol,Any}(),
    kwargs...,
) where {M<:QuantileRegression}
    # Extract arrays from data using formula
    f, y, X, extra = modelframe(f, data, contrasts, M; wts=wts)
    # Call the `fit` method with arrays
    fit(M, X, y; wts=extra.wts, contrasts=contrasts, __formula=f, kwargs...)
end


"""
    refit!(m::QuantileRegression,
          [y::FPVector ;
           verbose::Bool=false,
           quantile::Union{Nothing,
           AbstractFloat}=nothing,
          ]
    )

Refit the `QuantileRegression` model with the new values for the response,
weights and quantile. This function assumes that `m` was correctly initialized.
"""
function refit!(m::QuantileRegression, y::FPVector; kwargs...)
    # Update y
    # Check that the old and new y have the same number of rows
    if size(m.y, 1) != size(y, 1)
        mess =
            "the new response vector should have the same dimension: " *
            "$(size(m.y, 1)) != $(size(y, 1))"
        throw(DimensionMismatch(mess))
    end
    copyto!(m.y, y)

    refit!(m; kwargs...)
end

function refit!(
    m::QuantileRegression{T};
    quantile::Union{Nothing,AbstractFloat}=nothing,
    wts::Union{Nothing,FPVector}=nothing,
    kwargs...,
) where {T}

    # Update wts
    n = length(m.y)
    if !isnothing(wts) && (length(wts) in (0, n))
        copy!(m.wts, wts)
    end

    # Update quantile
    if !isnothing(quantile)
        (0 < quantile < 1) ||
            error("quantile should be a number between 0 and 1 excluded: $(quantile)")
        m.τ = quantile
    end

    m.fitted = false
    fit!(m; kwargs...)
end


"""
    fit!(m::QuantileRegression;
         verbose::Bool=false,
         quantile::Union{Nothing, AbstractFloat}=nothing,
         correct_leverage::Bool=false,
         kwargs...)

Optimize the objective of a `QuantileRegression`.  When `verbose` is `true` the values
of the objective and the parameters are printed on stdout at each function evaluation.

This function assumes that `m` was correctly initialized.
This function returns early if the model was already fitted, instead call `refit!`.
"""
function StatsAPI.fit!(
    m::QuantileRegression{T};
    verbose::Bool=false,
    quantile::Union{Nothing,AbstractFloat}=nothing,
    correct_leverage::Bool=false,
    kwargs...,
) where {T}

    # Return early if model has the fit flag set
    m.fitted && return m

    if correct_leverage
        wts = m.wts
        copy!(wts, leverage_weights(m))
        ## TODO: maybe multiply by the old wts?
    end

    # Update quantile
    if !isnothing(quantile)
        (0 < quantile < 1) || throw(
            DomainError(quantile, "quantile should be a number between 0 and 1 excluded"),
        )
        m.τ = quantile
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
    model = Tulip.Model{Float64}()
    pb = model.pbdata  # Internal problem data

    n, p = size(X)

    # Define variables
    β = Vector{Int}(undef, p)
    u = Vector{Int}(undef, n)
    v = Vector{Int}(undef, n)
    for i in 1:p
        β[i] = Tulip.add_variable!(pb, Int[], Float64[], 0.0 , -Inf , Inf, "β$i")
    end
    for i in 1:n
        u[i] = Tulip.add_variable!(pb, Int[], Float64[], τ   ,  0.0 , Inf, "u$i")
    end
    for i in 1:n
        v[i] = Tulip.add_variable!(pb, Int[], Float64[], 1-τ ,  0.0 , Inf, "v$i")
    end
    #    @variable(model, β[1:p])
    #    @variable(model, u[1:n] >= 0)
    #    @variable(model, v[1:n] >= 0)

    # Define objective
    # Nothing to do, already defined with the variables
    #    e = ones(n)
    #    @objective(model, Min, τ*dot(e, u) + (1-τ)*dot(e, v) )

    Wy, WX = if isempty(wts)
        y, X
    else
        (wts .* y), (wts .* X)
    end

    # Define constraints
    resid = Vector{Int}(undef, n)
    for i in 1:n
        ci = vcat(β, [u[i], v[i]])
        # Call Vector to transform to dense vector
        cci = vcat(Vector(WX[i, :]), [1.0, -1.0])
        vi = Wy[i]
        resid[i] = Tulip.add_constraint!(pb, ci, cci, vi, vi, "resid$i")
    end
    #    @constraint(model, resid, Wy .== WX * β + u - v)

    Tulip.optimize!(model)

    copyto!(βout, model.solution.x[β])
    copyto!(rout, model.solution.x[u] - model.solution.x[v])
    #    copyto!(βout, value.(β))
    #    copyto!(rout, value.(u) - value.(v))

    if verbose
        println("coef: ", βout)
        println("res: ", rout)
    end
    βout, rout
end

_objective(r::AbstractFloat, τ::AbstractFloat) = (τ - (r < 0)) * r

"""
    hall_sheather_bandwidth()
Optimal bandwidth for sparsity estimation according to Hall and Sheather (1988)
"""
function hall_sheather_bandwidth(q::Real, n::Int, α::Real=0.05)
    zα = quantile(Normal(), 1 - α / 2)
    ## Estimate of r=s/s" from a Normal distribution
    zq = quantile(Normal(), q)
    f = pdf(Normal(), zq)
    r = f^2 / (2 * zq^2 + 1)
    (zα^2 * 1.5 * r / n)^(1 / 5)
end


"""
    jones_bandwidth()
Optimal bandwidth for kernel sparsity estimation according to Jones (1992)
"""
function jones_bandwidth(q::Real, n::Int; kernel=:epanechnikov)
    ## From the kernel k(u) with domain [-1, 1]:
    ## kk = ∫k²(u)du / ( ∫u²k(u)du )^2

    kk = if kernel == :epanechnikov
        # For Epanechnikov kernel: kk = 3/5 / (1/5)^2
        15
    elseif kernel == :triangle
        # For triangle kernel (k(u) = 1 - |u|): kk = 2/3 / (1/6)^2
        24
    elseif kernel == :window
        # That is the Bofinger estimate of the bandwidth
        # For window kernel (k(u) = 1/2): kk = 1/2 / (1/3)^2
        4.5
    else
        error("kernel $(kernel) is not defined for sparsity estimation.")
    end

    ## Estimate of r=s/s" from a Normal distribution
    zq = quantile(Normal(), q)
    f = pdf(Normal(), zq)
    r = f^2 / (2 * zq^2 + 1)
    (kk * r^2 / n)^(1 / 5)
end

"""
    bofinger_bandwidth()
Optimal bandwidth for sparsity estimation according to Bofinger (1975)
"""
function bofinger_bandwidth(q::Real, n::Int)
    return jones_bandwidth(q, n; kernel=:window)
end


function epanechnikov_kernel(x::Real)
    if abs(x) < 1
        return 3 / 4 * (1 - x^2)
    else
        return zero(typeof(x))
    end
end

function triangle_kernel(x::Real)
    if abs(x) < 1
        return 1 - abs(x)
    else
        return zero(typeof(x))
    end
end

function window_kernel(x::Real)
    if abs(x) < 1
        return one(typeof(x)) / 2
    else
        return zero(typeof(x))
    end
end

"""
    sparcity(m::QuantileRegression, α::Real=0.05)

Compute the sparcity or quantile density ŝ(0)
using formula from Jones (1992) - Estimating densities, quantiles,
quantile densities and density quantiles.

"""
function sparcity(
    m::QuantileRegression;
    bw_method::Symbol=:jones,
    α::Real=0.05,
    kernel::Symbol=:epanechnikov,
)
    u = m.τ
    n = nobs(m)
    ## Select the optimal bandwidth from different methods
    h = if bw_method == :jones
        jones_bandwidth(u, n; kernel=kernel)
    elseif bw_method == :bofinger
        bofinger_bandwidth(u, n)
    elseif bw_method == :hall_sheather
        hall_sheather_bandwidth(u, n, α)
    else
        error(
            "only :jones, :bofinger and :hall_sheather methods for estimating the"*
            " optimal bandwidth are allowed: $(bw_method)",
        )
    end

    ## Select kernel for density estimation
    k = if kernel == :epanechnikov
        epanechnikov_kernel
    elseif kernel == :triangle
        triangle_kernel
    elseif kernel == :window
        window_kernel
    else
        error("only :epanechnikov, :triangle and :window kernels are allowed: $(kernel)")
    end

    r = sort(residuals(m))

    s0 = 0
    for i in eachindex(r)
        if i == 1
            s0 += r[i] / h * k(u / h)
        else
            s0 += (r[i] - r[i-1]) / h * k((u - (i - 1) / n) / h)
            if i == lastindex(r)
                s0 += r[i] / h * k((u - 1) / h)
            end
        end
    end
    s0
end

function location_variance(
    m::QuantileRegression,
    sqr::Bool=false;
    bw_method::Symbol=:jones,
    α::Real=0.05,
    kernel::Symbol=:epanechnikov,
)
    v = sparcity(m; bw_method=bw_method, α=α, kernel=kernel)^2
    v *= m.τ * (1 - m.τ)
    v *= (nobs(m) / dof_residual(m))
    return sqr ? v : sqrt(v)
end


"""
    nobs(m::QuantileRegression)
For linear and generalized linear models, returns the number of elements of the response.
"""
StatsAPI.nobs(m::QuantileRegression)::Integer = length(m.y)

"""
    wobs(m::QuantileRegression)
For unweighted linear models, equals to ``nobs``, it returns the number of elements of the response.
For models with prior weights, return the sum of the weights.
"""
function wobs(m::QuantileRegression)
    if !isempty(m.wts)
        ## Suppose that the weights are probability weights
        sum(m.wts)
    else
        oftype(sum(one(eltype(m.wts))), nobs(m))
    end
end

StatsAPI.coef(m::QuantileRegression) = m.β

GLM.dispersion(m::QuantileRegression) = mean(abs.(residuals(m)))

StatsAPI.stderror(m::QuantileRegression) = location_variance(m, false) .* sqrt.(diag(vcov(m)))

function StatsAPI.weights(m::QuantileRegression{T}) where {T<:AbstractFloat}
    if isempty(m.wts)
        weights(ones(T, length(m.y)))
    else
        weights(m.wts)
    end
end

workingweights(m::QuantileRegression) = m.wrkres

StatsAPI.response(m::QuantileRegression) = m.y

StatsAPI.isfitted(m::QuantileRegression) = m.fitted

StatsAPI.islinear(m::QuantileRegression) = true

StatsAPI.fitted(m::QuantileRegression) = m.y - m.wrkres

StatsAPI.residuals(m::QuantileRegression) = m.wrkres

StatsAPI.predict(m::QuantileRegression, newX::AbstractMatrix) = newX * coef(m)
StatsAPI.predict(m::QuantileRegression) = fitted(m)

function StatsAPI.nulldeviance(m::QuantileRegression)
    μ = quantile(m.y, m.τ)
    if isempty(m.wts)
        sum(_objective.(m.wrkres, Ref(m.τ)))
    else
        sum(m.wts .* _objective.(m.wrkres, Ref(m.τ)))
    end
end

StatsAPI.deviance(m::QuantileRegression) = sum(_objective.(m.wrkres, Ref(m.τ)))

## TODO: define correctly the loglikelihood of the full model
fullloglikelihood(m::QuantileRegression) = 0

StatsAPI.loglikelihood(m::QuantileRegression) = fullloglikelihood(m) - deviance(m) / 2

StatsAPI.nullloglikelihood(m::QuantileRegression) = fullloglikelihood(m) - nulldeviance(m) / 2

StatsAPI.modelmatrix(m::QuantileRegression) = m.X

StatsAPI.vcov(m::QuantileRegression) =
    inv(Hermitian(float(Matrix(modelmatrix(m)' * (weights(m) .* modelmatrix(m))))))

projectionmatrix(m::QuantileRegression) =
    Hermitian(modelmatrix(m) * vcov(m) * modelmatrix(m)') .* weights(m)

leverage_weights(m::QuantileRegression) = sqrt.(1 .- leverage(m))

StatsModels.hasintercept(m::QuantileRegression) = _hasintercept(modelmatrix(m))

hasformula(m::QuantileRegression) = isnothing(m.formula) ? false : true

function StatsModels.formula(m::QuantileRegression)
    if !hasformula(m)
        throw(ArgumentError("model was fitted without a formula"))
    end
    return m.formula
end
