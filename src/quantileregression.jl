

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
    dropmissing::Bool=false,
    wts::Union{Nothing,Symbol,FPVector}=nothing,
    contrasts::AbstractDict{Symbol,Any}=Dict{Symbol,Any}(),
    kwargs...,
) where {M<:QuantileRegression}
    # Extract arrays from data using formula
    f, y, X, extra = modelframe(f, data, contrasts, dropmissing, M; wts=wts)
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


function interiormethod(X::AbstractMatrix{T}, y::AbstractVector{T}, τ::T; kwargs...) where {T<:AbstractFloat}
    n, p = size(X)
    interiormethod!(zeros(T, p), zeros(T, n), X, y, τ; kwargs...)
end

function interiormethod!(
    βout::AbstractVector{T}, 
    rout::AbstractVector{T}, 
    X::AbstractMatrix{T}, 
    y::AbstractVector{T}, 
    τ::T; 
    wts::AbstractVector{T}=zeros(T, 0), 
    verbose::Bool=false,
) where {T<:AbstractFloat}
    model = Tulip.Model{T}()
    pb = model.pbdata  # Internal problem data

    n, p = size(X)

    # Define variables
    β = Vector{Int}(undef, p)
    u = Vector{Int}(undef, n)
    v = Vector{Int}(undef, n)
    for i in 1:p
        β[i] = Tulip.add_variable!(pb, Int[], T[], 0.0 , -Inf , Inf, "β$i")
    end
    for i in 1:n
        u[i] = Tulip.add_variable!(pb, Int[], T[], τ   ,  0.0 , Inf, "u$i")
    end
    for i in 1:n
        v[i] = Tulip.add_variable!(pb, Int[], T[], 1-τ ,  0.0 , Inf, "v$i")
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
    n = if isempty(m.wts)
        length(m.y)
    else
        count(!iszero, m.wts)
    end

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

    r = copy(residuals(m))
    if !isempty(m.wts)
        inds = findall(!iszero, m.wts)
        r = r[inds] .* m.wts[inds]
    end
    sort!(r)

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
    v *= (wobs(m) / dof_residual(m))
    return sqr ? v : sqrt(v)
end


"""
    nobs(m::QuantileRegression)::Integer
Returns the number of elements of the response.
For models with prior weights, return the number of non-zero weights.
"""
function StatsAPI.nobs(m::QuantileRegression)::Integer
    if !isempty(m.wts)
        ## Suppose that the weights are probability weights
        count(!iszero, m.wts)
    else
        length(m.y)
    end
end

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

function GLM.dispersion(m::QuantileRegression)
    if isempty(m.wts)
        mean(abs.(residuals(m)))
    else
        mean(abs.(residuals(m)), weights(m.wts))
    end
end

StatsAPI.stderror(m::QuantileRegression) = location_variance(m, false) .* sqrt.(diag(vcov(m)))

StatsAPI.weights(m::QuantileRegression) = m.wts

workingweights(m::QuantileRegression) = m.wrkres

StatsAPI.response(m::QuantileRegression) = m.y

StatsAPI.isfitted(m::QuantileRegression) = m.fitted

StatsAPI.fitted(m::QuantileRegression) = m.y - m.wrkres

StatsAPI.residuals(m::QuantileRegression) = m.wrkres

StatsAPI.predict(m::QuantileRegression, newX::AbstractMatrix) = newX * coef(m)
StatsAPI.predict(m::QuantileRegression) = fitted(m)

scale(m::QuantileRegression) = dispersion(m)

hasformula(m::QuantileRegression) = isnothing(m.formula) ? false : true

function StatsModels.formula(m::QuantileRegression)
    if !hasformula(m)
        throw(ArgumentError("model was fitted without a formula"))
    end
    return m.formula
end

function StatsAPI.nulldeviance(m::QuantileRegression)
    # Compute location of the null model
    μ = if !hasintercept(m)
        zero(eltype(m.y))
    elseif isempty(m.wts)
        quantile(m.y, m.τ)
    else
        quantile(m.y, weights(m.wts), m.τ)
    end

    # Sum deviance for each observation
    dev = 0
    if isempty(m.wts)
        @inbounds for i in eachindex(m.y)
            dev += 2 * _objective(m.y[i] - μ, m.τ)
        end
    else
        @inbounds for i in eachindex(m.y, m.wts)
            dev += 2 * m.wts[i] * _objective(m.y[i] - μ, m.τ)
        end
    end
    dev
end

StatsAPI.deviance(m::QuantileRegression) = 2 * sum(_objective.(m.wrkres, Ref(m.τ)))

## Loglikelihood of the full model
## l = Σi log fi = Σi log ( τ*(1-τ)  exp(-objective(xi)) ) = n log (τ*(1-τ)) - Σi objective(xi)
fullloglikelihood(m::QuantileRegression) = wobs(m) * log(m.τ * (1 - m.τ))

StatsAPI.loglikelihood(m::QuantileRegression) = fullloglikelihood(m) - deviance(m) / 2

StatsAPI.nullloglikelihood(m::QuantileRegression) = fullloglikelihood(m) - nulldeviance(m) / 2

StatsAPI.modelmatrix(m::QuantileRegression) = m.X

function StatsAPI.vcov(m::QuantileRegression)
    X = modelmatrix(m)
    wXt = isempty(weights(m)) ? X' : (X .* weights(m))'
    return inv(Hermitian(float(Matrix(wXt * X))))
end

function projectionmatrix(m::QuantileRegression)
    X = modelmatrix(m)
    wXt = isempty(weights(m)) ? X' : (X .* weights(m))'
    return Hermitian(X * vcov(m) * wXt)
end


