
using LinearAlgebra: qr, norm, eigmin, eigmax, rdiv!
using SparseArrays: sparse


####################################
### LinPred methods
####################################

initpred!(p::LinPred, wt::AbstractVector{T}=T[], σ::Real=one(T); verbose::Bool=false) where {T<:BlasReal} = p

solvepred!(p::LinPred, r::AbstractVector{T}) where {T<:BlasReal} = delbeta!(p, r)

solvepred!(p::LinPred, r::AbstractVector{T}, wts::AbstractVector{T}) where {T<:BlasReal} =
    delbeta!(p, r, wts)

updatepred!(p::LinPred, args...; kwargs...) = p

function update_beta!(
    p::LinPred,
    r::AbstractVector{T},
    wts::AbstractVector{T}=T[],
    σ2::T=one(T);  # placeholder
    verbose::Bool=false,
) where {T<:BlasReal}
    if isempty(wts)
        return solvepred!(p, r)
    else
        return solvepred!(p, r, wts)
    end
end

function initpred!(
    p::Union{DensePredCG{T},SparsePredCG{T}},
    wt::AbstractVector{T}=T[],
    σ::Real=one(T);  # placeholder
    verbose::Bool=false,
) where {T<:BlasReal}
    # Initialize the scratchm1 matrix
    if isempty(wt)
        scr = transpose(copy!(p.scratchm1, p.X))
    else
        scr = transpose(broadcast!(*, p.scratchm1, wt, p.X))
    end
    # Initialize the Gram matrix
    mul!(p.Σ, scr, p.X)
    p
end

function solvepred!(
    p::Union{DensePredCG{T},SparsePredCG{T}},
    r::AbstractVector{T},
) where {T<:BlasReal}
    ## Assume that the relevant matrices are pre-computed
    # Compute the left-hand side.
    mul!(p.scratchbeta, transpose(p.scratchm1), r)

    # Solve the linear system
    cg!(p.delbeta, Hermitian(p.Σ, :U), p.scratchbeta)
    p
end


####################################
### IPODResp
####################################

"""
    IPODResp

Robust Θ-IPOD linear response structure.

Solve the following minimization problem:

```math
\\min \\left\\lVert(\\dfrac{\\mathbf{y} - \\mathbf{X}\\mathbf{\\beta} - \\mathbf{gamma}}
{\\hat{\\sigma}}\right\\rVert^2_2 + P_c\\left(\\dfrac{\\mathbf{\\gamma}}{\\hat{\\sigma}}\\right)
```

# Fields

- `loss`: loss used for the model
- `y`: response vector
- `μ`: mean response vector
- `wts`: prior case weights.  Can be of length 0.
- `outliers`: outlier vector subtracted from `y` to form the working response.
- `σ`: current estimate of the scale or dispersion
- `wrky`: working response
- `wrkres`: working residuals

"""
mutable struct IPODResp{T<:AbstractFloat,V<:AbstractVector{T},L<:LossFunction} <: RobustResp{T}
    "`loss`: loss used for the model"
    loss::L
    "`y`: response vector"
    y::V
    "`μ`: mean response vector"
    μ::V
    "`precision`: prior precision weights. Can be of length 0."
    precision::V
    "`outliers`: outlier vector subtracted from `y` to form the working response."
    outliers::V
    "`σ`: current estimate of the scale or dispersion"
    σ::T
    "`wrky`: working response."
    wrky::V
    "`wrkres`: working residuals"
    wrkres::V

    function IPODResp{T,V,L}(
        l::L,
        y::V,
        precision::V,
        outliers::V,
        σ::Real,
    ) where {L<:LossFunction,V<:AbstractVector{T}} where {T<:AbstractFloat}
        n = length(y)
        ll = length(precision)
        ll == 0 || ll == n || throw(DimensionMismatch("length of precision is $ll, must be $n or 0"))
        σ > 0 || throw(ArgumentError("σ must be positive: $σ"))
        wrky = y - outliers
        new{T,V,L}(l, y, zeros(T, n), precision, outliers, convert(T, σ), wrky, copy(wrky))
    end
end

"""
    IPODResp(l::L, y::V, precision::V, outliers::V=zeros(eltype(y), length(y)), σ::Real=1)
            where {L<:LossFunction, V<:FPVector}

Initialize the Robust Θ-IPOD linear response structure.

"""
function IPODResp(
    l::L,
    y::V,
    precision::V=T[],
    outliers::V=zeros(T, length(y)),
    σ::Real=1,
) where {L<:LossFunction, V<:AbstractVector{T}} where {T<:AbstractFloat}
    r = IPODResp{T,V,L}(l, y, precision, outliers, T(σ))
    initresp!(r)
    return r
end

"""
    IPODResp(l::L, y, wts, outliers=zeros(eltype(y), length(y)), σ::Real=1) where {L<:LossFunction}

Convert the arguments to float arrays.
"""
IPODResp(l::L, y, precision=eltype(y)[], outliers=zeros(eltype(y), length(y)), σ::Real=1) where {L<:LossFunction} =
    IPODResp(l, float(collect(y)), float(collect(precision)), float(collect(outliers)), σ)

function Base.getproperty(r::IPODResp, s::Symbol)
    if s ∈ (:mu, :η, :eta)
        r.μ
    elseif s ∈ (:sigma, :scale)
        r.σ
    elseif s ∈ (:γ, :gamma)
        r.outliers
    elseif s ∈ (:λ, :lambda)
        r.precision
    else
        getfield(r, s)
    end
end

"""
    initresp!(r::IPODResp)

Initialize the response structure.
"""
function initresp!(r::IPODResp)
    # Set working y
    broadcast!(-, r.wrky, r.y, r.outliers)

    # Set residual (without offset)
    broadcast!(-, r.wrkres, r.wrky, r.μ)
end

function update_outliers!(r::IPODResp)
    if isempty(r.precision)
        λi = one(eltype(r.y))
        @inbounds @simd for i in eachindex(r.y, r.μ, r.wrkres, r.outliers)
            # Use threshold to compute outliers
            r.outliers[i] = r.σ * threshold(r.loss, (r.y[i] - r.μ[i]) / r.σ, λi)
            r.wrky[i] = r.y[i] - r.outliers[i]
            r.wrkres[i] = r.wrky[i] - r.μ[i]
        end
    else
        @inbounds @simd for i in eachindex(r.y, r.μ, r.wrkres, r.outliers, r.precision)
            λi = r.precision[i]
            # Use threshold to compute outliers
            r.outliers[i] = r.σ * threshold(r.loss, (r.y[i] - r.μ[i]) / r.σ, λi)
            r.wrky[i] = r.y[i] - r.outliers[i]
            r.wrkres[i] = r.wrky[i] - r.μ[i]
        end
    end
    r
end

function update_residuals!(r::IPODResp)
    # update residuals
    @. r.wrkres = r.wrky - r.μ
    r
end

function update_scale!(r::IPODResp)
    r.σ = sqrt((sum(abs2, r.wrkres) + dot(r.outliers, r.wrkres)) / length(r.y))
    r
end

outliers_criteria(r::IPODResp{T}, γold::AbstractVector{T}) where {T<:AbstractFloat} =
    maximum(abs(γold[i] - r.outliers[i]) / r.σ for i in eachindex(γold, r.outliers))

"""
    dev_criteria(r::IPODResp)

Deviance part coming from the loss and the response struct.
"""
dev_criteria(r::IPODResp) = sum(abs2, r.wrkres) / r.σ^2

"""
    outliers(r::IPODResp)

Returns the vector of the outlier part γ of the response y, so that `(y - γ) | X ~ Normal`.
"""
outliers(r::IPODResp) = r.outliers


####################################
### IPODRegression
####################################


"""
    IPODRegression

Robust regression using the Φ-IPOD algorithm

## Fields

* `resp`: the [`IPODResp`](@ref) structure.
* `pred`: the [`IPODPred`](@ref) structure.
* `formula`: either a `FormulaTerm` object or `nothing`
* `wts`: the prior observation weights (can be empty).
* `fitdispersion`: if true, the dispersion is estimated otherwise it is kept fixed
* `fitted`: if true, the model was already fitted
"""
mutable struct IPODRegression{
    T<:AbstractFloat,
    R<:IPODResp{T},
    P<:Union{LinPred,AbstractRegularizedPred},
    V<:AbstractVector{T},
} <: AbstractRobustModel{T}
    resp::R
    pred::P
    formula::Union{FormulaTerm,Nothing}
    wts::V
    fitdispersion::Bool
    fitted::Bool
end

function IPODRegression(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    loss::LossFunction,
    penalty::Union{Nothing,PenaltyFunction}=nothing;
    method::Symbol=:auto,  # :chol, :qr, :cg, :cgd, :fista, :ama, :admm
    wts::FPVector=similar(y, 0),
    precision::FPVector=similar(y, 0),
    fitdispersion::Bool=false,
    pivot::Bool=false,
    formula::Union{Nothing,FormulaTerm}=nothing,
    use_backtracking::Bool=false,
    AMA::Bool=false,
    A::AbstractMatrix{<:Real}=zeros(T, 0, 0),
    b::AbstractVector{<:Real}=zeros(T, 0),
    restart::Bool=true,
    adapt::Bool=true,
    penalty_omit_intercept::Bool=true,
) where {T<:AbstractFloat}

    # Check that X and y have the same number of observations
    n, p = size(X)
    n == size(y, 1) || throw(DimensionMismatch("number of rows in X and y must match"))
    ll = size(wts, 1)
    ll in (0, n) || throw(DimensionMismatch("length of wts is $ll, must be 0 or $n."))

    # Response object
    rr = IPODResp(loss, y, precision)

    # Method
    nopen_methods = (:auto, :chol, :cg, :qr)
    pen_methods = (:auto, :cgd, :fista, :ama, :admm)
    if isnothing(penalty) && method ∉ nopen_methods
        @warn("Incorrect method `:$(method)` without a penalty, should be one of $(nopen_methods)")
        method = :auto
    elseif !isnothing(penalty) && method ∉ pen_methods
        @warn("Incorrect method `:$(method)` with a penalty, should be one of $(pen_methods)")
        method = :auto
    end

    # Predictor without penalty
    if isnothing(penalty)
        pred = if method === :cg
            cgpred(X)
        elseif method === :qr
            qrpred(X, pivot)
        elseif method in (:chol, :auto)
            cholpred(X, pivot)
        else
            error("without penalty, method :$method is not allowed, should be in: $nopen_methods")
        end

    # Predictor with penalty
    else
        # Penalty
        new_penalty = try
            intercept_col = penalty_omit_intercept ? get_intercept_col(X, formula) : nothing
            concrete(penalty, p, intercept_col)
        catch
            error("penalty is not compatible with coefficients size $p: $(penalty)")
        end

        # Predictor
        pred = if method === :fista
            FISTARegPred(X, new_penalty, wts, use_backtracking)
        elseif method === :ama
            AMARegPred(X, new_penalty, wts, A, b, restart)
        elseif method === :admm
            ADMMRegPred(X, new_penalty, wts, A, b, restart, adapt)
        elseif method in (:cgd, :auto)
            CGDRegPred(X, new_penalty, wts)
        else
            error("with penalty, method :$method is not allowed, should be in: $pen_methods")
        end
    end

    return IPODRegression(rr, pred, formula, wts, fitdispersion, false)
end

function Base.getproperty(r::IPODRegression, s::Symbol)
    if s ∈ (:beta0, :β)
        r.pred.beta0
    elseif s ∈ (:delbeta, :dβ)
        r.pred.delbeta
    else
        getfield(r, s)
    end
end

function Base.show(io::IO, obj::IPODRegression)
    msg = "Robust Θ-IPOD regression with $(obj.resp.loss)\n\n"
    if hasformula(obj)
        msg *= "$(formula(obj))\n\n"
    end
    msg *= "Coefficients:\n"
    println(io, msg, coeftable(obj))
end

####################################
### Interface
####################################

hasformula(m::IPODRegression) = isnothing(m.formula) ? false : true

function StatsModels.formula(m::IPODRegression)
    if !hasformula(m)
        throw(ArgumentError("model was fitted without a formula"))
    end
    return m.formula
end

StatsAPI.modelmatrix(r::IPODRegression) = r.pred.X

function StatsAPI.vcov(r::IPODRegression, wt::AbstractVector)
    wXt = isempty(wt) ? modelmatrix(r)' : (modelmatrix(r) .* wt)'
    return inv(Hermitian(float(Matrix(wXt * modelmatrix(r)))))
end
StatsAPI.vcov(r::IPODRegression) = vcov(r, weights(r))

function projectionmatrix(r::IPODRegression, wt::AbstractVector)
    X = modelmatrix(r)
    wXt = isempty(wt) ? X' : (wt .* X)'
    return Hermitian(X * vcov(r, wt) * wXt)
end
projectionmatrix(r::IPODRegression) = projectionmatrix(r, weights(r))

# Define the methods, but all `variant` give the same result
for fun in (:vcov, :projectionmatrix, :leverage, :leverage_weights)
    @eval begin
        $(fun)(m::IPODRegression, variant::Symbol) = $(fun)(m)
    end
end

function GLM.dispersion(
    r::IPODRegression,
    sqr::Bool=false,
)
    wts = weights(r)
    res = residuals(r)
    dofres = dof_residual(r)

    s = if isempty(wts)
        sum(abs2, res) / dofres
    else
        sum(i -> wts[i] * abs2(res[i]), eachindex(wts, res)) / dofres
    end

    return sqr ? s : sqrt(s)
end

"""
    location_variance(r::RobustLinResp, sqr::Bool=false)

Compute the part of the variance of the coefficients `β` that is due to the encertainty
from the location. If `sqr` is `false`, return the standard deviation instead.

From Maronna et al., Robust Statistics: Theory and Methods, Equation 4.49
"""
location_variance(r::IPODRegression, sqr::Bool=false) = dispersion(r, sqr)

StatsAPI.stderror(r::IPODRegression) = location_variance(r, false) .* sqrt.(diag(vcov(r)))

## Loglikelihood of the full model
## l = Σi log fi = Σi log ( 1/(σ * Z) exp( - ρ(ri/σ) ) = -n (log σ + log Z) - Σi ρ(ri/σ)
fullloglikelihood(r::IPODRegression) = -wobs(r) *(log(r.resp.σ) + log(2 * π) / 2)

StatsAPI.deviance(r::IPODRegression) = sum(abs2, residuals(r)) / scale(r) ^ 2

function StatsAPI.nulldeviance(r::IPODRegression)
    y = response(r)
    wts = weights(r)

    # Compute location of the null model
    μ = if !hasintercept(r)
        zero(eltype(y))
    elseif isempty(wts)
        mean(y)
    else
        mean(y, weights(wts))
    end

    # Sum deviance for each observation
    dev = 0
    if isempty(wts)
        @inbounds for i in eachindex(y)
            dev += abs2((y[i] - μ) / scale(r))
        end
    else
        @inbounds for i in eachindex(y, wts)
            dev += wts[i] * abs2((y[i] - μ) / scale(r))
        end
    end
    dev
end

StatsAPI.loglikelihood(r::IPODRegression) = fullloglikelihood(r) - deviance(r) / 2

StatsAPI.nullloglikelihood(r::IPODRegression) = fullloglikelihood(r) - nulldeviance(r) / 2

StatsAPI.response(r::IPODRegression) = r.resp.y

StatsAPI.isfitted(r::IPODRegression) = r.fitted

StatsAPI.fitted(r::IPODRegression) = r.resp.μ

StatsAPI.residuals(r::IPODRegression) = r.resp.wrkres

StatsAPI.predict(r::IPODRegression) = fitted(r)

"""
    scale(m::RobustLinearModel, sqr::Bool=false)

The robust scale estimate used for the robust estimation.

If `sqr` is `true`, the square of the scale is returned.
"""
scale(r::IPODRegression, sqr::Bool=false) = sqr ? r.resp.σ^2 : r.resp.σ

StatsAPI.weights(r::IPODRegression) = r.wts

"""
    coef(m::IPODRegression)
The coefficients of the model.
"""
StatsAPI.coef(r::IPODRegression) = coef(r.pred)

"""
    nobs(obj::IPODRegression)::Integer
For linear and generalized linear models, returns the number of elements of the response.
For models with prior weights, return the number of non-zero weights.
"""
function StatsAPI.nobs(r::IPODRegression{T}) where {T}
    wts = weights(r)
    if !isempty(wts)
        ## Suppose that the weights are probability weights
        count(!iszero, wts)
    else
        length(response(r))
    end
end

"""
    wobs(obj::IPODRegression)
For unweighted linear models, equals to ``nobs``, it returns the number of elements of the response.
For models with prior weights, return the sum of the weights.
"""
function wobs(r::IPODRegression{T}) where {T}
    wts = weights(r)
    if !isempty(wts)
        ## Suppose that the weights are probability weights
        sum(wts)
    else
        oftype(sum(one(eltype(wts))), nobs(r))
    end
end

haspenalty(r::IPODRegression{T,R,P,V}) where {T,R,P<:AbstractRegularizedPred,V} = true

penalty(r::IPODRegression{T,R,P,V}) where {T,R,P<:AbstractRegularizedPred,V} = r.pred.penalty

"""
    outliers(r::IPODRegression)

Returns the vector of the outlier part γ of the response y, so that `(y - γ) | X ~ Normal`.
"""
outliers(r::IPODRegression) = outliers(r.resp)

function update_fields!(
    m::IPODRegression{T};
    wts::Union{Nothing,AbstractVector{<:Real}}=nothing,
    initial_scale::Union{Nothing,Symbol,Real}=nothing,
    σ0::Union{Nothing,Symbol,Real}=initial_scale,
    initial_coef::Union{Nothing,AbstractVector{<:Real}}=nothing,
    β0::Union{Nothing,AbstractVector{<:Real}}=initial_coef,
    initial_outliers::Union{Nothing,AbstractVector{<:Real}}=nothing,
    γ0::Union{Nothing,AbstractVector{<:Real}}=initial_outliers,
    precision::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
    λ0::Union{Nothing,Real,AbstractVector{<:Real}}=precision,
    kwargs...,
) where {T<:AbstractFloat}

    resp = m.resp
    pred = m.pred
    n = length(response(resp))
    p = length(coef(pred))

    if !isnothing(wts)
        if length(wts) in (0, n)
            copy!(m.wts, float(wts))
        else
            throw(ArgumentError("wts should be a vector of length 0 or $n: $(length(wts))"))
        end
    end
    if !isnothing(σ0)
        # convert to float
        if σ0 isa Symbol
            σ0 = initialscale(modelmatrix(m), response(m), weights(m), σ0)
        else
            σ0 = float(σ0)
        end

        if σ0 > 0
            resp.σ = σ0
        else
            throw(ArgumentError("σ0 should be a positive real: $(σ0))"))
        end
    end
    if !isnothing(γ0)
        if length(γ0) == n
            copy!(resp.outliers, float(γ0))
        else
            throw(ArgumentError("γ0 should be a vector of length $n: $(length(γ0))"))
        end
    end
    if !isnothing(λ0)
        if λ0 isa Real
            copy!(resp.precision, fill(float(λ0), n))
        elseif length(λ0) in (0, n)
            copy!(resp.precision, λ0)
        else
            throw(ArgumentError("λ0 should be a real or a vector of length 0 or $n: $(λ0))"))
        end
    end
    if !isnothing(β0)
        if length(β0) == p
            copy!(pred.beta0, float(β0))
        else
            throw(ArgumentError("β0 should be a vector of length $p: $(length(β0))"))
        end
    else
        fill!(pred.beta0, zero(eltype(coef(m))))
    end

    ## The predictor must be updated if wts, σ0 or β0 was changed

    # return the rest of the keyword arguments
    return kwargs
end


####################################
### Fit IPOD model
####################################

"""
    ipod(X, y, args...; kwargs...)

An alias for `fit(IPODRegression, X, y, loss, penalty; kwargs...)`.

The arguments `X` and `y` can be a `Matrix` and a `Vector` or a `Formula` and a `DataFrame`.
"""
ipod(X, y, args...; kwargs...) = fit(IPODRegression, X, y, args...; kwargs...)


"""
    fit(::Type{M},
        X::AbstractMatrix{T},
        y::AbstractVector{T},
        loss::LossFunction,
        penalty::Union{Nothing,PenaltyFunction}=nothing;
        method::Symbol = :auto,  # :chol, :cg, :cgd, :fista
        dofit::Bool = true,
        wts::FPVector = similar(y, 0),
        offset::FPVector = similar(y, 0),
        fitdispersion::Bool = false,
        initial_scale::Union{Symbol, Real}=:mad,
        σ0::Union{Symbol, Real}=initial_scale,
        initial_coef::AbstractVector=[],
        β0::AbstractVector=initial_coef,
        correct_leverage::Bool=false
        penalty_omit_intercept::Bool=true,
        fitargs...,
    ) where {M<:IPODRegression, T<:AbstractFloat}

Create a robust model with the model matrix (or formula) X and response vector (or dataframe) y,
using a robust estimator.


# Arguments

- `X`: the model matrix (it can be dense or sparse) or a formula
- `y`: the response vector or a table (dataframe, namedtuple, ...).
- `loss`: a robust loss function
- `penalty`: a penalty function, or `nothing` if the coefficients are not penalized.

# Keywords

- `method::Symbol = :auto`: the method used to solve the linear system,
    Without penalty:
    - Direct method, Cholesky decomposition: `:chol` (default)
    - Direct method, QR decomposition: `:qr`
    - Iterative method, Conjugate Gradient: `:cg`
    With penalty:
    - Coordinate Gradient Descent: `:cgd` (default)
    - Fast Iterative Shrinkage-Thresholding Algorithm: `:fista`
    - Alternating Minimization Algorithm: `:ama`
    - Alternating Direction Method of Multipliers: `:admm`
    Use :auto to select the default method based on whether penalty is or is not `nothing`.
- `dofit::Bool = true`: if false, return the model object without fitting;
- `dropmissing::Bool = false`: if true, drop the rows with missing values (and convert to
    Non-Missing type). With `dropmissing=true` the number of observations may be smaller
    than the size of the input arrays;
- `wts::Vector = similar(y, 0)`: Prior probability weights of observations.
    Can be empty (length 0) if no weights are used (default);
- `precision::Vector = similar(y, 0)`: Prior precision weights of observations for outlier
    detection. Can be empty (length 0) if all observations have the same precision (default);
- `fitdispersion::Bool = false`: reevaluate the dispersion;
- `contrasts::AbstractDict{Symbol,Any} = Dict{Symbol,Any}()`: a `Dict` mapping term names
    (as `Symbol`s) to term types (e.g. `ContinuousTerm`) or contrasts (e.g., `HelmertCoding()`,
    `SeqDiffCoding(; levels=["a", "b", "c"])`, etc.). If contrasts are not provided for a variable,
    the appropriate term type will be guessed based on the data type from the data column:
    any numeric data is assumed to be continuous, and any non-numeric data is assumed to be
    categorical (with `DummyCoding()` as the default contrast type);
- `initial_scale::Union{Symbol, Real}=:mad`: the initial scale estimate, for non-convex estimator
    it helps to find the global minimum.
    Automatic computation using `:mad`, `:L1` or `:extrema` (non-robust).
- `σ0::Union{Nothing, Symbol, Real}=initial_scale`: alias of `initial_scale`;
- `initial_coef::AbstractVector=[]`: the initial coefficients estimate, for non-convex estimator
    it helps to find the global minimum.
- `β0::AbstractVector=initial_coef`: alias of `initial_coef`;
- `penalty_omit_intercept::Bool=true`: if true, do not penalize the intercept,
    otherwise use the penalty on all the coefficients;
- `fitargs...`: other keyword arguments used to control the convergence of the IRLS algorithm
    (see [`pirls!`](@ref)).

# Output

the IPODRegression object.

"""
function StatsAPI.fit(
    ::Type{M},
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    loss::LossFunction,
    penalty::Union{Nothing,PenaltyFunction}=nothing;
    method::Symbol=:auto,  # :chol, :qr, :cg, :cgd, :fista, :ama, :admm
    dofit::Bool=true,
    dropmissing::Bool=false,  # placeholder
    initial_scale::Union{Nothing,Symbol,Real}=:mad,
    σ0::Union{Nothing,Symbol,Real}=initial_scale,
    wts::FPVector=similar(y, 0),
    precision::FPVector=similar(y, 0),
    fitdispersion::Bool=false,
    pivot::Bool=false,
    contrasts::AbstractDict{Symbol,Any}=Dict{Symbol,Any}(),  # placeholder
    __formula::Union{Nothing,FormulaTerm}=nothing,
    use_backtracking::Bool=false,
    A::AbstractMatrix{<:Real}=zeros(T, 0, 0),
    b::AbstractVector{<:Real}=zeros(T, 0),
    restart::Bool=true,
    adapt::Bool=true,
    penalty_omit_intercept::Bool=true,
    fitargs...,
) where {M<:IPODRegression,T<:AbstractFloat}

    m = IPODRegression(X, y, loss, penalty;
        method=method,
        wts=wts,
        precision=precision,
        fitdispersion=fitdispersion,
        pivot=pivot,
        formula=__formula,
        use_backtracking=use_backtracking,
        A=A,
        b=b,
        restart=restart,
        adapt=adapt,
        penalty_omit_intercept=penalty_omit_intercept,
    )

    # Update fields
    fitargs = update_fields!(m; σ0=σ0, fitargs...)

    return dofit ? fit!(m; fitargs...) : m
end

## Convert from formula-data to modelmatrix-response calling form
## the `fit` method must allow the `wts`, `precision`, `contrasts` and `__formula` keyword arguments
function StatsAPI.fit(
    ::Type{M},
    f::FormulaTerm,
    data,
    args...;
    dropmissing::Bool=false,
    wts::Union{Nothing,Symbol,FPVector}=nothing,
    precision::Union{Nothing,Symbol,FPVector}=nothing,
    contrasts::AbstractDict{Symbol,Any}=Dict{Symbol,Any}(),
    kwargs...,
) where {M<:IPODRegression}
    # Extract arrays from data using formula
    f, y, X, extra = modelframe(f, data, contrasts, dropmissing, M; wts=wts, precision=precision)
    # Call the `fit` method with arrays
    fit(M, X, y, args...;
        wts=extra.wts, precision=extra.precision, contrasts=contrasts, __formula=f, kwargs...)
end


"""
    ipod(X, y, l::LossFunction; λ=1)

She & Owen (2011) - Outlier Detection Using Nonconvex Penalized Regression
"""
function StatsAPI.fit!(
    m::IPODRegression{T,R,P,V};
    correct_leverage::Bool=false,
    updatescale::Bool=false,
    maxiter::Integer= (P <: FISTARegPred) ? 10_000 : 100,
    minstepfac::Real=1e-3,
    atol::Real=1e-8,
    rtol::Real=1e-7,
    verbose::Bool=false,
) where {T,R,P,V}
    # Return early if model has the fit flag set
    m.fitted && return m

    verbose && println("\nFit with IPOD model: $(m.resp.loss), $(penalty(m))")

    # TODO: check if it works and if it can work
    updatescale = false

    # Initialize the response and predictors
    init!(m; verbose=verbose)

    # convergence criteria
    γold = copy(m.resp.outliers)
    Δγ = 0

    devold = dev_criteria(m)
    dev = devold
    Δdev = 0

    ### Loop until convergence
    cvg = false
    for i in 1:maxiter
        ## Update outliers vector
        update_outliers!(m.resp)
        Δγ = outliers_criteria(m.resp, γold)

        ## Update coefficients
        setη!(m; verbose=verbose)
        dev = dev_criteria(m)
        Δdev = abs(dev - devold)

        # ## Update scale
         if updatescale
             update_scale!(m.resp)
         end

        ## Postupdate (only for ADMM)
        updatepred!(m.pred, scale(m); verbose=verbose, force=updatescale)

        # Test for convergence
        verbose && println("Iteration: $i, Δoutliers: $(Δγ), Δdev: $(Δdev)")
        tol = max(rtol * abs(devold), atol)
        if Δγ < atol && Δdev < tol
            cvg = true
            break
        end
        copyto!(γold, m.resp.outliers)
        @assert isfinite(dev)
        devold = dev
    end
    cvg || throw(ConvergenceException(maxiter))
    m.fitted = true
    return m
end

"""
    refit!(m::RobustLinearModel, [y::FPVector];
                                 wts::Union{Nothing, FPVector} = nothing,
                                 offset::Union{Nothing, FPVector} = nothing,
                                 quantile::Union{Nothing, AbstractFloat} = nothing,
                                 ridgeλ::Union{Nothing, Real} = nothing,
                                 kwargs...)

Refit the [`RobustLinearModel`](@ref).

This function assumes that `m` was correctly initialized and the model is refitted with
the new values for the response, weights, offset, quantile and ridge shrinkage.

Defining a new `quantile` is only possible for [`GeneralizedQuantileEstimator`](@ref).

Defining a new `ridgeλ` is only possible for [`RidgePred`](@ref) objects.
"""
function refit!(m::IPODRegression, y::FPVector; kwargs...)
    r = m.resp
    # Check that old and new y have the same number of observations
    if size(r.y, 1) != size(y, 1)
        mess = "the new response vector should have the same dimension: "*
               "$(size(r.y, 1)) != $(size(y, 1))"
        throw(DimensionMismatch(mess))
    end
    # Update y
    copyto!(r.y, y)

    refit!(m; kwargs...)
end

function refit!(
    m::IPODRegression;
    method=nothing,
    kwargs...,
)

    if !isnothing(method)
        @warn("the method cannot be changed when refitting,"*
              " ignore the keyword argument `method=:$(method)`."
        )
    end

    # Update fields
    kwargs = update_fields!(m; kwargs...)

    m.fitted = false
    fit!(m; kwargs...)
end


function dev_criteria(m::IPODRegression)
    if !haspenalty(m)
        # this criteria is not used with no penalty
        return 0
    end
    l = dev_criteria(m.resp)
    l += dev_criteria(m.pred)
    return l
end


function init!(
    m::IPODRegression{T};
    verbose::Bool=false,
) where {T<:AbstractFloat}

    resp = m.resp
    pred = m.pred

    # reset ∇β
    copyto!(pred.delbeta, pred.beta0)

    # res = y - Xβ
    mul!(resp.μ, pred.X, pred.beta0)
    initresp!(resp)

    # Initialize the predictor
    initpred!(pred, m.wts, resp.σ; verbose=verbose)

    m
end

function setη!(
    m::IPODRegression{T,R,P,V},
    linesearch_f::Real=1;
    verbose::Bool=false,
) where {T,R,P<:LinPred,V}
    μ = m.resp.μ
    # TODO: add line search

#    # dβ = Σ \ (X' * r)
#    mul!(dβ, wXt, p.wrkres)
#    ldiv!(facΣ, dβ)
    update_beta!(m.pred, m.resp.wrkres, m.wts)

    # Install beta
    broadcast!(+, m.pred.beta0, m.pred.beta0, m.pred.delbeta)

    # Update linear predictor
    mul!(μ, m.pred.X, m.pred.beta0)

    # Update residuals with the new μ
    update_residuals!(m.resp)
end

function setη!(
    m::IPODRegression{T,R,P,V},
    linesearch_f::Real=1;
    verbose::Bool=false,
) where {T,R,P<:AbstractRegularizedPred,V}
    wts = weights(m)

    μ = m.resp.μ
    σ2 = m.resp.σ^2
    wrky = m.resp.wrky

    if m.pred isa CGDRegPred
        # Update coefs and μ
        update_βμ!(m.pred, wrky, μ, σ2, wts; verbose=verbose)

    else
        # Update coefs
        update_beta!(m.pred, wrky, wts, σ2; verbose=verbose)

        # Update linear predictor
        mul!(μ, m.pred.X, m.pred.beta0)
    end

    # Update residuals with the new μ
    update_residuals!(m.resp)

    m
end

