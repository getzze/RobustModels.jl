

######
##    AbstractRobustModel methods
######

StatsAPI.islinear(m::AbstractRobustModel) = true

StatsAPI.dof(m::AbstractRobustModel) = length(coef(m))

StatsAPI.dof_residual(m::AbstractRobustModel) = wobs(m) - dof(m)

hasformula(m::AbstractRobustModel) = false

StatsModels.formula(m::AbstractRobustModel)::FormulaTerm =
    throw(ArgumentError("model was fitted without a formula"))

function StatsModels.hasintercept(m::AbstractRobustModel)
    return hasformula(m) ? hasintercept(formula(m)) : _hasintercept(modelmatrix(m))
end

function StatsModels.responsename(m::AbstractRobustModel)
    return !hasformula(m) ? "y" : coefnames(formula(m).lhs)
end

function StatsAPI.coefnames(m::AbstractRobustModel)
    if hasformula(m)
        return coefnames(formula(m).rhs)
    else
        return ["x$i" for i in 1:length(coef(m))]
    end
end

function StatsAPI.coeftable(m::AbstractRobustModel; level::Real=0.95)
    cc = coef(m)
    se = stderror(m)
    tt = cc ./ se
    ci = se * quantile(TDist(dof_residual(m)), (1 - level) / 2)
    p = ccdf.(Ref(FDist(1, dof_residual(m))), abs2.(tt))
    levstr = isinteger(level * 100) ? string(Integer(level * 100)) : string(level * 100)
    cn = coefnames(m)
    return CoefTable(
        hcat(cc, se, tt, p, cc + ci, cc - ci),
        ["Coef.", "Std. Error", "t", "Pr(>|t|)", "Lower $(levstr)%", "Upper $(levstr)%"],
        cn,
        4,
        3,
    )
end

function StatsAPI.confint(m::AbstractRobustModel; level::Real=0.95)
    alpha = quantile(TDist(dof_residual(m)), (1 - level) / 2)
    return hcat(coef(m), coef(m)) + stderror(m) * alpha * hcat(1.0, -1.0)
end

## TODO: specialize to make it faster
StatsAPI.leverage(m::AbstractRobustModel) = diag(projectionmatrix(m))

leverage_weights(m::AbstractRobustModel) = sqrt.(1 .- leverage(m))

## Convert to float, optionally drop rows with missing values (and convert to Non-Missing types)
function StatsAPI.fit(
    ::Type{M},
    X::Union{AbstractMatrix{T1},AbstractMatrix{M1}},
    y::Union{AbstractVector{T2},AbstractVector{M2}},
    args...;
    dropmissing::Bool=false,
    kwargs...,
) where {
    M<:AbstractRobustModel,
    T1<:Real,
    T2<:Real,
    M1<:Union{Missing,<:Real},
    M2<:Union{Missing,<:Real},
}
    X_ismissing = eltype(X) >: Missing
    y_ismissing = eltype(y) >: Missing
    if any([y_ismissing, X_ismissing])
        if !dropmissing
            msg = (
                "X and y eltypes need to be <:Real, if they have missing values use " *
                "`dropmissing=true`: typeof(X)=$(typeof(X)), typeof(y)=$(typeof(y))"
            )
            throw(ArgumentError(msg))
        end
        X, y, _ = missing_omit(X, y)
    end

    return fit(M, float(X), float(y), args...; kwargs...)
end

## Convert from formula-data to modelmatrix-response calling form
## the `fit` method must allow the `wts`, `contrasts` and `__formula` keyword arguments
## Specialize to allow other keyword arguments (offset, precision...) to be taken from
## a column of the dataframe.
function StatsAPI.fit(
    ::Type{M},
    f::FormulaTerm,
    data,
    args...;
    dropmissing::Bool=false,
    wts::Union{Nothing,Symbol,FPVector}=nothing,
    contrasts::AbstractDict{Symbol,Any}=Dict{Symbol,Any}(),
    kwargs...,
) where {M<:AbstractRobustModel}
    # Extract arrays from data using formula
    f, y, X, extra = modelframe(f, data, contrasts, dropmissing, M; wts=wts)
    # Call the `fit` method with arrays
    return fit(M, X, y, args...; wts=extra.wts, contrasts=contrasts, __formula=f, kwargs...)
end


######
##    RobustLinearModel methods
######

"""
    RobustLinearModel

Robust linear model representation

## Fields

* `resp`: the [`RobustLinResp`](@ref) structure.
* `pred`: the predictor structure, of type [`DensePredChol`](@ref), [`SparsePredChol`](@ref), [`DensePredCG`](@ref), [`SparsePredCG`](@ref) or [`RidgePred`](@ref).
* `formula`: either a `FormulaTerm` object or `nothing`
* `fitdispersion`: if true, the dispersion is estimated otherwise it is kept fixed
* `fitted`: if true, the model was already fitted
"""
mutable struct RobustLinearModel{
    T<:AbstractFloat,R<:RobustResp{T},L<:Union{LinPred,AbstractRegularizedPred{T}}
} <: AbstractRobustModel{T}
    resp::R
    pred::L
    formula::Union{FormulaTerm,Nothing}
    fitdispersion::Bool
    fitted::Bool
end

function Base.show(io::IO, obj::RobustLinearModel)
    msg = "Robust regression with $(Estimator(obj))\n\n"
    if hasformula(obj)
        msg *= "$(formula(obj))\n\n"
    end
    msg *= "Coefficients:\n"
    return println(io, msg, coeftable(obj))
end


hasformula(m::RobustLinearModel) = isnothing(m.formula) ? false : true

function StatsModels.formula(m::RobustLinearModel)
    if !hasformula(m)
        throw(ArgumentError("model was fitted without a formula"))
    end
    return m.formula
end

"""
    deviance(m::RobustLinearModel)

The sum of twice the loss/objective applied to the scaled residuals.

It is consistent with the definition of the deviance for OLS.
"""
StatsAPI.deviance(m::RobustLinearModel) = deviance(m.resp)

function StatsAPI.nulldeviance(m::RobustLinearModel)
    return nulldeviance(m.resp; intercept=hasintercept(m))
end

"""
    dispersion(m::RobustLinearModel, sqr::Bool=false)

The dispersion is the (weighted) sum of robust residuals. If `sqr` is true, return the squared dispersion.
"""
function GLM.dispersion(m::RobustLinearModel, sqr::Bool=false)
    return dispersion(m.resp, dof_residual(m), sqr)
end

"""
    coef(m::RobustLinearModel)
The coefficients of the model.
"""
StatsAPI.coef(m::RobustLinearModel) = coef(m.pred)

"""
    nobs(m::QuantileRegression)
For linear and generalized linear models, returns the number of elements of the response.
"""
StatsAPI.nobs(m::RobustLinearModel)::Integer = nobs(m.resp)

"""
    wobs(m::RobustLinearModel)
For unweighted linear models, equals to ``nobs``, it returns the number of elements of the response.
For models with prior weights, return the sum of the weights.
"""
wobs(m::RobustLinearModel) = wobs(m.resp)

"""
    Estimator(m::RobustLinearModel)

The robust estimator object used to fit the model.
"""
Estimator(m::RobustLinearModel) = Estimator(m.resp)

function StatsAPI.stderror(m::RobustLinearModel{T,R,P}) where {T,R,P<:LinPred}
    return location_variance(m.resp, dof_residual(m), false) .* sqrt.(diag(vcov(m)))
end

function StatsAPI.stderror(
    m::RobustLinearModel{T,R,P}
) where {T,R,P<:AbstractRegularizedPred}
    return location_variance(m.resp, dof_residual(m), false) .* sqrt.(diag(vcov(m)))
end

StatsAPI.loglikelihood(m::RobustLinearModel) = loglikelihood(m.resp)

function StatsAPI.nullloglikelihood(m::RobustLinearModel)
    return nullloglikelihood(m.resp; intercept=hasintercept(m))
end

StatsAPI.weights(m::RobustLinearModel) = weights(m.resp)

"""
    workingweights(m::RobustLinearModel)

The robust weights computed by the model.

This can be used to detect outliers, as outliers weights are lower than the
weights of valid data points.
"""
workingweights(m::RobustLinearModel) = workingweights(m.resp)

StatsAPI.response(m::RobustLinearModel) = response(m.resp)

StatsAPI.isfitted(m::RobustLinearModel) = m.fitted

StatsAPI.fitted(m::RobustLinearModel) = fitted(m.resp)

StatsAPI.residuals(m::RobustLinearModel) = residuals(m.resp)

"""
    scale(m::RobustLinearModel, sqr::Bool=false)

The robust scale estimate used for the robust estimation.

If `sqr` is `true`, the square of the scale is returned.
"""
scale(m::RobustLinearModel, sqr::Bool=false) = scale(m.resp, sqr)

"""
    tauscale(m::RobustLinearModel, sqr::Bool=false; kwargs...)

The robust τ-scale that is minimized in τ-estimation.

If `sqr` is `true`, the square of the τ-scale is returned.
"""
tauscale(m::RobustLinearModel, args...; kwargs...) = tauscale(m.resp, args...; kwargs...)

StatsAPI.modelmatrix(m::RobustLinearModel) = modelmatrix(m.pred)

StatsAPI.vcov(m::RobustLinearModel) = vcov(m.pred, workingweights(m.resp))

"""
    projectionmatrix(m::RobustLinearModel)

The robust projection matrix from the predictor: X (X' W X)⁻¹ X' W,
where W are the working weights.
"""
projectionmatrix(m::RobustLinearModel) = projectionmatrix(m.pred, workingweights(m.resp))

function check_variant(variant::Symbol, allowed::Tuple)
    if !(variant in allowed)
        msg = "`variant` argument can only take a value in $(allowed): $(variant)"
        throw(ArgumentError(msg))
    end
end

for fun in (:vcov, :projectionmatrix, :leverage, :leverage_weights)
    @eval begin
        @doc """
            $($(fun))(m::RobustLinearModel, variant::Symbol)

        Returns `$($(fun))` for the model using a different weights vector depending on the variant:
            - `variant = :original`: use the user-defined weights, if no weights were used
                (size of the weights vector is 0), no weights are used.
            - `variant = :fitted`: use the working weights of the fitted model from the IRLS
                procedure.
        """
        function $(fun)(m::RobustLinearModel, variant::Symbol)
            check_variant(variant, (:original, :fitted))
            if variant == :original
                w = weights(m)
            else  # variant == :fitted
                w = workingweights(m.resp)
            end
            return $(fun)(m.pred, w)
        end
    end
end

function StatsAPI.predict(
    m::RobustLinearModel, newX::AbstractMatrix; offset::FPVector=eltype(newX)[]
)
    mu = newX * coef(m)
    if !isempty(m.resp.offset)
        if !(length(offset) == size(newX, 1))
            mess =
                "fit with offset, so `offset` keyword argument" *
                " must be an offset of length `size(newX, 1)`"
            throw(ArgumentError(mess))
        end
        broadcast!(+, mu, mu, offset)
    else
        if length(offset) > 0
            mess = "fit without offset, so value of `offset` kwarg does not make sense"
            throw(ArgumentError(mess))
        end
    end
    return mu
end
StatsAPI.predict(m::RobustLinearModel) = fitted(m)

### With RidgePred

function StatsAPI.dof(m::RobustLinearModel{T,R,P}) where {T,R,P<:RidgePred}
    return tr(projectionmatrix(m.pred, workingweights(m.resp)))
end

function StatsAPI.stderror(m::RobustLinearModel{T,R,P}) where {T,R,P<:RidgePred}
    wXt = (workingweights(m.resp) .* modelmatrix(m.pred))'
    Σ = Hermitian(wXt * modelmatrix(m.pred))
    M = vcov(m) * Σ * vcov(m)'
    s = location_variance(m.resp, dof_residual(m), false)
    return s .* sqrt.(diag(M))
end

########################################################################
## RobustLinearModel fit methods
########################################################################


function update_fields!(
    m::RobustLinearModel{T};
    wts::Union{Nothing,AbstractVector{<:Real}}=nothing,
    initial_scale::Union{Nothing,Symbol,Real}=nothing,
    σ0::Union{Nothing,Symbol,Real}=initial_scale,
    initial_coef::Union{Nothing,AbstractVector{<:Real}}=nothing,
    β0::Union{Nothing,AbstractVector{<:Real}}=initial_coef,
    offset::Union{Nothing,AbstractVector{<:Real}}=nothing,
    kwargs...,
) where {T<:AbstractFloat}

    resp = m.resp
    pred = m.pred
    n = length(response(resp))
    p = length(coef(pred))

    if !isnothing(wts)
        if length(wts) in (0, n)
            copy!(resp.wts, float(wts))
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
    if !isnothing(offset)
        if length(offset) in (0, n)
            copy!(resp.offset, offset)
        else
            throw(
                ArgumentError("λ0 should be a vector of length 0 or $n: $(length(offset))")
            )
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


"""
    rlm(X, y, args...; kwargs...)

An alias for `fit(RobustLinearModel, X, y, est; kwargs...)`.

The arguments `X` and `y` can be a `Matrix` and a `Vector` or a `Formula` and a `DataFrame`.
"""
rlm(X, y, args...; kwargs...) = fit(RobustLinearModel, X, y, args...; kwargs...)


"""
    fit(::Type{M},
        X::Union{AbstractMatrix{T},SparseMatrixCSC{T}},
        y::AbstractVector{T},
        est::Estimator;
        method::Symbol       = :auto,  # :chol, :qr, :cg
        dofit::Bool          = true,
        wts::FPVector        = similar(y, 0),
        offset::FPVector     = similar(y, 0),
        fitdispersion::Bool  = false,
        ridgeλ::Real         = 0,
        ridgeG::Union{UniformScaling, AbstractArray} = I,
        βprior::AbstractVector = [],
        quantile::Union{Nothing, AbstractFloat} = nothing,
        pivot::Bool = false,
        initial_scale::Union{Symbol, Real}=:mad,
        σ0::Union{Nothing, Symbol, Real}=initial_scale,
        initial_coef::AbstractVector=[],
        β0::AbstractVector=initial_coef,
        correct_leverage::Bool=false,
        fitargs...,
    ) where {M<:RobustLinearModel, T<:AbstractFloat}

Create a robust model with the model matrix (or formula) X and response vector (or dataframe) y,
using a robust estimator.


# Arguments

- `X`: the model matrix (it can be dense or sparse) or a formula
- `y`: the response vector or a table (dataframe, namedtuple, ...).
- `est`: a robust estimator

# Keywords

- `method::Symbol = :auto`: the method to use for solving the weighted linear system,
    `chol` (default), `qr` or `cg`. Use :auto to select the default method;
- `dofit::Bool = true`: if false, return the model object without fitting;
- `dropmissing::Bool = false`: if true, drop the rows with missing values (and convert to
    Non-Missing type). With `dropmissing=true` the number of observations may be smaller than
    the size of the input arrays;
- `wts::Vector = similar(y, 0)`: Prior probability weights of observations.
    Can be empty (length 0) if no weights are used (default);
- `offset::Vector = similar(y, 0)`: an offset vector, should be empty if no offset is used;
- `fitdispersion::Bool = false`: reevaluate the dispersion;
- `ridgeλ::Real = 0`: if positive, perform a robust ridge regression with shrinkage
    parameter `ridgeλ`. [`RidgePred`](@ref) object will be used;
- `ridgeG::Union{UniformScaling, AbstractArray} = I`: define a custom regularization matrix.
    Default to unity matrix (with 0 for the intercept);
- `βprior::AbstractVector = []`: define a custom prior for the coefficients for ridge regression.
    Default to `zeros(p)`;
- `quantile::Union{Nothing, AbstractFloat} = nothing`:
    only for [`GeneralizedQuantileEstimator`](@ref), define the quantile to estimate;
- `pivot::Bool=false`: use pivoted factorization;
- `contrasts::AbstractDict{Symbol,Any} = Dict{Symbol,Any}()`: a `Dict` mapping term names
    (as `Symbol`s) to term types (e.g. `ContinuousTerm`) or contrasts (e.g., `HelmertCoding()`,
    `SeqDiffCoding(; levels=["a", "b", "c"])`, etc.). If contrasts are not provided for a variable,
    the appropriate term type will be guessed based on the data type from the data column:
    any numeric data is assumed to be continuous, and any non-numeric data is assumed to be
    categorical (with `DummyCoding()` as the default contrast type);
- `initial_scale::Union{Symbol, Real}=:mad`: the initial scale estimate, for non-convex estimator
    it helps to find the global minimum. Automatic computation using `:mad`, `L1` or
    `extrema` (non-robust).
- `σ0::Union{Nothing, Symbol, Real}=initial_scale`: alias of `initial_scale`;
- `initial_coef::AbstractVector=[]`: the initial coefficients estimate, for non-convex estimator
    it helps to find the global minimum.
- `β0::AbstractVector=initial_coef`: alias of `initial_coef`;
- `correct_leverage::Bool=false`: apply the leverage correction weights with
    [`leverage_weights`](@ref).
- `fitargs...`: other keyword arguments used to control the convergence of the IRLS algorithm
    (see [`pirls!`](@ref)).

# Output

the RobustLinearModel object.

"""
function StatsAPI.fit(
    ::Type{M},
    X::Union{AbstractMatrix{T},SparseMatrixCSC{T}},
    y::AbstractVector{T},
    est::AbstractMEstimator;
    method::Symbol=:chol,  # :qr, :cg
    dofit::Bool=true,
    dropmissing::Bool=false,  # placeholder
    initial_scale::Union{Nothing,Symbol,Real}=:mad,
    σ0::Union{Nothing,Symbol,Real}=initial_scale,
    wts::FPVector=similar(y, 0),
    offset::FPVector=similar(y, 0),
    fitdispersion::Bool=false,
    ridgeλ::Real=0,
    ridgeG::Union{UniformScaling,AbstractArray{<:Real}}=I,
    βprior::AbstractVector{<:Real}=T[],
    quantile::Union{Nothing,AbstractFloat}=nothing,
    pivot::Bool=false,
    contrasts::AbstractDict{Symbol,Any}=Dict{Symbol,Any}(),  # placeholder
    __formula::Union{Nothing,FormulaTerm}=nothing,
    fitargs...,
) where {M<:RobustLinearModel,T<:AbstractFloat}

    # Check that X and y have the same number of observations
    n, p = size(X)
    if n != size(y, 1)
        throw(DimensionMismatch("number of rows in X and y must match"))
    end

    # Change quantile
    if !isnothing(quantile)
        if !isa(est, AbstractQuantileEstimator)
            throw(
                TypeError(
                    :fit,
                    "arguments, quantile cannot be changed for this type",
                    AbstractQuantileEstimator,
                    est,
                ),
            )
        end
        est.quantile = quantile
    end

    # Response object
    rr = RobustLinResp(est, y, offset, wts)

    # Predictor object
    methods = (:auto, :chol, :cg, :qr)
    if method ∉ methods
        @warn("Incorrect method `:$(method)`, should be one of $(methods)")
        method = :auto
    end

    pp = if ridgeλ > 0
        # With ridge regularization
        G = if isa(ridgeG, UniformScaling)
            # Has an intersect
            intercept_col = get_intercept_col(X, __formula)
            if !isnothing(intercept_col)
                spdiagm(0 => [float(i != intercept_col) for i in 1:p])
            else
                I(p)
            end
        else
            ridgeG
        end
        if method == :cg
            cgpred(X, float(ridgeλ), G, βprior, pivot)
        elseif method == :qr
            qrpred(X, float(ridgeλ), G, βprior, pivot)
        elseif method in (:chol, :auto)
            cholpred(X, float(ridgeλ), G, βprior, pivot)
        else
            error("method :$method is not allowed, should be in: $methods")
        end
    else
        # No regularization
        if method == :cg
            cgpred(X, pivot)
        elseif method == :qr
            qrpred(X, pivot)
        elseif method in (:chol, :auto)
            cholpred(X, pivot)
        else
            error("method :$method is not allowed, should be in: $methods")
        end
    end

    m = RobustLinearModel(rr, pp, __formula, fitdispersion, false)

    # Update fields
    fitargs = update_fields!(m; σ0=σ0, fitargs...)

    return dofit ? fit!(m; fitargs...) : m
end

## Convert from formula-data to modelmatrix-response calling form
## the `fit` method must allow the `wts`, `offset`, `contrasts` and `__formula` keyword arguments
function StatsAPI.fit(
    ::Type{M},
    f::FormulaTerm,
    data,
    args...;
    dropmissing::Bool=false,
    wts::Union{Nothing,Symbol,FPVector}=nothing,
    offset::Union{Nothing,Symbol,FPVector}=nothing,
    contrasts::AbstractDict{Symbol,Any}=Dict{Symbol,Any}(),
    kwargs...,
) where {M<:RobustLinearModel}
    # Extract arrays from data using formula
    f, y, X, extra = modelframe(f, data, contrasts, dropmissing, M; wts=wts, offset=offset)
    # Call the `fit` method with arrays
    return fit(
        M,
        X,
        y,
        args...;
        wts=extra.wts,
        offset=extra.offset,
        contrasts=contrasts,
        __formula=f,
        kwargs...,
    )
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
function refit!(m::RobustLinearModel, y::FPVector; kwargs...)
    r = m.resp
    # Check that old and new y have the same number of observations
    if size(r.y, 1) != size(y, 1)
        mess = (
            "the new response vector should have the same dimension: " *
            "$(size(r.y, 1)) != $(size(y, 1))"
        )
        throw(DimensionMismatch(mess))
    end
    # Update y
    copyto!(r.y, y)

    return refit!(m; kwargs...)
end

function refit!(
    m::RobustLinearModel{T};
    method=nothing,
    quantile::Union{Nothing,AbstractFloat}=nothing,
    ridgeλ::Union{Nothing,Real}=nothing,
    kwargs...,
) where {T}

    if !isnothing(method)
        @warn(
            "the method cannot be changed when refitting, " *
            "ignore the keyword argument `method=:$(method)`."
        )
    end

    r = m.resp
    n = length(r.y)

    # Update quantile, if the estimator is AbstractQuantileEstimator
    if !isnothing(quantile)
        isa(r.est, AbstractQuantileEstimator) || throw(
            TypeError(
                :refit!,
                "arguments, quantile can be changed only if isa(r.est, AbstractQuantileEstimator)",
                AbstractQuantileEstimator,
                r.est,
            ),
        )
        r.est.quantile = quantile
    end

    # Update ridge shrinkage parameter
    if !isnothing(ridgeλ)
        isa(m.pred, RidgePred) || throw(
            TypeError(
                :refit!,
                "arguments, ridgeλ can be changed only if the predictor is a RidgePred",
                RidgePred,
                m.pred,
            ),
        )
        m.pred.λ = float(ridgeλ)
    end

    # Update fields, last thing to do because it resets β0
    kwargs = update_fields!(m; kwargs...)

    # Reinitialize the response
    initresp!(r)

    m.fitted = false
    return fit!(m; kwargs...)
end


"""
    fit!(m::RobustLinearModel; initial_scale::Union{Symbol, Real}=:mad,
              σ0::Union{Nothing, Symbol, Real}=initial_scale,
              initial_coef::AbstractVector=[],
              β0::AbstractVector=initial_coef,
              correct_leverage::Bool=false, kwargs...)

Optimize the objective of a `RobustLinearModel`.  When `verbose` is `true` the values of the
objective and the parameters are printed on stdout at each iteration.

This function assumes that `m` was correctly initialized.

This function returns early if the model was already fitted, instead call `refit!`.
"""
function StatsAPI.fit!(
    m::RobustLinearModel{T,R,P}; correct_leverage::Bool=false, kwargs...
) where {T,R,P<:LinPred}

    # Return early if model has the fit flag set
    m.fitted && return m

    # Compute the initial values
    σ0, β0 = scale(m), coef(m)

    if correct_leverage
        wts = m.resp.wts
        copy!(wts, leverage_weights(m, :original))
    end

    # Get type
    V = typeof(m.resp.est)

    _fit!(m, V; σ0=σ0, β0=β0, kwargs...)

    m.fitted = true
    return m
end

## Error message
function _fit!(m::RobustLinearModel, ::Type{E}; kwargs...) where {E<:AbstractMEstimator}
    allowed_estimators = (
        MEstimator, SEstimator, MMEstimator, TauEstimator, GeneralizedQuantileEstimator
    )
    mess = (
        "only types $(allowed_estimators) are allowed, " *
        "you must define the `_fit!` method for the type: $(E)"
    )
    return error(mess)
end

# Fit M-estimator
function _fit!(
    m::RobustLinearModel,
    ::Type{E};
    σ0::AbstractFloat=1.0,
    β0::AbstractVector{<:Real}=Float64[],
    verbose::Bool=false,
    kwargs...,
) where {E<:MEstimator}

    verbose && println("\nFit with M-estimator: $(Estimator(m))")
    ## Minimize the objective
    pirls!(m; sigma0=σ0, beta0=β0, verbose=verbose, kwargs...)

    ## TODO: update scale is fitdispersion is true
    return m
end

# Fit Generalized M-Quantile estimator
function _fit!(
    m::RobustLinearModel,
    ::Type{E};
    σ0::AbstractFloat=1.0,
    β0::AbstractVector{<:Real}=Float64[],
    verbose::Bool=false,
    kwargs...,
) where {E<:GeneralizedQuantileEstimator}

    verbose && println("\nFit with M-Quantile-estimator: $(Estimator(m))")
    ## Minimize the objective
    pirls!(m; sigma0=σ0, beta0=β0, verbose=verbose, kwargs...)

    ## TODO: update scale if fitdispersion is true
    return m
end

# Fit S-estimator
function _fit!(
    m::RobustLinearModel,
    ::Type{E};
    σ0::AbstractFloat=1.0,
    β0::AbstractVector{<:Real}=Float64[],
    verbose::Bool=false,
    resample::Bool=false,
    resampling_options::Dict{Symbol,F}=Dict{Symbol,Any}(:verbose => verbose),
    kwargs...,
) where {F,E<:SEstimator}

    ## Resampling algorithm to find a starting point close to the global minimum
    if resample
        σ0, β0 = resampling_best_estimate(m, E; resampling_options...)
    end

    verbose && println("\nFit with S-estimator: $(Estimator(m))")
    ## Minimize the objective
    pirls_Sestimate!(m; sigma0=σ0, beta0=β0, verbose=verbose, kwargs...)

    # Set the `fitdispersion` flag to true, because σ was estimated
    m.fitdispersion = true

    return m
end

# Fit MM-estimator
function _fit!(
    m::RobustLinearModel,
    ::Type{E};
    σ0::AbstractFloat=1.0,
    β0::AbstractVector{<:Real}=Float64[],
    verbose::Bool=false,
    resample::Bool=false,
    resampling_options::Dict{Symbol,F}=Dict{Symbol,Any}(:verbose => verbose),
    kwargs...,
) where {F,E<:MMEstimator}

    ## Set the S-Estimator for robust estimation of σ and β0
    set_SEstimator(Estimator(m.resp))

    ## Resampling algorithm to find a starting point close to the global minimum
    if resample
        σ0, β0 = resampling_best_estimate(m, E; resampling_options...)
    end

    verbose && println("\nFit with MM-estimator - 1. S-estimator: $(Estimator(m.resp))")
    ## Minimize the objective
    pirls_Sestimate!(m; sigma0=σ0, beta0=β0, verbose=verbose, kwargs...)

    ## Use an M-estimate to estimate coefficients
    β0 = coef(m)
    σ0 = scale(m)

    ## Set the M-Estimator for efficient estimation of β
    set_MEstimator(Estimator(m.resp))

    verbose && println("\nFit with MM-estimator - 2. M-estimator: $(Estimator(m.resp))")
    ## Minimize the objective
    pirls!(m; sigma0=σ0, beta0=β0, verbose=verbose, kwargs...)

    # Set the `fitdispersion` flag to true, because σ was estimated
    m.fitdispersion = true

    return m
end

# Fit τ-estimator
function _fit!(
    m::RobustLinearModel,
    ::Type{E};
    σ0::AbstractFloat=1.0,
    β0::AbstractVector{<:Real}=Float64[],
    verbose::Bool=false,
    resample::Bool=false,
    resampling_options::Dict{Symbol,F}=Dict{Symbol,Any}(:verbose => verbose),
    kwargs...,
) where {F,E<:TauEstimator}

    ## Resampling algorithm to find a starting point close to the global minimum
    if resample
        σ0, β0 = resampling_best_estimate(m, E; resampling_options...)
    end

    verbose && println("\nFit with τ-estimator: $(Estimator(m))")
    ## Minimize the objective
    pirls_τestimate!(m; sigma0=σ0, beta0=β0, verbose=verbose, kwargs...)

    # Set the `fitdispersion` flag to true, because σ was estimated
    m.fitdispersion = true

    return m
end

function setβ0!(
    m::RobustLinearModel{T}, β0::AbstractVector{<:Real}=T[]
) where {T<:AbstractFloat}
    r = m.resp
    p = m.pred

    initresp!(r)
    if isempty(β0)
        # Compute beta0 from solving the least square with the response value r.y
        delbeta!(p, r.wrkres, r.wrkwt)
        installbeta!(p)
    else
        copyto!(p.beta0, float(β0))
        fill!(p.delbeta, 0)
    end

    return m
end

"""
    setinitη!(m)
Compute the predictor using the initial value of β0 and compute the residuals
"""
function setinitη!(m::RobustLinearModel{T}) where {T}
    r = m.resp
    p = m.pred

    ## Initially, β0 is defined but not ∇β, so use f=0
    linpred!(r.μ, p, 0)
    updateres!(r; updatescale=false)

    return m
end

"""
    setinitσ!(m)
Compute the predictor scale using the MAD of the residuals
Use only for rough estimate, like in the resampling phase.
"""
function setinitσ!(m::RobustLinearModel; kwargs...)
    m.resp.σ = madresidualscale(m.resp; kwargs...)
    return m
end

"""
    setη!(m, f=1.0; updatescale=false, kwargs...)
Compute the ∇β using the current residuals and working weights (only if f=1,
which corresponds to the first iteration of linesearch), then compute
the predictor using the ∇β value and compute the new residuals and deviance.
The scaletype argument defines if the location or scale loss function should be used
If updatescale is true, the scale is also updated along with the residuals.
"""
function setη!(
    m::RobustLinearModel{T,R,P}, f::T=1.0; updatescale::Bool=false, kwargs...
) where {T,R,P<:LinPred}
    r = m.resp
    p = m.pred

    # First update of linesearch algorithm, compute ∇β
    if f == 1
        delbeta!(p, r.wrkres, r.wrkwt)
    end
    # Compute and set the predictor η from β0 and ∇β
    linpred!(r.η, p, f)

    # Update the residuals and weights (and scale if updatescale=true)
    updateres!(r; updatescale=updatescale, kwargs...)
    return m
end



"""
    pirls!(m::RobustLinearModel{T}; verbose::Bool=false, maxiter::Integer=30,
           minstepfac::Real=1e-3, atol::Real=1e-6, rtol::Real=1e-6,
           beta0::AbstractVector=[], sigma0::Union{Nothing, T}=nothing)

(Penalized) Iteratively Reweighted Least Square procedure for M-estimation.
The Penalized aspect is not implemented (yet).
"""
function pirls!(
    m::RobustLinearModel{T};
    verbose::Bool=false,
    maxiter::Integer=30,
    minstepfac::Real=1e-3,
    atol::Real=1e-6,
    rtol::Real=1e-5,
    beta0::AbstractVector=[],
    sigma0::Union{Nothing,T}=nothing,
) where {T<:AbstractFloat}

    # Check arguments
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive"))
    0 < minstepfac < 1 || throw(ArgumentError("minstepfac must be in (0, 1)"))

    # Extract fields and set convergence flag
    cvg, p, r = false, m.pred, m.resp

    ## Initialize σ, default to do not change
    if !isnothing(sigma0)
        r.σ = sigma0
    end

    ## Initialize β or set it to the provided values
    setβ0!(m, beta0)

    # Initialize μ and compute residuals
    setinitη!(m)

    # If σ==0, iterations will fail, so return here
    if iszero(r.σ)
        verbose && println("Initial scale is 0.0, no iterations performed.")
        return m
    end

    # Compute initial deviance
    devold = deviance(m)
    absdev = abs(devold)
    dev = devold
    Δdev = 0

    verbose && println("initial deviance: $(@sprintf("%.4g", devold))")
    for i in 1:maxiter
        f = 1.0 # line search factor
        # local dev
        absdev = abs(devold)

        # Compute the change to β, update μ and compute deviance
        dev = try
            deviance(setη!(m; updatescale=false))
        catch e
            isa(e, DomainError) ? Inf : rethrow(e)
        end

        # Assert the deviance is positive (up to rounding error)
        @assert dev > -atol

        verbose && println(
            "deviance at step $i: $(@sprintf("%.4g", dev)), crit=$((devold - dev)/absdev)",
        )

        # Line search
        ## If the deviance isn't declining then half the step size
        ## The rtol*abs(devold) term is to avoid failure when deviance
        ## is unchanged except for rounding errors.
        while dev > devold + rtol * absdev
            f /= 2
            f > minstepfac ||
                error("linesearch failed at iteration $(i) with beta0 = $(p.beta0)")

            dev = try
                # Update μ and compute deviance with new f. Do not recompute ∇β
                deviance(setη!(m, f))
            catch e
                isa(e, DomainError) ? Inf : rethrow(e)
            end
        end
        installbeta!(p, f)

        # Test for convergence
        Δdev = (devold - dev)
        verbose && println("Iteration: $i, deviance: $dev, Δdev: $(Δdev)")
        tol = max(rtol * absdev, atol)
        if -tol < Δdev < tol || dev < atol
            cvg = true
            break
        end
        @assert isfinite(dev)
        devold = dev
    end
    cvg || throw(ConvergenceException(maxiter))
    return m
end


"""
    pirls_Sestimate!(m::RobustLinearModel{T}; verbose::Bool=false, maxiter::Integer=30,
           minstepfac::Real=1e-3, atol::Real=1e-6, rtol::Real=1e-6,
           beta0::AbstractVector=T[], sigma0::Union{Nothing, T}=nothing)

(Penalized) Iteratively Reweighted Least Square procedure for S-estimation.
The Penalized aspect is not implemented (yet).
"""
function pirls_Sestimate!(
    m::RobustLinearModel{T};
    verbose::Bool=false,
    maxiter::Integer=30,
    minstepfac::Real=1e-3,
    atol::Real=1e-6,
    rtol::Real=1e-5,
    miniter::Integer=2,
    beta0::AbstractVector{<:Real}=T[],
    sigma0::Union{Nothing,T}=nothing,
) where {T<:AbstractFloat}

    # Check arguments
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive"))
    0 < minstepfac < 1 || throw(ArgumentError("minstepfac must be in (0, 1)"))

    # Extract fields and set convergence flag
    cvg, p, r = false, m.pred, m.resp

    ## Initialize σ, default to largest value
    maxσ = (maximum(r.y) - minimum(r.y)) / 2
    verbose && println("maximum scale: $(@sprintf("%.4g", maxσ))")
    if !isnothing(sigma0)
        r.σ = sigma0
    else
        r.σ = maxσ
    end
    verbose && println("initial scale: $(@sprintf("%.4g", r.σ))")

    ## Initialize β or set it to the provided values
    setβ0!(m, beta0)

    # Initialize μ and compute residuals
    setinitη!(m)

    # Compute initial scale
    sigold = scale(
        setη!(m; updatescale=true, verbose=verbose, sigma0=sigma0, fallback=maxσ)
    )
    installbeta!(p, 1)
    r.σ = sigold

    verbose && println("initial iteration scale: $(@sprintf("%.4g", sigold))")
    for i in 1:maxiter
        f = 1.0 # line search factor
        local sig

        # Compute the change to β, update μ and compute deviance
        sig = scale(
            setη!(m; updatescale=true, verbose=verbose, sigma0=sigold, fallback=maxσ)
        )

        # Assert the deviance is positive (up to rounding error)
        @assert sig > -atol

        verbose && println(
            "scale at step $i: $(@sprintf("%.4g", sig)), crit=$((sigold - sig)/sigold)"
        )

        # Line search
        ## If the scale isn't declining then half the step size
        ## The rtol*abs(sigold) term is to avoid failure when scale
        ## is unchanged except for rounding errors.
        linesearch_failed = false
        while sig > sigold * (1 + rtol)
            f /= 2
            if f <= minstepfac
                if i <= miniter
                    linesearch_failed = true
                    break
                else
                    error("linesearch failed at iteration $(i) with beta0 = $(p.beta0)")
                end
            end
            # Update μ and compute deviance with new f. Do not recompute ∇β
            sig = scale(
                setη!(m, f; updatescale=true, verbose=verbose, sigma0=sigold, fallback=maxσ)
            )
        end

        # Reset initial scale
        if linesearch_failed
            # Allow scale to increase in early iterations
            sigold = max(maxσ, sig)
            r.σ = sigold
            verbose && println(
                "linesearch failed at early iteration $(i), set scale to maximum value: $(sigold)",
            )
            # Skip test for convergence
            continue
        end

        # Update coefficients
        installbeta!(p, f)
        r.σ = sig

        # Test for convergence
        Δsig = (sigold - sig)
        verbose && println("Iteration: $i, scale: $sig, Δsig: $(Δsig)")
        tol = max(rtol * sigold, atol)
        if -tol < Δsig < tol || sig < atol
            cvg = true
            break
        end
        @assert isfinite(sig) && !iszero(sig)
        sigold = sig
    end
    cvg || throw(ConvergenceException(maxiter))
    return m
end


"""
    pirls_τestimate!(m::RobustLinearModel{T}; verbose::Bool=false, maxiter::Integer=30,
           minstepfac::Real=1e-3, atol::Real=1e-6, rtol::Real=1e-6,
           beta0::AbstractVector=T[], sigma0::Union{Nothing, T}=nothing)

(Penalized) Iteratively Reweighted Least Square procedure for τ-estimation.
The Penalized aspect is not implemented (yet).
"""
function pirls_τestimate!(
    m::RobustLinearModel{T};
    verbose::Bool=false,
    maxiter::Integer=30,
    minstepfac::Real=1e-3,
    atol::Real=1e-6,
    rtol::Real=1e-5,
    miniter::Integer=2,
    beta0::AbstractVector{<:Real}=T[],
    sigma0::Union{Nothing,T}=nothing,
) where {T<:AbstractFloat}

    # Check arguments
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive"))
    0 < minstepfac < 1 || throw(ArgumentError("minstepfac must be in (0, 1)"))

    # Extract fields and set convergence flag
    cvg, p, r = false, m.pred, m.resp

    ## Initialize σ, default to largest value
    maxσ = (maximum(r.y) - minimum(r.y)) / 2
    verbose && println("maximum scale: $(@sprintf("%.4g", maxσ))")
    if !isnothing(sigma0)
        r.σ = sigma0
    else
        r.σ = maxσ
    end
    verbose && println("initial scale: $(@sprintf("%.4g", r.σ))")

    ## Initialize β or set it to the provided values
    setβ0!(m, beta0)

    # Initialize μ and compute residuals
    setinitη!(m)

    # Compute initial τ-scale
    tauold = tauscale(setη!(m; updatescale=true); verbose=verbose)
    installbeta!(p, 1)

    verbose && println("initial iteration τ-scale: $(@sprintf("%.4g", tauold))")
    for i in 1:maxiter
        f = 1.0 # line search factor
        local tau

        # Compute the change to β, update μ and compute deviance
        tau = tauscale(
            setη!(m; updatescale=true, verbose=verbose, fallback=maxσ); verbose=verbose
        )

        # Assert the deviance is positive (up to rounding error)
        @assert tau > -atol

        verbose && println(
            "scale at step $i: $(@sprintf("%.4g", tau)), crit=$((tauold - tau)/tauold)"
        )

        # Line search
        ## If the scale isn't declining then half the step size
        ## The rtol*abs(sigold) term is to avoid failure when scale
        ## is unchanged except for rounding errors.
        linesearch_failed = false
        while tau > tauold + rtol * tau
            f /= 2
            if f <= minstepfac
                if i <= miniter
                    linesearch_failed = true
                    break
                else
                    error("linesearch failed at iteration $(i) with beta0 = $(p.beta0)")
                end
            end

            # Update μ and compute deviance with new f. Do not recompute ∇β
            tau = tauscale(setη!(m, f; updatescale=true))
        end

        if linesearch_failed
            # Allow scale to increase in early iterations
            tauold = max(maxσ, tau)
            r.σ = tauold
            verbose && println(
                "linesearch failed at early iteration $(i), set scale to maximum value: $(tauold)",
            )
            # Skip test for convergence
            continue
        end

        # Update coefficients
        installbeta!(p, f)

        # Test for convergence
        Δtau = (tauold - tau)
        verbose && println("Iteration: $i, scale: $tau, Δsig: $(Δtau)")
        tol = max(rtol * tauold, atol)
        if -tol < Δtau < tol || tau < atol
            cvg = true
            break
        end
        @assert isfinite(tau) && !iszero(tau)
        tauold = tau
    end
    cvg || throw(ConvergenceException(maxiter))
    return m
end



##########
###   Resampling
##########

"""
For S- and τ-Estimators, compute the minimum number of subsamples to draw
to ensure that with probability 1-α, at least one of the subsample
is free of outlier, given that the ratio of outlier/clean data is ε.
The number of data point per subsample is p, that should be at least
equal to the degree of freedom.
"""
function resampling_minN(p::Int, α::Real=0.05, ε::Real=0.5)
    return ceil(Int, abs(log(α) / log(1 - (1 - ε)^p)))
end


function resampling_initialcoef(m::RobustLinearModel, inds::AbstractVector{<:Integer})
    # Get the subsampled model matrix, response and weights
    Xi = modelmatrix(m)[inds, :]
    yi = response(m)[inds]
    w = weights(m)
    if isempty(w)
        # Fit with OLS
        return coef(lm(Xi, yi))
    else
        wi = w[inds]
        # Fit with OLS
        return coef(lm(Xi, yi; wts=wi))
    end
end

"""
    best_from_resampling(m::RobustLinearModel, ::Type{E}; kwargs...)
        where {E<:Union{SEstimator, MMEstimator, TauEstimator}}

Return the best scale σ0 and coefficients β0 from resampling of the S- or τ-Estimate.
"""
function resampling_best_estimate(
    m::RobustLinearModel,
    ::Type{E};
    propoutliers::Real=0.5,
    Nsamples::Union{Nothing,Int}=nothing,
    Nsubsamples::Int=10,
    Npoints::Union{Nothing,Int}=nothing,
    Nsteps_β::Int=2,
    Nsteps_σ::Int=1,
    verbose::Bool=false,
    rng::AbstractRNG=GLOBAL_RNG,
) where {E<:Union{SEstimator,MMEstimator,TauEstimator}}

    ## TODO: implement something similar to DetS (not sure it could apply)
    ## Hubert2015 - The DetS and DetMM estimators for multivariate location and scatter
    ## (https://www.sciencedirect.com/science/article/abs/pii/S0167947314002175)

    if isnothing(Nsamples)
        Nsamples = resampling_minN(dof(m), 0.05, propoutliers)
    end
    if isnothing(Npoints)
        Npoints = dof(m)
    end
    Nsubsamples = min(Nsubsamples, Nsamples)


    verbose && println("Start $(Nsamples) subsamples...")
    σis = zeros(Nsamples)
    βis = zeros(dof(m), Nsamples)
    for i in 1:Nsamples
        # TODO: to parallelize, make a deepcopy of m
        inds = sample(rng, axes(response(m), 1), Npoints; replace=false, ordered=false)
        # Find OLS fit of the subsample
        βi = resampling_initialcoef(m, inds)
        verbose && println("Sample $(i)/$(Nsamples): β0 = $(βi)")

        ## Initialize β or set it to the provided values
        setβ0!(m, βi)
        # Initialize μ and compute residuals
        setinitη!(m)
        # Initialize σ as mad(residuals)
        setinitσ!(m)

        σi = 0
        for k in 1:Nsteps_β
            setη!(
                m;
                updatescale=true,
                verbose=verbose,
                sigma0=:mad,
                nmax=Nsteps_σ,
                approx=true,
            )

            σi = if E <: TauEstimator
                tauscale(m)
            else # if E <: Union{SEstimator, MMEstimator}
                scale(m)
            end
            installbeta!(m.pred, 1)
        end
        σis[i] = σi
        βis[:, i] .= coef(m)
        verbose && println("Sample $(i)/$(Nsamples): β1=$(βis[:, i])\tσ1=$(σi)")
    end

    verbose && println("Sorted scales: $(sort(σis))")
    inds = sortperm(σis)[1:Nsubsamples]
    σls = σis[inds]
    βls = βis[:, inds]

    verbose && println("Keep best $(Nsubsamples) subsamples: $(inds)")
    for l in 1:Nsubsamples
        σl, βl = σls[l], βls[:, l]
        # TODO: to parallelize, make a deepcopy of m

        if E <: Union{SEstimator,MMEstimator}
            try
                pirls_Sestimate!(m; verbose=verbose, beta0=βl, sigma0=σl, miniter=3)
                σls[l] = scale(m)
            catch e
                # Didn't converge, set to infinite scale
                σls[l] = Inf
            end
        elseif E <: TauEstimator
            try
                pirls_τestimate!(m; verbose=verbose, beta0=βl, sigma0=σl)
                σls[l] = tauscale(m)
            catch e
                # Didn't converge, set to infinite scale
                σls[l] = Inf
            end
        else
            error("estimator $E not supported.")
        end
        # Update coefficients
        βls[:, l] .= coef(m)

        verbose && println("Subsample $(l)/$(Nsubsamples): β2=$(βls[:, l])\tσ2=$(σls[l])")
    end
    N = argmin(σls)
    ## TODO: for τ-Estimate, the returned scale is τ not σ
    verbose && println("Best subsample: β=$(βls[:, N])\tσ=$(σls[N])")
    return σls[N], βls[:, N]
end
