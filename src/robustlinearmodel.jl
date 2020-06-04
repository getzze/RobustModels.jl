

######
##    TableRegressionModel methods to forward
######

leverage(p::TableRegressionModel)    = leverage(p.model)
residuals(p::TableRegressionModel)   = residuals(p.model)
weights(p::TableRegressionModel)     = weights(p.model)
scale(p::TableRegressionModel)       = scale(p.model)
dispersion(p::TableRegressionModel)  = dispersion(p.model)
modelmatrix(p::TableRegressionModel) = modelmatrix(p.model)
fit!(p::TableRegressionModel, args...; kwargs...) = (fit!(p.model, args...; kwargs...); p)
refit!(p::TableRegressionModel, args...; kwargs...) = (refit!(p.model, args...; kwargs...); p)

"""
    RobustLinearModel
Robust linear model representation
## Fields
* `resp`: the response structure
* `pred`: the predictor structure
* `fitdispersion`: if true, the dispersion is estimated otherwise it is kept fixed
* `fitted`: if true, the model was already fitted
"""
mutable struct RobustLinearModel{T<:AbstractFloat, R<:RobustResp{T}, L<:LinPred} <: AbstractRobustModel{T}
    resp::R
    pred::L
    fitdispersion::Bool
    fitted::Bool
end


######
##    AbstractRobustModel methods
######

objective(m::AbstractRobustModel) = deviance(m)

dof(m::AbstractRobustModel)::Int = length(coef(m))

dof_residual(m::AbstractRobustModel)::Int = nobs(m) - dof(m)

function coeftable(m::AbstractRobustModel)
    cc = coef(m)
    se = stderror(m)
    X = modelmatrix(m)
    zz = cc ./ se
    CoefTable(hcat(cc, se, zz, ccdf.(Chisq(1), abs2.(zz))),
              ["Estimate","Std.Error","z value", "Pr(>|z|)"],
              ["x$i" for i = 1:size(X, 2)], 4)
end

function confint(m::AbstractRobustModel; level::Real=0.95)
    alpha = quantile(TDist(dof_residual(m)), (1-level)/2)
    hcat(coef(m), coef(m)) + stderror(m)*alpha*[1.0  -1.0]
end
confint(m::AbstractRobustModel, level::Real) = confint(m; level=level)


function cor(m::AbstractRobustModel)
    Σ = vcov(m)
    invstd = inv.(sqrt.(diag(Σ)))
    lmul!(Diagonal(invstd), rmul!(Σ, Diagonal(invstd)))
end

## TODO: specialize to make it faster
leverage(p::AbstractRobustModel) = diag(projectionmatrix(p))



######
##    RobustLinearModel methods
######

function show(io::IO, obj::RobustLinearModel)
    println(io, "Robust regression with $(obj.resp.est)\n\nCoefficients:\n", coeftable(obj))
end

function show(io::IO, obj::TableRegressionModel{M, T}) where {T, M<:RobustLinearModel}
    println(io, "Robust regression with $(obj.model.resp.est)\n\n$(obj.mf.f)\n\nCoefficients:\n", coeftable(obj))
end

"""
    deviance(m::RobustLinearModel{T})::T where {T}
Return the deviance of the RobustLinearModel.
"""
deviance(m::RobustLinearModel{T}) where {T} = Base.convert(T, deviance(m.resp))

nulldeviance(m::RobustLinearModel{T}) where {T} = Base.convert(T, nulldeviance(m.resp))

dispersion(m::RobustLinearModel{T}, sqr::Bool=false) where {T} = dispersion(m.resp, dof_residual(m), sqr)


"""
    nobs(m::RobustLinearModel{T})
For linear and generalized linear models, returns the number of elements of the response.
"""
nobs(m::RobustLinearModel)::Int = nobs(m.resp)

coef(m::RobustLinearModel) = coef(m.pred)

function Estimator(m::RobustLinearModel)
    Estimator(m.resp)
end

stderror(m::RobustLinearModel) = location_variance(m.resp, dof_residual(m), false) .* sqrt.(diag(vcov(m)))

loglikelihood(m::RobustLinearModel) = loglikelihood(m.resp)

nullloglikelihood(m::RobustLinearModel) = nullloglikelihood(m.resp)

weights(m::RobustLinearModel) = weights(m.resp)

response(m::RobustLinearModel) = response(m.resp)

isfitted(m::RobustLinearModel) = m.fitted

fitted(m::RobustLinearModel) = fitted(m.resp)

residuals(m::RobustLinearModel) = residuals(m.resp)

scale(m::RobustLinearModel) = m.resp.scale

function predict(m::RobustLinearModel, newX::AbstractMatrix;
                 offset::FPVector=eltype(newX)[])
    mu = newX * coef(m)
    if !isempty(m.resp.offset)
        length(offset) == size(newX, 1) ||
            throw(ArgumentError("fit with offset, so `offset` kwarg must be an offset of length `size(newX, 1)`"))
        broadcast!(+, mu, mu, offset)
    else
        length(offset) > 0 && throw(ArgumentError("fit without offset, so value of `offset` kwarg does not make sense"))
    end
    mu
end
predict(m::RobustLinearModel) = fitted(m)



"""
    rlm(X, y, args...; kwargs...)
An alias for `fit(RobustLinearModel, X, y)`
The arguments `X` and `y` can be a `Matrix` and a `Vector` or a `Formula` and a `DataFrame`.
"""
rlm(X, y, args...; kwargs...) = fit(RobustLinearModel, X, y, args...; kwargs...)


"""
   fit(::Type{M}, X::Union{AbstractMatrix{T},SparseMatrixCSC{T}},
             y::AbstractVector{T}, est::Estimator;
             quantile::Union{Nothing, AbstractFloat} = nothing,
             method::Symbol       = :chol, # :cg
             scale::AbstractFloat = 1.0,
             dofit::Bool          = true,
             wts::FPVector        = similar(y, 0),
             offset::FPVector     = similar(y, 0),
             fitdispersion::Bool  = false,
             fitargs...) where {M<:RobustLinearModel, T<:AbstractFloat}

Create a robust model with the model matrix X and response vector y (or formula/data),
using a robust estimator.
A quantile can be provided to perform MQuantile regression.


# Arguments

- `X`: the model matrix (it can be dense or sparse) or a formula
- `y`: the response vector or a dataframe.
- `est`: a robust estimator

## Keywords

- `quantile::Union{Nothing, AbstractFloat} = nothing`: optionnaly run a M-quantile regression using the estimator `est`;
- `method::Symbol = :chol`: the method to use for solving the weighted linear system, `chol` (default) or `cg`;
- `scale::AbstractFloat = 1.0`: an estimate for the scale;
- `dofit::Bool = true`: if false, return the model object without fitting;
- `wts::Vector = []`: a weight vector, should be empty if no weights are used;
- `offset::Vector = []`: an offset vector, should be empty if no offset is used;
- `fitdispersion::Bool = false`: reevaluate the dispersion;
- `fitargs...`: other keyword arguments used to control the convergence of the IRLS algorithm (see [`pirls!`](@ref)).

# Output

the RobustLinearModel object.

"""
function fit(::Type{M}, X::Union{AbstractMatrix{T},SparseMatrixCSC{T}},
             y::AbstractVector{T}, est::Estimator;
             quantile::Union{Nothing, AbstractFloat} = nothing,
             method::Symbol       = :chol, # :cg
             scale::AbstractFloat = 1.0,
             dofit::Bool          = true,
             wts::FPVector        = similar(y, 0),
             offset::FPVector     = similar(y, 0),
             fitdispersion::Bool  = false,
             fitargs...) where {M<:RobustLinearModel, T<:AbstractFloat}

    # Check that X and y have the same number of observations
    if size(X, 1) != size(y, 1)
        throw(DimensionMismatch("number of rows in X and y must match"))
    end

    est2 = if !isnothing(quantile) && (0 < quantile < 1)
        if isa(est, GeneralQuantileEstimator)
            GeneralQuantileEstimator(est.est, quantile)
        else
            GeneralQuantileEstimator(est, quantile)
        end
    else
        est
    end

    rr = RobustLinResp(est2, y, offset, wts, scale)
#    pp = if method == :cg; cgpred(X) elseif method==:chol; cholpred(X) else qrpred(X) end
    pp = if method==:cg; cgpred(X) else cholpred(X) end

    m = RobustLinearModel(rr, pp, fitdispersion, false)
    return if dofit; fit!(m; fitargs...) else m end
end

function fit(::Type{M}, X::Union{AbstractMatrix,SparseMatrixCSC},
             y::AbstractVector, est::Estimator;
             kwargs...) where {M<:AbstractRobustModel}
    fit(M, float(X), float(y), est; kwargs...)
end


"""
    refit!(m::RobustLinearModel, [y::FPVector ; verbose::Bool=false, kind::Symbol=:Mestimate])
Optimize the objective of a `RobustLinearModel`.  When `verbose` is `true` the values of the
objective and the parameters are printed on stdout at each function evaluation.
This function assumes that `m` was correctly initialized and the model is refitted with
the new values for the response, weights, offset, scale, quantile and initial_coef.
"""
function refit!(m::RobustLinearModel, y::FPVector; kwargs...)
    r = m.resp
    # Check that old and new y have the same number of observations
    if size(r.y, 1) != size(y, 1)
        throw(DimensionMismatch("the new response vector should have the same dimension:  $(size(r.y, 1)) != $(size(y, 1))"))
    end
    # Update y
    copyto!(r.y, y)
    
    refit!(m; kwargs...)
end

function refit!(m::RobustLinearModel{T};
                wts::Union{Nothing, FPVector}=nothing,
                offset::Union{Nothing, FPVector}=nothing,
                σ::Union{Nothing, AbstractFloat}=nothing,
                quantile::Union{Nothing, AbstractFloat} = nothing,
                kwargs...) where {T}

    if haskey(kwargs, :method)
        @warn("the method argument is not used for refitting, ignore.")
        delete!(kwargs, :method)
    end

    r = m.resp

    if !isa(σ, Nothing); r.σ = σ end
    n = length(r.y)
    if !isa(wts, Nothing) && (length(wts) in (0, n))
        copy!(r.wts, wts)
    end
    if !isa(offset, Nothing) && (length(offset) in (0, n))
        copy!(r.offset, offset)
    end

    # Update quantile, if it was defined before
    if !isnothing(quantile)
        if isa(r.est, GeneralQuantileEstimator)
            (0 < quantile < 1) || throw(DomainError(quantile, "quantile should be a number between 0 and 1 excluded"))
            r.est = GeneralQuantileEstimator(r.est.est, quantile)
        else
            error("quantile can only be changed if the original model is a GeneralQuantileEstimator.")
        end
    end

    # Reinitialize the coefficients and the response
    fill!(coef(m), zero(T))
    initresp!(r)

    m.fitted = false
    fit!(m; kwargs...)
end



"""
    fit!(m::RobustLinearModel[; verbose::Bool=false, kind::Symbol=:Mestimate])
Optimize the objective of a `RobustLinearModel`.  When `verbose` is `true` the values of the
objective and the parameters are printed on stdout at each function evaluation.
This function assumes that `m` was correctly initialized.
This function returns early if the model was already fitted, instead call `refit!`.
"""
function fit!(m::RobustLinearModel{T}; initial_scale_estimate::Union{Nothing, Symbol, Real}=nothing,
              verbose::Bool=false, kind::Symbol=:Mestimate, sestimator::Union{Nothing, Type{E}}=nothing,
              initial_coef::AbstractVector=[], 
              resample::Bool=false, resampling_options::Dict{Symbol, F}=Dict{Symbol, Any}(:verbose=>verbose),
              correct_leverage::Bool=false, kwargs...) where {T, E<:BoundedEstimator, F}

    # Return early if model has the fit flag set
    m.fitted && return m

    σ0 = if isa(initial_scale_estimate, Real)
        float(initial_scale_estimate)
    elseif isa(initial_scale_estimate, Symbol)
        initialscale(m, initial_scale_estimate)
    end

    β0 = if isempty(initial_coef) || size(initial_coef, 1) != size(coef(m), 1)
        []
    else
        float(initial_coef)
    end


    if correct_leverage
        wts = m.resp.wts
        copy!(wts, leverage_weights(m))
        ## TODO: maybe multiply by the old wts?
    end

    if kind == :Sestimate
        Mest = Estimator(m)
        isbounded(Mest) || error("Only bounded estimators are allowed for S-Estimation: $(Mest)")

        ## Change the estimator to an S-Estimator of the same kind
        m.resp.est = SEstimator(Mest)

        ## TODO: Resampling algorithm
        if resample
            σ0, β0 = resampling_best_estimate(m, kind; resampling_options...)
        end
        
        verbose && println("\nFit with S-estimator: $(Estimator(m))")
        ## Minimize the objective
        pirls_Sestimate!(m; sigma0=σ0, beta0=β0, verbose=verbose, kwargs...)

        # Set the `fitdispersion` flag to true, because σ was estimated
        m.fitdispersion = true

    elseif kind == :MMestimate
        ## Use an S-estimate to estimate the scale/dispersion
        Mest = Estimator(m)

        ## TODO: Create a type MMEstimator that holds the two estimator
        Sest = if !isnothing(sestimator)
            # Use an S-Estimator of the `sestimator`'s kind
            SEstimator(Mest; fallback=sestimator, force=true)
        else
            # Use an S-Estimator of the same kind as the M-Estimator or fallback
            SEstimator(Mest)
        end

        Sresp = RobustLinResp(Sest, m.resp.y, m.resp.offset, m.resp.wts, m.resp.σ)
        Sm = RobustLinearModel(Sresp, m.pred, true, m.fitted)

        ## TODO: Resampling algorithm
        if resample
            σ0, β0 = resampling_best_estimate(m, :Sestimate; resampling_options...)
        end

        verbose && println("\nFit with MM-estimator - 1. S-estimator: $(Estimator(Sm))")
        ## Minimize the objective
        pirls_Sestimate!(Sm; sigma0=σ0, beta0=β0, verbose=verbose, kwargs...)

        ## Use an M-estimate to estimate coefficients
        β0 = coef(Sm)
        σ0 = scale(Sm)

        verbose && println("\nFit with MM-estimator - 2. M-estimator: $(Estimator(m))")
        ## Minimize the objective
        pirls!(m; sigma0=σ0, beta0=β0, verbose=verbose, kwargs...)

        # Set the `fitdispersion` flag to true, because σ was estimated
        m.fitdispersion = true

    elseif kind == :Tauestimate
        Mest = Estimator(m)
        isa(Mest, TauEstimator) || error("Use a TauEstimator for τ-Estimation: $(Mest)")
        
        ## TODO: Resampling algorithm
        if resample
            ## TODO: add extract rng key from resampling_options
            σ0, β0 = resampling_best_estimate(m, kind; resampling_options...)
        end

        verbose && println("\nFit with τ-estimator: $(Estimator(m))")
        ## Minimize the objective
        pirls_τestimate!(m; sigma0=σ0, beta0=β0, verbose=verbose, kwargs...)

        # Set the `fitdispersion` flag to true, because σ was estimated
        m.fitdispersion = true
        
    elseif kind == :Mestimate
        verbose && println("\nFit with M-estimator: $(Estimator(m))")
        ## Minimize the objective
        pirls!(m; sigma0=σ0, beta0=β0, verbose=verbose, kwargs...)

        ## TODO: update scale is fitdispersion is true
    else
        error("only :Mestimate, :Sestimate, :MMestimate and :Tauestimate are allowed: $(kind)")
    end

    m.fitted = true
    m
end


function initialscale(m::RobustLinearModel, method::Symbol=:mad; factor::AbstractFloat=1.0)
    factor > 0 || error("factor should be positive")

    y = response(m)
    wts = weights(m)

    allowed_methods = (:mad, :extrema, :L1)
    if method == :mad
        σ = if length(wts) == length(y)
            factor*mad(wts .* abs.(y); normalize=true)
        else
            factor*mad(abs.(y); normalize=true)
        end
    elseif method == :extrema
        σ = -(-(extrema(y)...))/2
    elseif method == :L1
        X = modelmatrix(m)
        σ = dispersion(quantreg(X, y; wts=wts))
    else
        error("only $(join(allowed_methods, ", ", " and ")) methods are allowed")
    end
    return σ
end

function setβ0!(m::RobustLinearModel{T}, β0::AbstractVector=[]) where {T<:AbstractFloat}
    r = m.resp
    p = m.pred

    initresp!(r)
    if isempty(β0)
        # Compute beta0 from solving the least square with the response value r.y
#        initresp!(r)
        delbeta!(p, r.wrkres, r.wrkwt)
        installbeta!(p)
    else
        copyto!(p.beta0, float(β0))
        fill!(p.delbeta, 0)
    end

    m
end

"""
    setinitη!(m)
Compute the predictor using the initial value of β0 and compute the residuals
"""
function setinitη!(m::RobustLinearModel{T}) where {T}
    r = m.resp
    p = m.pred

    ## Initially, β0 is defined but not ∇β, so use f=0
    linpred!(r.η, p, 0)
    updateres!(r)

    m
end

setinitσ!(m::RobustLinearModel; kwargs...) = (m.resp.σ = initialresidualscale(m.resp; kwargs...); m)

"""
    setη!(m)
Compute the ∇β using the current residuals and working weights (only if f=1,
which corresponds to the first iteration of linesearch), then compute
the predictor using the ∇β value and compute the new residuals and deviance.
if update_scale is true, the scale is also updated using the residuals.
"""
function setη!(m::RobustLinearModel{T}, f::T=1.0; update_scale::Bool=false, kwargs...) where {T}
    r = m.resp
    p = m.pred

    # First update of linesearch algorithm, compute ∇β
    if f==1
        delbeta!(p, r.wrkres, r.wrkwt)
    end
    # Compute and set the predictor η from β0 and ∇β
    linpred!(r.η, p, f)

    # Update the residuals and weights (and scale)
    if update_scale
        update_res_and_scale!(r; kwargs...)
    else
        updateres!(r; kwargs...)
    end
    m
end

tauscale(m::RobustLinearModel) = tauscale(m.resp)


"""
    pirls!(m::RobustLinearModel{T}; verbose::Bool=false, maxiter::Integer=50,
           minstepfac::Real=1e-3, atol::Real=1e-6, rtol::Real=1e-6,
           beta0::FPVector=T[], sigma0::Union{Nothing, T}=nothing,
           update_scale::Bool=false)

(Penalized) Iteratively Reweighted Least Square procedure.
The Penalized aspect is not implemented (yet).
"""
function pirls!(m::RobustLinearModel{T}; verbose::Bool=false, maxiter::Integer=30,
              minstepfac::Real=1e-3, atol::Real=1e-6, rtol::Real=1e-6,
              beta0::AbstractVector=[], sigma0::Union{Nothing, T}=nothing) where {T<:AbstractFloat}

    # Check arguments
    maxiter >= 1       || throw(ArgumentError("maxiter must be positive"))
    0 < minstepfac < 1 || throw(ArgumentError("minstepfac must be in (0, 1)"))

    # Extract fields and set convergence flag
    cvg, p, r = false, m.pred, m.resp

    if !isnothing(sigma0)
        r.σ = sigma0
    end

    ## Initialize β or set it to the provided values
    setβ0!(m, beta0)

    # Initialize μ and compute residuals
    setinitη!(m)

    # Compute initial deviance
    devold = deviance(m)

    verbose && println("initial deviance: $(@sprintf("%.4g", devold))")
    for i = 1:maxiter
        f = 1.0 # line search factor
        local dev
        absdev = abs(devold)

        # Compute the change to β, update μ and compute deviance
        try
            dev = deviance(setη!(m))
        catch e
            if isa(e, DomainError)
                dev = Inf
            else
                rethrow(e)
            end
        end

        # Assert the deviance is positive (up to rounding error)
        @assert dev > -atol

        verbose && println("deviance at step $i: $(@sprintf("%.4g", dev)), crit=$((devold - dev)/abs(devold))")

        # Line search
        ## If the deviance isn't declining then half the step size
        ## The rtol*abs(devold) term is to avoid failure when deviance
        ## is unchanged except for rounding errors.
        while dev > devold + rtol*absdev
            f /= 2
            f > minstepfac || error("linesearch failed at iteration $(i) with beta0 = $(p.beta0)")

            try
                # Update μ and compute deviance with new f. Do not recompute ∇β
                dev = deviance(setη!(m, f))
            catch e
                if isa(e, DomainError)
                    dev = Inf
                else
                    rethrow(e)
                end
            end
        end
        installbeta!(p, f)

        # Test for convergence
        Δdev = (devold - dev)
        verbose && println("Iteration: $i, deviance: $dev, Δdev: $(Δdev)")
        tol = max(rtol*absdev, atol)
        if -tol < Δdev < tol || dev < atol
            cvg = true
            break
        end
        @assert isfinite(dev)
        devold = dev
    end
    cvg || throw(ConvergenceException(maxiter))
    m
end


"""
    pirls_Sestimate!(m::RobustLinearModel{T}; verbose::Bool=false, maxiter::Integer=50,
           minstepfac::Real=1e-3, atol::Real=1e-6, rtol::Real=1e-6,
           beta0::FPVector=T[], sigma0::Union{Nothing, T}=nothing,
           update_scale::Bool=false)

(Penalized) Iteratively Reweighted Least Square procedure.
The Penalized aspect is not implemented (yet).
"""
function pirls_Sestimate!(m::RobustLinearModel{T}; verbose::Bool=false, maxiter::Integer=30,
              minstepfac::Real=1e-3, atol::Real=1e-6, rtol::Real=1e-6, miniter::Int=2,
              beta0::AbstractVector=[], sigma0::Union{Nothing, T}=nothing) where {T<:AbstractFloat}

    # Check arguments
    maxiter >= 1       || throw(ArgumentError("maxiter must be positive"))
    0 < minstepfac < 1 || throw(ArgumentError("minstepfac must be in (0, 1)"))

    # Extract fields and set convergence flag
    cvg, p, r = false, m.pred, m.resp

    maxσ = -(-(extrema(r.y)...))/2
    if !isnothing(sigma0)
        r.σ = sigma0
    else
        r.σ = maxσ
    end

    ## Initialize β or set it to the provided values
    setβ0!(m, beta0)

    # Initialize μ and compute residuals
    setinitη!(m)

    # Compute initial scale
    sigold = scale(setη!(m; update_scale=true, verbose=verbose, sigma0=sigma0, fallback=maxσ))
#    sigold = optimscale(setη!(m).resp; verbose=verbose, sigma0=sigma0, fallback=maxσ)
    installbeta!(p, 1)
    r.σ = sigold

    verbose && println("initial scale: $(@sprintf("%.4g", sigold))")
    for i = 1:maxiter
        f = 1.0 # line search factor
        local sig

        # Compute the change to β, update μ and compute deviance
        sig = scale(setη!(m; update_scale=true, verbose=verbose, sigma0=sigold, fallback=maxσ))
#        sig = optimscale(setη!(m).resp; verbose=verbose, sigma0=sigold, fallback=maxσ)

        # Assert the deviance is positive (up to rounding error)
        @assert sig > -atol

        verbose && println("scale at step $i: $(@sprintf("%.4g", sig)), crit=$((sigold - sig)/sigold)")

        # Line search
        ## If the scale isn't declining then half the step size
        ## The rtol*abs(sigold) term is to avoid failure when scale
        ## is unchanged except for rounding errors.
        while sig > sigold*(1 + rtol)
            f /= 2
            if f <= minstepfac
                if i <= miniter
                    sigold = maxσ
                    r.σ = sigold
                    verbose && println("linesearch failed at early iteration $(i), set scale to maximum value: $(sigold)")
                else
                    error("linesearch failed at iteration $(i) with beta0 = $(p.beta0)")
                end
            end
            # Update μ and compute deviance with new f. Do not recompute ∇β
            sig = scale(setη!(m; update_scale=true, verbose=verbose, sigma0=sigold, fallback=maxσ))
#            sig = optimscale(setη!(m).resp; sigma0=sigold, fallback=maxσ)
        end
        installbeta!(p, f)
        r.σ = sig

        # Test for convergence
        Δsig = (sigold - sig)
        verbose && println("Iteration: $i, scale: $sig, Δsig: $(Δsig)")
        tol = max(rtol*sigold, atol)
        if -tol < Δsig < tol || sig < atol
            cvg = true
            break
        end
        @assert isfinite(sig) && !iszero(sig)
        sigold = sig
    end
    cvg || throw(ConvergenceException(maxiter))
    m
end


"""
    pirls_τestimate!(m::RobustLinearModel{T}; verbose::Bool=false, maxiter::Integer=50,
           minstepfac::Real=1e-3, atol::Real=1e-6, rtol::Real=1e-6,
           beta0::FPVector=T[], sigma0::Union{Nothing, T}=nothing,
           update_scale::Bool=false)

(Penalized) Iteratively Reweighted Least Square procedure.
The Penalized aspect is not implemented (yet).
"""
function pirls_τestimate!(m::RobustLinearModel{T}; verbose::Bool=false, maxiter::Integer=30,
              minstepfac::Real=1e-3, atol::Real=1e-6, rtol::Real=1e-6, miniter::Int=2,
              beta0::AbstractVector=[], sigma0::Union{Nothing, T}=nothing) where {T<:AbstractFloat}

    # Check arguments
    maxiter >= 1       || throw(ArgumentError("maxiter must be positive"))
    0 < minstepfac < 1 || throw(ArgumentError("minstepfac must be in (0, 1)"))

    # Extract fields and set convergence flag
    cvg, p, r = false, m.pred, m.resp

    maxσ = -(-(extrema(r.y)...))/2
    if !isnothing(sigma0)
        r.σ = sigma0
    else
        r.σ = maxσ
    end

    ## Initialize β or set it to the provided values
    setβ0!(m, beta0)

    # Initialize μ and compute residuals
    setinitη!(m)

    # Compute initial τ-scale
    tauold = tauscale(setη!(m; update_scale=true).resp; verbose=verbose)
    installbeta!(p, 1)

    verbose && println("initial τ-scale: $(@sprintf("%.4g", tauold))")
    for i = 1:maxiter
        f = 1.0 # line search factor
        local tau

        # Compute the change to β, update μ and compute deviance
        tau = tauscale(setη!(m; update_scale=true, verbose=true, fallback=maxσ).resp; verbose=verbose)

        # Assert the deviance is positive (up to rounding error)
        @assert tau > -atol

        verbose && println("scale at step $i: $(@sprintf("%.4g", tau)), crit=$((tauold - tau)/tauold)")

        # Line search
        ## If the scale isn't declining then half the step size
        ## The rtol*abs(sigold) term is to avoid failure when scale
        ## is unchanged except for rounding errors.
        while tau > tauold + rtol*tau
            f /= 2
            if f <= minstepfac
                if i <= miniter
                    tauold = maxσ
                    r.σ = tauold
                    verbose && println("linesearch failed at early iteration $(i), set scale to maximum value: $(tauold)")
                else
                    error("linesearch failed at iteration $(i) with beta0 = $(p.beta0)")
                end
            end

            # Update μ and compute deviance with new f. Do not recompute ∇β
            tau = tauscale(setη!(m; update_scale=true).resp)
        end
        installbeta!(p, f)

        # Test for convergence
        Δtau = (tauold - tau)
        verbose && println("Iteration: $i, scale: $tau, Δsig: $(Δtau)")
        tol = max(rtol*tauold, atol)
        if -tol < Δtau < tol || tau < atol
            cvg = true
            break
        end
        @assert isfinite(tau) && !iszero(tau)
        tauold = tau
    end
    cvg || throw(ConvergenceException(maxiter))
    m
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
    ceil(Int, abs(log(α) / log(1 - (1-ε)^p)))
end


function resampling_initialcoef(m, inds)
    # Get the subsampled model matrix, response and weights
    Xi = modelmatrix(m)[inds, :]
    yi = response(m)[inds]
    wi = Vector(weights(m)[inds])

    # Fit with OLS
    coef(lm(Xi, yi; wts=wi))
end

"""
    best_from_resampling(m::RobustLinearModel, kind::Symbol; Nsamples=nothing)

Return the best scale σ0 and coefficients β0 from resampling of the S- or τ-Estimate.
"""
function resampling_best_estimate(m::RobustLinearModel, kind::Symbol;
            propoutliers::Real=0.5, Nsamples::Union{Nothing, Int}=nothing, Nsubsamples::Int=10,
            Npoints::Union{Nothing, Int}=nothing, Nsteps_β::Int=2, Nsteps_σ::Int=1,
            verbose::Bool=false, rng::AbstractRNG=GLOBAL_RNG)
    kind in (:Sestimate, :Tauestimate) || error("resampling is implemented for :Sestimate or :Tauestimate only: $(kind)")
    
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
        inds = sample(rng, 1:nobs(m), Npoints; replace=false, ordered=false)
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
            setη!(m; update_scale=true, verbose=verbose, sigma0=:mad, nmax=Nsteps_σ, approx=true)
            
            σi = if kind == :Sestimate
                scale(m)
            else # if kind == :Tauestimate
                tauscale(m)
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

        try
            if kind == :Sestimate
                pirls_Sestimate!(m; verbose=verbose, beta0=βl, sigma0=σl, miniter=3)
                σls[l] = scale(m)
            else
                pirls_τestimate!(m; verbose=verbose, beta0=βl, sigma0=σl)
                σls[l] = tauscale(m)
            end
            βls[:, l] .= coef(m)
        catch e
            # Didn't converge, set to infinite scale
            σls[l] = Inf
        end
        verbose && println("Subsample $(l)/$(Nsubsamples): β2=$(βls[:, l])\tσ2=$(σls[l])")
    end
    N = argmin(σls)
    ## TODO: for τ-Estimate, the returned scale is τ not σ
    verbose && println("Best subsample: β=$(βls[:, N])\tσ=$(σls[N])")
    return σls[N], βls[:, N]
end
