

######
##    TableRegressionModel methods to forward
######

leverage(p::TableRegressionModel) = leverage(p.model)
residuals(p::TableRegressionModel) = residuals(p.model)



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
    hcat(coef(m),coef(m)) + stderror(m)*quantile(Normal(), (1-level)/2)*[1. -1.]
end
confint(m::AbstractRobustModel, level::Real) = confint(m; level=level)

function show(io::IO, obj::AbstractRobustModel)
    println(io, "$(typeof(obj)):\n\nCoefficients:\n", coeftable(obj))
end

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

"""
    deviance(m::RobustLinearModel{T})::T where {T}
Return the deviance of the RobustLinearModel.
"""
deviance(m::RobustLinearModel{T}) where {T} = Base.convert(T, deviance(m.resp))

function dispersion(m::RobustLinearModel{T}, sqr::Bool = false) where T<:AbstractFloat
    r = m.resp
    if dispersion_parameter(m)
        dispersion(r, dof_residual(m), sqr)
    else
        one(T)
    end
end


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
             weights::FPVector    = similar(y, 0),
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
- `weights::Vector = []`: a weight vector, should be empty if no weights are used;
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
             weights::FPVector    = similar(y, 0),
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

    rr = RobustLinResp(est2, y, offset, weights, scale)
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



function fit!(m::RobustLinearModel{T}, y::FPVector;
                wts::Union{Nothing, FPVector}=nothing,
                offset::Union{Nothing, FPVector}=nothing,
                σ::Union{Nothing, AbstractFloat}=nothing,
                kwargs...) where {T}
    r = m.resp

    # Update y, wts and offset in the response
    copy!(r.y, y)
    if !isa(σ, Nothing); r.σ = σ end
    n = length(r.y)
    l = length(wts)
    if !isa(wts, Nothing) && (l==n || l==0)
        copy!(r.wts, wts)
    end
    l = length(offset)
    if !isa(offset, Nothing) && (l==n || l==0)
        copy!(r.offset, offset)
    end

    # Reinitialize the coefficients and the response
    fill!(coef(m), zero(T))
    initresp!(r)

    fit!(m; kwargs...)
end


"""
    fit!(m::RobustLinearModel[; verbose::Bool=false, kind::Symbol=:Mestimate])
Optimize the objective of a `RobustLinearModel`.  When `verbose` is `true` the values of the
objective and the parameters are printed on stdout at each function evaluation.
This function assumes that `m` was correctly initialized.
"""
function fit!(m::RobustLinearModel{T}; initial_scale_estimate::Union{Nothing, Symbol, Real}=nothing,
              correct_leverage::Bool=false, kind::Symbol=:Mestimate, sestimator::Union{Nothing, Type{E}}=nothing,
              verbose::Bool=false, kwargs...) where {T, E<:BoundedEstimator}

    # Return early if model has the fit flag set
    m.fitted && return m

    σ0 = if isa(initial_scale_estimate, Real)
        m.fitdispersion = false
        float(initial_scale_estimate)
    elseif isa(initial_scale_estimate, Symbol) && initial_scale_estimate == :mad
        m.fitdispersion = true
        mad(m.resp.y; normalize=true)
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

        verbose && println("\nFit with S-estimator: $(Estimator(m))")
        ## Minimize the objective
        pirls_Sestimate!(m; sigma0=σ0, beta0=[], verbose=verbose, kwargs...)

        # Set the `fitdispersion` flag to true, because σ was estimated
        m.fitdispersion = true

    elseif kind == :MMestimate
        ## Use an S-estimate to estimate the scale/dispersion
        Mest = Estimator(m)

        Sest = if !isnothing(sestimator)
            # Use an S-Estimator of the `sestimator`'s kind
            SEstimator(Mest; fallback=sestimator, force=true)
        else
            # Use an S-Estimator of the same kind as the M-Estimator or fallback
            SEstimator(Mest)
        end

        Sresp = RobustLinResp(Sest, m.resp.y, m.resp.offset, m.resp.wts, m.resp.σ)
        Sm = RobustLinearModel(Sresp, m.pred, true, m.fitted)

        verbose && println("\nFit with MM-estimator - 1. S-estimator: $(Estimator(Sm))")
        ## Minimize the objective
        pirls_Sestimate!(Sm; sigma0=σ0, beta0=[], verbose=verbose, kwargs...)

        ## Use an M-estimate to estimate coefficients
        σ0 = scale(Sm)

        verbose && println("\nFit with MM-estimator - 2. M-estimator: $(Estimator(m))")
        ## Minimize the objective
        pirls!(m; sigma0=σ0, beta0=[], verbose=verbose, kwargs...)

        # Set the `fitdispersion` flag to true, because σ was estimated
        m.fitdispersion = true
    elseif kind == :Mestimate
        verbose && println("\nFit with M-estimator: $(Estimator(m))")
        ## Minimize the objective
        pirls!(m; sigma0=σ0, beta0=[], verbose=verbose, kwargs...)
    else
        error("only :Mestimate, :Sestimate and :MMestimate are allowed: $(kind)")
    end

    m.fitted = true
    m
end



function setβ0!(m::RobustLinearModel{T}, β0::AbstractVector=[]) where {T<:AbstractFloat}
    r = m.resp
    p = m.pred

    if isempty(β0)
        # Compute beta0 from solving the least square with the response value r.y
        initresp!(r)
        delbeta!(p, r.wrkres, r.wrkwt)
        installbeta!(p)
    else
        copyto!(p.beta0, float(β0))
        fill!(p.delbeta, 0)
    end

    m
end

function setinitη!(m::RobustLinearModel{T}) where {T}
    r = m.resp
    p = m.pred

    ## Initially, β0 is defined but not ∇β, so use f=0
    linpred!(r.η, p, 0)
    updateres!(r)

    m
end

function setη!(m::RobustLinearModel{T}, f::T=1.0) where {T}
    r = m.resp
    p = m.pred

    # First update trial, compute ∇β
    if f==1
        delbeta!(p, r.wrkres, r.wrkwt)
    end
    linpred!(r.η, p, f)
    updateres!(r)

    m
end



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
            f > minstepfac || error("step-halving failed at beta0 = $(p.beta0)")
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
    pirls!(m::RobustLinearModel{T}; verbose::Bool=false, maxiter::Integer=50,
           minstepfac::Real=1e-3, atol::Real=1e-6, rtol::Real=1e-6,
           beta0::FPVector=T[], sigma0::Union{Nothing, T}=nothing,
           update_scale::Bool=false)

(Penalized) Iteratively Reweighted Least Square procedure.
The Penalized aspect is not implemented (yet).
"""
function pirls_Sestimate!(m::RobustLinearModel{T}; verbose::Bool=false, maxiter::Integer=30,
              minstepfac::Real=1e-3, atol::Real=1e-6, rtol::Real=1e-6,
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
    sigold = try
        optimscale(setη!(m).resp; verbose=verbose)
    catch e
        if isa(e, ConvergenceFailed)
            sigold = maxσ
        else
            rethrow(e)
        end
    end
    installbeta!(p, 1)
    r.σ = sigold

    verbose && println("initial scale: $(@sprintf("%.4g", sigold))")
    for i = 1:maxiter
        f = 1.0 # line search factor
        local sig

        # Compute the change to β, update μ and compute deviance
        try
            sig = optimscale(setη!(m).resp; verbose=verbose)
        catch e
            if isa(e, ConvergenceFailed)
                sig = maxσ
            else
                rethrow(e)
            end
        end

        # Assert the deviance is positive (up to rounding error)
        @assert sig > -atol

        verbose && println("scale at step $i: $(@sprintf("%.4g", sig)), crit=$((sigold - sig)/sigold)")

        # Line search
        ## If the scale isn't declining then half the step size
        ## The rtol*abs(sigold) term is to avoid failure when scale
        ## is unchanged except for rounding errors.
        while sig > sigold + rtol*sig
            f /= 2
            f > minstepfac || error("step-halving failed at beta0 = $(p.beta0)")
            try
                # Update μ and compute deviance with new f. Do not recompute ∇β
                sig = optimscale(setη!(m).resp; sigma0=sigold)
            catch e
                if isa(e, ConvergenceFailed)
                    sig = maxσ
                else
                    rethrow(e)
                end
            end
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
