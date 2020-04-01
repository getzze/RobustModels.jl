"""
    RobustLinearModel
Robust linear model representation
## Fields
* `formula`: the formula for the model
* `est`: The Estimator used for the model. It can be an MEstimator, quantile or expectile estimator.
* `allterms`: a vector of random-effects terms, the fixed-effects terms and the response
* `sqrtwts`: vector of square roots of the case weights.  Can be empty.
* `A`: an `nt × nt` symmetric `BlockMatrix` of matrices representing `hcat(Z,X,y)'hcat(Z,X,y)`
* `L`: a `nt × nt` `BlockMatrix` - the lower Cholesky factor of `Λ'AΛ+I`
* `optsum`: an [`OptSummary`](@ref) object
## Properties
* `θ` or `theta`: the covariance parameter vector used to form λ
* `β` or `beta`: the fixed-effects coefficient vector
* `λ` or `lambda`: a vector of lower triangular matrices repeated on the diagonal blocks of `Λ`
* `σ` or `sigma`: current value of the standard deviation of the per-observation noise
* `lowerbd`: lower bounds on the elements of θ
* `X`: the fixed-effects model matrix
* `y`: the response vector
"""
mutable struct RobustLinearModel{T<:AbstractFloat, R<:RobustResp{T}, L<:LinPred} <: AbstractRobustModel{T}
    resp::R
    pred::L
    fitdispersion::Bool
    fitted::Bool
end


"""
    deviance(m::RobustLinearModel{T})::T where {T}
Return the deviance of the RobustLinearModel.
"""
deviance(m::RobustLinearModel{T}) where {T} = Base.convert(T, deviance(m.resp))

objective(m::RobustLinearModel) = deviance(m)

function dispersion(m::RobustLinearModel{T}, sqr::Bool = false) where T<:AbstractFloat
    r = m.resp
    if dispersion_parameter(m)
        dispersion(r, dof_residual(m), sqr)
    else
        one(T)
    end
end

dispersion_parameter(m::RobustLinearModel) = if m.dofitdispersion; true else false end
#GLM.dispersion_parameter(m::GeneralizedRobustLinearModel) = if m.dofitdispersion; dispersion_parameter(m.resp.d) else false end

"""
    nobs(m::AbstractRobustModel)
For linear and generalized linear models, returns the number of elements of the response.
"""
nobs(m::AbstractRobustModel) = nobs(m.resp)

coef(m::AbstractRobustModel) = coef(m.pred)

function dof(m::AbstractRobustModel)::Int
    length(coef(m)) + GLM.dispersion_parameter(m)
end

function dof_residual(m::AbstractRobustModel)::Int
    nobs(m) - dof(m)
end

function show(io::IO, obj::AbstractRobustModel)
    println(io, "$(typeof(obj)):\n\nCoefficients:\n", coeftable(obj))
end

function Estimator(m::AbstractRobustModel)
    Estimator(m.resp)
end

function coeftable(m::RobustLinearModel)
    cc = coef(m)
    se = stderror(m)
    X = modelmatrix(m)
    zz = cc ./ se
    CoefTable(hcat(cc, se, zz, ccdf.(Chisq(1), abs2.(zz))),
              ["Estimate","Std.Error","z value", "Pr(>|z|)"],
              ["x$i" for i = 1:size(X, 2)], 4)
end

function confint(m::RobustLinearModel, level::Real)
    hcat(coef(m),coef(m)) + stderror(m)*quantile(Normal(), (1-level)/2)*[1. -1.]
end
confint(m::RobustLinearModel) = confint(m, 0.95)

stderror(m::RobustLinearModel) = sqrt.(diag(vcov(m)))

loglikelihood(m::RobustLinearModel) = loglikelihood(m.resp)

weights(m::RobustLinearModel) = weights(m.resp)

response(m::RobustLinearModel) = response(m.resp)

leverage(m::RobustLinearModel) = leverage(m.pred)

isfitted(m::RobustLinearModel) = m.fitted

fitted(m::RobustLinearModel) = fitted(m.resp)

residuals(m::RobustLinearModel) = residuals(m.resp)

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
    rlm(X, y, allowrankdeficient::Bool=false; kwargs...)
An alias for `fit(LinearModel, X, y, allowrankdeficient)`
The arguments `X` and `y` can be a `Matrix` and a `Vector` or a `Formula` and a `DataFrame`.
"""
rlm(X, y, args...; kwargs...) = fit(RobustLinearModel, X, y, args...; kwargs...)


function fit(::Type{M}, X::Union{AbstractMatrix{T},SparseMatrixCSC{T}},
             y::AbstractVector{T}, est::Estimator;
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
    L = size(X, 2)

    rr = RobustLinResp(est, y, offset, weights, scale)
    pp = ifelse(method == :cg, cgpred(X), cholpred(X))
    
    m = RobustLinearModel(rr, pp, fitdispersion, false)
    return ifelse(dofit, fit!(m; fitargs...), m)
end

function fit(::Type{M}, X::Union{AbstractMatrix,SparseMatrixCSC},
             y::AbstractVector, est::Estimator;
             kwargs...) where {M<:AbstractRobustModel}
    fit(M, float(X), float(y), est; kwargs...)
end



function setη!(m::RobustLinearModel{T}) where {T}
    r = m.resp
    p = m.pred

    delbeta!(p, r.wrkresid, r.wrkwt)
    linpred!(r.η, p)
    updateres!(r)

    m
end

function setinitη!(m::RobustLinearModel{T}) where {T}
    r = m.resp
    p = m.pred

    initwt = ifelse(!isempty(r.wts), r.wts, ones(size(r.y)))

    delbeta!(p, r.y, initwt)
    linpred!(r.η, p)
    updateres!(r)

    m
end


function fit!(m::RobustLinearModel{T}, y::FPVector; 
                wts::Union{Nothing, FPVector}=nothing,
                offset::Union{Nothing, FPVector}=nothing,
                λ::Union{Nothing, AbstractFloat}=nothing,
                kwargs...) where {T}
    r = m.resp
    
    # Update y, wts and offset in the response
    copy!(r.y, y)
    if !isa(wts, Nothing); r.λ = λ end
    n = length(r.y)
    if !isa(wts, Nothing) && n == length(wts)
        copy!(r.wts, wts)
    end
    if !isa(offset, Nothing) && n == length(offset)
        copy!(r.offset, offset)
    end

    # Reinitialize the coefficients
    fill!(coef(m), zero(T))
    
    fit!(m; kwargs...)
end


"""
    fit!(m::RobustLinearModel[; verbose::Bool=false, REML::Bool=false])
Optimize the objective of a `RobustLinearModel`.  When `verbose` is `true` the values of the
objective and the parameters are printed on stdout at each function evaluation.
This function assumes that `m` was correctly initialized.
"""
function fit!(m::RobustLinearModel{T}; verbose::Bool=false, estimate_scale::Union{Nothing, Symbol, Real}=nothing, maxiter::Integer=30,
              minstepfac::Real=1e-3, atol::Real=1e-6, rtol::Real=1e-6) where {T}

    # Return early if model has the fit flag set
    m.fitted && return m

    # Check arguments
    maxiter >= 1       || throw(ArgumentError("maxiter must be positive"))
    0 < minstepfac < 1 || throw(ArgumentError("minstepfac must be in (0, 1)"))

    # Extract fields and set convergence flag
    cvg, p, r = false, m.pred, m.resp

    if isa(estimate_scale, Real)
        r.λ = estimate_scale
    elseif isa(estimate_scale, Symbol)
        if estimate_scale == :mad
            λ = mad(r.y; normalize=true)
            r.λ = λ
        end
    end

    # Initialize β, μ, and compute initial deviance
    # Compute beta update based on default response value
    setinitη!(m)

    ## TODO: scale/dispersion updating is not working
#    if m.fitdispersion
#        updateλ!(r)
#    end
    devold = deviance(m)
#    println(p.delbeta)
#    println(r.wrkresid)
#    println(r.wrkwt)
    installbeta!(p)

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
                updateμ!(r, linpred(p, f))
                dev = deviance(m)
            catch e
                isa(e, DomainError) ? (dev = Inf) : rethrow(e)
            end
        end
        installbeta!(p, f)

        ## TODO: iterative scale/dispersion updating is not working
        ## it is not clear if it can work at all
        #if m.fitdispersion
        #    updateλ!(r)
        #end

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
    m.fitted = true
    m
end

