


function show(io::IO, obj::ConvergenceFailed)
    println(io, "failed to find a solution: $(obj.reason)")
end


"""
    RobustLinearModel

Robust linear model representation

Structures for response and model of RobustLinearModel
   min ∑ᵢ ½ ρ(rᵢ²)

With scaling factor C:
    ^ρ(r²) = C² ρ(r²/C²)

"""
mutable struct RobustLinResp{T<:AbstractFloat, V<:AbstractVector{T}, M<:Estimator} <: RobustResp{T}
    "`est`: Estimator instance"
    est::M
    "`y`: response vector"
    y::V
    "`μ`: mean response"
    μ::V
    "`offset:` offset added to `Xβ` to form `μ`. Can be of length 0"
    offset::V
    "`wts:` prior case weights.  Can be of length 0."
    wts::V
    "`σ`: the estimation of the scale or dispersion"
    σ::T
    "`devresid`: the deviance residuals"
    devresid::V
    "`wrkwt`: working case weights for the Iteratively Reweighted Least Squares (IRLS) algorithm"
    wrkwt::V
    "`wrkres`: working residuals for IRLS"
    wrkres::V
    "`wrkscaledres`: scaled residuals for IRLS"
    wrkscaledres::V

    function RobustLinResp{T, V, M}(est::M, y::V, μ::V, off::V, wts::V, σ::AbstractFloat=1) where {V<:AbstractVector{T}, M<:Estimator} where {T<:AbstractFloat}
        n = length(y)
        length(μ) == n || error("mismatched lengths of μ and y")
        ll = length(off)
        ll == 0 || ll == n || error("length of offset is $ll, must be $n or 0")
        ll = length(wts)
        ll == 0 || ll == n || error("length of wts is $ll, must be $n or 0")
        σ > 0 || error("scale/dispersion must be positive")
        new{T, V, M}(est, y, μ, off, wts, σ, similar(y), similar(y), similar(y), similar(y))
    end
end

"""
    RobustLinResp(est::M, y::V, off::V, wts::V, σ::AbstractFloat=1) where {V<:FP, M<:Estimator}
Create a response structure by copying the value of y to the mean μ.

"""
function RobustLinResp(est::M, y::V, off::V, wts::V, σ::AbstractFloat=1) where {V<:FPVector, M<:Estimator}
    T = eltype(y)
    r = RobustLinResp{T,V,M}(est, y, similar(y), off, wts, σ)
    initresp!(r)
    return r
end

function initresp!(r::RobustResp{T}) where {T}
    # Set μ to 0
    μ0 = 0
    fill!(r.μ, μ0)

    # Set residual (without offset)
    broadcast!(-, r.wrkres, r.y, r.μ)

    # Set working weights to the data weights
    initwt = if !isempty(r.wts); r.wts else ones(T, size(r.y)) end
    copyto!(r.wrkwt, initwt)
end


function Base.getproperty(r::RobustLinResp, s::Symbol)
    if s ∈ (:mu, :η, :eta)
        r.μ
    elseif s ∈ (:sigma, :scale)
        r.σ
    else
        getfield(r, s)
    end
end

function dispersion(r::RobustResp, dof_residual::Int=nobs(r), sqr::Bool=false, robust::Bool=true)
    wrkwt, wrkres, wrkscaledres = r.wrkwt, r.wrkres, r.wrkscaledres
    if robust
        s = (r.σ)^2*sum(i -> wrkwt[i] * abs2(wrkscaledres[i]), eachindex(wrkwt, wrkres)) / dof_residual
    else
        s = sum(i -> wrkwt[i] * abs2(wrkres[i]), eachindex(wrkwt, wrkres)) / dof_residual
    end
    if sqr; s else sqrt(s) end
end
#dispersion(r::RobustResp) = r.scale


"""
    location_variance(r::RobustLinResp, sqr::Bool = false)
Compute the part of the variance of the coefficients β that is due to the encertainty from the location.
If `sqr` is false, return the standard deviation instead.

From Maronna et al., Robust Statistics: Theory and Methods, Equation 4.49
"""
function location_variance(r::RobustLinResp, dof_residual::Int=(nobs(r)-1), sqr::Bool=false)
    println(typeof(r))
    psi(x)    = estimator_psi(r.est, x)
    psider(x) = estimator_psider(r.est, x)
    
    if isa(r.est, UnionL1)
        @warn "coefficient variance is not well defined for L1Estimator."
        return Inf
    end

    v = if isempty(r.wts)
        v = mean( (psi.(r.wrkscaledres)).^2 )
        v /= ( mean(psider.(r.wrkscaledres)) )^2
    else
        wts = weights(r.wts)
        v = mean( (psi.(r.wrkscaledres)).^2, wts )
        v /= ( mean(psider.(r.wrkscaledres), wts) )^2
    end
    v *= r.σ^2
    v *= (nobs(r)/dof_residual)
    if sqr; v else sqrt(v) end
end

deviance(r::RobustResp) = sum(r.devresid)

function nulldeviance(r::RobustResp)
    ## TODO: take wts into account
    y, σ = r.y, r.σ
    μi = mean(y)

    dev = 0
    @inbounds for i in eachindex(y)
        dev += estimator_rho(r.est, (y[i] - μi)/σ)
    end
    dev
end

## TODO: define correctly the loglikelihood of the full model
fullloglikelihood(r::RobustLinResp) = r.scale * log(estimator_norm(r.est))
loglikelihood(r::RobustResp) = fullloglikelihood(r) - deviance(r)/2
nullloglikelihood(r::RobustResp) = fullloglikelihood(r) - nulldeviance(r)/2

response(r::RobustResp) = r.y

Estimator(r::RobustResp) = r.est

weights(r::RobustResp{T}) where T<:AbstractFloat = if isempty(r.wts); weights(ones(T, length(r.y))) else weights(r.wts) end

fitted(r::RobustLinResp) = r.μ

residuals(r::RobustLinResp) = r.wrkres

workingweights(r::RobustLinResp) = r.wrkwt

"""
    nobs(obj::RobustResp)
For linear and generalized linear models, returns the number of elements of the response
"""
function nobs(r::RobustResp{T}) where T
    if !isempty(r.wts)
        ## Suppose that the weights are probability weights
        count(!iszero, r.wts)
#        convert(T, sum(r.wts))
    else
        size(r.y, 1)
    end
end

"""
    setμ!{T<:FPVector}(r::RobustResp{T}, μ::T)
Update the mean to the given value `μ`.
If no value is provided, μ is set to the mean response `mean(r.y)`.
"""
function setμ! end

function setμ!(r::RobustLinResp, μ::T) where {T<:Real}
    fill!(r.μ, μ)

    # Subtract the offset because it will be added in the updateres! step
    if size(r.offset, 1) == size(r.y, 1)
        @inbounds @simd for i = eachindex(r.μ, r.offset)
            r.μ[i] -= r.offset[i]
        end
    end
end

function setμ!(r::RobustLinResp, μ::V) where {V<:FPVector}
    copyto!(r.μ, μ)

    # Subtract the offset because it will be added in the updateres! step
    if size(r.offset, 1) == size(r.y, 1)
        @inbounds @simd for i = eachindex(r.μ, r.offset)
            r.μ[i] -= r.offset[i]
        end
    end
end

function setμ!(r::RobustLinResp)
    if !isempty(r.wts)
        # Weighted mean
        m = mean(r.y, weights(r.wts))
    else
        m = mean(r.y)
    end
    setμ!(r, m)
end


"""
    updateres!{T<:FPVector}(r::RobustResp{T})
Update the mean, working weights, working residuals and deviance residuals, of the response `r`.
"""
function updateres! end

function updateres!(r::RobustLinResp{T, V, M}) where {T<:AbstractFloat, V<:FPVector, M<:Estimator}
    ## Add offset to the linear predictor, if offset is defined
    ## and copy the linear predictor to the mean
    if length(r.offset) != 0
        broadcast!(+, r.μ, r.μ, r.offset)
    end

    y, μ, σ, devresid = r.y, r.μ, r.σ, r.devresid
    wrkres, wrkscaledres, wrkwt = r.wrkres, r.wrkscaledres, r.wrkwt

    @inbounds for i in eachindex(y, μ, wrkres, wrkscaledres, wrkwt, devresid)
        wrkres[i] = y[i] - μ[i]
        wrkscaledres[i] = wrkres[i]/σ
        wrkwt[i] = estimator_weight(r.est, wrkscaledres[i])
        devresid[i] = estimator_rho(r.est, wrkscaledres[i])
    end

    ## Multiply by the observation weights
    if !isempty(r.wts)
        broadcast!(*, r.devresid, r.devresid, r.wts)
        broadcast!(*, r.wrkwt, r.wrkwt, r.wts)
    end

end

function optimscale(r::RobustLinResp; sigma0::Union{Nothing, AbstractFloat}=nothing, verbose::Bool=false)
    est, res = r.est, r.wrkres
    σ0 = ifelse( isnothing(sigma0), r.σ, sigma0 )

    if !isbounded(est)
        @warn "scale/dispersion is not changed because the estimator ($(est)) does not allow scale estimation, a bounded estimator should be used, like TukeyEstimator."
        return σ0
    end

    verbose && print("Update scale: $(σ0)")
    σ = scale_estimate(est, res; σ0=σ0, wts=r.wts, use_reciprocal=true)
    if σ <= 0
        println("  ->  error")
        throw(ConvergenceFailed("the resulting scale is non-positive"))
    end
    verbose && println("  ->  $(σ)")
    σ
end

function updatescale!(r::RobustLinResp, method::Symbol; factor::AbstractFloat=1.0)
    res = r.wrkres
    allowed_methods = [:extrema, :mad, :Sestimate, :highbreakpoint]
    if !(method in allowed_methods)
        @warn "scale/dispersion is not changed because `method` argument must be in $(allowed_methods): $(method)"
        return r
    elseif method == :extrema
        r.σ = -(-(extrema(res)...))/2 * factor
    elseif method == :mad
        if isempty(r.wts)
            r.σ = abs(factor*mad(res; normalize=true))
        else
            r.σ = abs(factor*mad(r.wts .* res; normalize=true))
        end
    elseif method == :Sestimate
        Mest = r.est
        if !isbounded(Mest)
            @warn "the current estimator is not bounded, use TukeyEstimator to estimate the scale."
        end
        Sest = SEstimator(Mest)
        r.σ = scale_estimate(Sest, res; σ0=r.σ, wts=r.wts, use_reciprocal=true)
    elseif method == :highbreakpoint
        K = 4.5
        if isempty(r.wts)
            r.σ = sqrt(sum( res.^2 ./ (1 .+ (res./(K*r.σ)).^2))/size(res, 1))
        else
            wres = r.wts .* res
            N = sum(r.wts)
            r.σ = sqrt(sum( wres.^2 ./ (1 .+ (wres./(K*r.σ)).^2))/N)
        end
    end
    @debug "new robust scale/dispersion for $(typeof(r.est)): $(r.σ)"
    r
end
####################




