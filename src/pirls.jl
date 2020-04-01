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
    "`λ`: the estimation of the scale or dispersion"
    λ::T
    "`devresid`: the deviance residuals"
    devresid::V
    "`wrkwt`: working case weights for the Iteratively Reweighted Least Squares (IRLS) algorithm"
    wrkwt::V
    "`wrkresid`: working residuals for IRLS"
    wrkresid::V
    "`wrkscaledres`: scaled residuals for IRLS"
    wrkscaledres::V 
     
    function RobustLinResp{T, V, M}(est::M, y::V, μ::V, off::V, wts::V, λ::AbstractFloat=1) where {V<:AbstractVector{T}, M<:Estimator} where {T<:AbstractFloat}
        n = length(y)
        length(μ) == n || error("mismatched lengths of μ and y")
        ll = length(off)
        ll == 0 || ll == n || error("length of offset is $ll, must be $n or 0")
        ll = length(wts)
        ll == 0 || ll == n || error("length of wts is $ll, must be $n or 0")
        λ > 0 || error("scale/dispersion must be positive")
        new{T, V, M}(est, y, μ, off, wts, λ, similar(y), similar(y), similar(y), similar(y))
    end
end

"""
    RobustLinResp(est::M, y::V, off::V, wts::V, λ::AbstractFloat=1) where {V<:FP, M<:Estimator}
Create a response structure by copying the value of y to the mean μ.
    
"""
function RobustLinResp(est::M, y::V, off::V, wts::V, λ::AbstractFloat=1) where {V<:FPVector, M<:Estimator}
    r = RobustLinResp{eltype(y),V,M}(est, y, similar(y), off, wts, λ)
    setμ!(r)
    updateres!(r)
    return r
end

function Base.getproperty(r::RobustLinResp, s::Symbol)
    if s ∈ (:mu, :η, :eta)
        r.μ
    elseif s == :lambda
        r.λ
    else
        getfield(r, s)
    end
end

function dispersion(r::RobustResp, dof_residual::Int = 1, sqr::Bool = false)
    wrkwt, wrkresid = r.wrkwt, r.wrkresid
    s = sum(i -> wrkwt[i] * abs2(wrkresid[i]), eachindex(wrkwt, wrkresid)) / dof_residual
    if sqr; s else sqrt(s) end
end
#dispersion(r::RobustResp) = r.scale


deviance(r::RobustResp) = sum(r.devresid)

function nulldeviance(r::RobustResp)
    y, λ = r.y, r.λ
    μi = mean(y)

    dev = 0
    @inbounds for i in eachindex(y)
        dev += estimator_rho(r.est, (y[i] - μi)/λ)
    end
    dev
end


## TODO: add scale, but assure that deviance is positive
#deviance(r::RobustResp) = sum(r.devresid) + 2*size(r.devresid, 1)*log(r.scale) - r.nulldev
#function nulldeviance(r::RobustResp{T, M}) where {T<:FPVector, M<:Estimator}
#    y, rscale2 = r.y, r.wrkscale2
#    μi = mean(y)

#    dev = 0
#    @inbounds for i in eachindex(y)
#        wrkres2 = (y[i] - μi)^2/rscale2
#        dev += rscale2*estimator_rho(M(), wrkres2)
#    end
#end

## TODO: define correctly the loglikelihood of the full model
fullloglikelihood(r::RobustResp) = 0
loglikelihood(r::RobustResp) = fullloglikelihood(r) - deviance(r)/2

response(r::RobustResp) = r.y

Estimator(r::RobustResp) = r.est

weights(r::RobustResp) = weights(r.wts)

fitted(r::RobustLinResp) = r.μ

residuals(r::RobustLinResp) = r.wrkresid

"""
    nobs(obj::RobustResp)
For linear and generalized linear models, returns the number of elements of the response
"""
function nobs(r::RobustResp{T}) where T
    if !isempty(r.wts)
        convert(T, count(r.wts .!= 0))
#        convert(T, sum(r.wts))
    else
        convert(T, size(r.y, 1))
    end
end

"""
    setμ!{T<:FPVector}(r::RobustResp{T}, μ::T)
Update the mean to the given value `μ`.
If no value is provided, μ is set to the mean response `mean(r.y)`.
"""
function setμ! end

function setμ!(r::RobustLinResp{T, V, M}, μ::T) where {T<:AbstractFloat, V, M}
    r.μ .= μ[:]
end

function setμ!(r::RobustLinResp)
    if !isempty(r.wts)
        # Weighted mean
        m = mean(r.y, weights(r.wts))
    else
        m = mean(r.y)
    end
    r.μ .= m
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

    y, μ, λ, devresid = r.y, r.μ, r.λ, r.devresid
    wrkresid, wrkscaledres, wrkwt = r.wrkresid, r.wrkscaledres, r.wrkwt
#    λ² = λ^2

    @inbounds for i in eachindex(y, μ, wrkresid, wrkscaledres, wrkwt, devresid)
        wrkresid[i] = y[i] - μ[i]
        wrkscaledres[i] = wrkresid[i]/λ
        wrkwt[i] = estimator_weight(r.est, wrkscaledres[i])
#        devresid[i] = λ²*estimator_rho(r.est, wrkscaledres[i])
        devresid[i] = estimator_rho(r.est, wrkscaledres[i])
    end
    
    ## Multiply by the observation weights
    if !isempty(r.wts)
        broadcast!(*, r.devresid, r.devresid, r.wts)
        broadcast!(*, r.wrkwt, r.wrkwt, r.wts)
    end

end


function updateλ!(r::RobustLinResp{T, V, M}, factor::T=1, method::Symbol=:mad, sestimator=CauchyEstimator) where {T<:AbstractFloat, V<:FPVector, M<:Estimator}
    res = r.wrkresid
    allowed_methods = [:mad, :Sestimate, :largebreakpoint]
    if !(method in allowed_methods)
        @warn "scale/dispersion is not changed because `method` argument must be in $(allowed_methods): $(method)"
        return r
    elseif method == :mad
        if isempty(r.wts)
            r.λ = abs(factor*StatsBase.mad(res; normalize=true))
        else
            r.λ = abs(factor*StatsBase.mad(r.wts .* res; normalize=true))
        end
    elseif method == :Sestimate
        chi = SEstimator(sestimator)
        if isempty(r.wts)
            N = size(res, 1)
            r.λ = find_zero(s->sum(chi.(res ./ s))/N - 1, r.λ, Order1())
        else
            N = sum(r.wts)
            r.λ = find_zero(s->sum(r.wts .* chi.(res ./ s))/N - 1, r.λ, Order1())
        end
    elseif method == :largebreakpoint
        K = 4.5
        if isempty(r.wts)
            r.λ = sqrt(sum( res.^2 ./ (1 .+ (res./(K*r.λ)).^2))/size(res, 1))
        else
            wres = r.wts .* res
            N = sum(r.wts)
            r.λ = sqrt(sum( wres.^2 ./ (1 .+ (wres./(K*r.λ)).^2))/N)
        end
    end
    @debug "new robust scale/dispersion for $(typeof(r.est)): $(r.λ)"
    r
end
####################




