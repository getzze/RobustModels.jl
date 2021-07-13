


#import StatsBase: mean, var, std, sem, mean_and_std, mean_and_var

function check_l1loss(est)
    if isa(loss(est), L1Loss)
        throw(ArgumentError("variance is not well defined with L1Loss, use another loss or use `mad`."))
    end
end

function _model_from_univariate(est::AbstractEstimator, x::AbstractVector; method=:cg, kwargs...)
    check_l1loss(est)
    X = ones(eltype(x), (size(x, 1), 1))
    m = rlm(X, x, est; method, kwargs...)
end

function mean(est::AbstractEstimator, x::AbstractVector; σ0=:mad, kwargs...)
    m = _model_from_univariate(est, x; σ0, kwargs...)
    return coef(m)[1]
end

function std(est::AbstractEstimator, x::AbstractVector; corrected::Bool=true, σ0=:mad, kwargs...)
    m = _model_from_univariate(est, x; σ0, kwargs...)
    n = dof_residual(m)
    if !corrected; n += 1 end
    return dispersion(m.resp, n, false)
end

function var(est::AbstractEstimator, x::AbstractVector; corrected::Bool=true, σ0=:mad, kwargs...)
    m = _model_from_univariate(est, x; σ0, kwargs...)
    n = dof_residual(m)
    if !corrected; n += 1 end
    return dispersion(m.resp, n, true)
end

function sem(est::AbstractEstimator, x::AbstractVector; σ0=:mad, kwargs...)
    m = _model_from_univariate(est, x; σ0, kwargs...)
    return stderror(m)[1]
end

function mean_and_std(est::AbstractEstimator, x::AbstractVector; corrected::Bool=true, σ0=:mad, kwargs...)
    m = _model_from_univariate(est, x; σ0, kwargs...)
    n = dof_residual(m)
    if !corrected; n += 1 end
    return coef(m)[1], dispersion(m.resp, n, false)
end

function mean_and_var(est::AbstractEstimator, x::AbstractVector; corrected::Bool=true, σ0=:mad, kwargs...)
    m = _model_from_univariate(est, x; σ0, kwargs...)
    n = dof_residual(m)
    if !corrected; n += 1 end
    return coef(m)[1], dispersion(m.resp, n, true)
end

function mean_and_sem(est::AbstractEstimator, x::AbstractVector; σ0=:mad, kwargs...)
    m = _model_from_univariate(est, x; σ0, kwargs...)
    return coef(m)[1], stderror(m)[1]
end


## For arrays, along specific dimension with dims keyword argument

for fun in (:mean, :std, :var, :sem)
    @eval begin
        # `where Dims` to allow Colon
        function $(fun)(est::AbstractEstimator, x::AbstractArray; dims::Dims=:, kwargs...) where Dims
            check_l1loss(est)
            if dims === (:)
                $(fun)(est, vec(x); kwargs...)
            else
                mapslices(r->$(fun)(est, r; kwargs...), x; dims)
            end
        end
    end
end

for fun in (:mean_and_std, :mean_and_var, :mean_and_sem)
    @eval begin
        # `where Dims` to allow Colon
        function $(fun)(est::AbstractEstimator, x::AbstractArray; dims::Dims=:, kwargs...) where Dims
            check_l1loss(est)
            if dims === (:)
                ret = $(fun)(est, vec(x); kwargs...)
            else
                ret = mapslices(r->$(fun)(est, r; kwargs...), x; dims)
            end
            (first.(ret), last.(ret))
        end
    end
end


## For iterators
for fun in (:mean, :std, :var, :sem, :mean_and_std, :mean_and_var, :mean_and_sem)
    @eval $(fun)(est::AbstractEstimator, itr; kwargs...) = $(fun)(est, collect(itr); kwargs...)
end
