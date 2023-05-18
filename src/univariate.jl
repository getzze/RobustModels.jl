


#import StatsBase: mean, var, std, sem, mean_and_std, mean_and_var
mean_and_sem(x) = (m = mean(x); s = sem(x); (m, s))

function check_l1loss(est)
    if isa(loss(est), L1Loss)
        mess = "variance is not well defined with L1Loss, use another loss or use `mad`."
        throw(ArgumentError(mess))
    end
end

function _model_from_univariate(
    est::AbstractMEstimator,
    x::AbstractVector;
    method=:cg, # use :cg instead of :chol to avoid PosDefException
    dims=nothing, # explicit, so it is not passed to `rlm`
    kwargs...,
)

    check_l1loss(est)
    n = size(x, 1)
    X = ones(eltype(x), (n, 1))
    return m = rlm(X, x, est; method=method, kwargs...)
end

function _mean(est::AbstractMEstimator, x::AbstractVector; kwargs...)
    m = _model_from_univariate(est, x; kwargs...)
    return coef(m)[1]
end

function _std(est::AbstractMEstimator, x::AbstractVector; corrected::Bool=true, kwargs...)
    m = _model_from_univariate(est, x; kwargs...)
    n = dof_residual(m)
    if !corrected
        n += 1
    end
    return dispersion(m.resp, n, false)
end

function _var(est::AbstractMEstimator, x::AbstractVector; corrected::Bool=true, kwargs...)
    m = _model_from_univariate(est, x; kwargs...)
    n = dof_residual(m)
    if !corrected
        n += 1
    end
    return dispersion(m.resp, n, true)
end

function _sem(est::AbstractMEstimator, x::AbstractVector; kwargs...)
    m = _model_from_univariate(est, x; kwargs...)
    return stderror(m)[1]
end

function _mean_and_std(
    est::AbstractMEstimator, x::AbstractVector; corrected::Bool=true, kwargs...
)
    m = _model_from_univariate(est, x; kwargs...)
    n = dof_residual(m)
    if !corrected
        n += 1
    end
    return coef(m)[1], dispersion(m.resp, n, false)
end

function _mean_and_var(
    est::AbstractMEstimator, x::AbstractVector; corrected::Bool=true, kwargs...
)
    m = _model_from_univariate(est, x; kwargs...)
    n = dof_residual(m)
    if !corrected
        n += 1
    end
    return coef(m)[1], dispersion(m.resp, n, true)
end

function _mean_and_sem(est::AbstractMEstimator, x::AbstractVector; kwargs...)
    m = _model_from_univariate(est, x; kwargs...)
    return coef(m)[1], stderror(m)[1]
end


## For arrays, along specific dimension with dims keyword argument

#Dims = Union{Colon,T,Tuple{Vararg{T}}} where T<:Int

function compatdims(array_ndims, dims)
    if isa(dims, Int)
        if 0 < dims <= array_ndims
            return dims
        end
    elseif isa(dims, Tuple)
        newdims = Tuple(d for d in dims if 0 < d <= array_ndims)
        if length(newdims) > 0
            return newdims
        end
    end
    return nothing
end

for fun in (:mean, :std, :var, :sem)
    _fun = Symbol("_$(fun)")
    @eval begin
        # `where Dims` to allow Colon
        function StatsBase.$(fun)(
            est::AbstractMEstimator, x::AbstractArray; dims::Dims=:, kwargs...
        ) where {Dims}
            check_l1loss(est)
            if dims === (:)
                return $(_fun)(est, vec(x); kwargs...)
            else
                dims = compatdims(ndims(x), dims)
                if isnothing(dims)
                    return x .* $(fun)(one(eltype(x)))
                else
                    return mapslices(r -> $(_fun)(est, vec(r); kwargs...), x; dims=dims)
                end
            end
        end
    end
end

for fun in (:mean_and_std, :mean_and_var, :mean_and_sem)
    _fun = Symbol("_$(fun)")
    @eval begin
        # `where Dims` to allow Colon
        function $(fun)(
            est::AbstractMEstimator, x::AbstractArray; dims::Dims=:, kwargs...
        ) where {Dims}
            check_l1loss(est)
            if dims === (:)
                ret = $(_fun)(est, vec(x); kwargs...)
                return ret
            else
                dims = compatdims(ndims(x), dims)
                if isnothing(dims)
                    m = mean(one(eltype(x)))
                    s = std(one(eltype(x)))  # NaN or NaN32
                    return (x * m, x * s)
                else
                    ret = mapslices(r -> $(_fun)(est, vec(r); kwargs...), x; dims=dims)
                    return (first.(ret), last.(ret))
                end
            end
        end
    end
end


## For iterators
for fun in (:mean, :std, :var, :sem, :mean_and_std, :mean_and_var, :mean_and_sem)
    _fun = Symbol("_$(fun)")
    @eval function $(fun)(est::AbstractMEstimator, itr; kwargs...)
        return $(_fun)(est, collect(itr); kwargs...)
    end
end
