


#import StatsBase: mean, var, std, sem, mean_and_std, mean_and_var
mean_and_sem(x) = (m=mean(x); s=sem(x); (m,s))

function check_l1loss(est)
    if isa(loss(est), L1Loss)
        throw(ArgumentError("variance is not well defined with L1Loss, use another loss or use `mad`."))
    end
end

function _model_from_univariate(est::AbstractEstimator, x::AbstractVector; method=:cg, dims=nothing, kwargs...)
    # explicitly name `dims` keyword, so it is not passed to `rlm`
    check_l1loss(est)
    X = ones(eltype(x), (size(x, 1), 1))
    m = rlm(X, x, est; method, kwargs...)
end

function _mean(est::AbstractEstimator, x::AbstractVector; σ0=:mad, kwargs...)
    m = _model_from_univariate(est, x; σ0, kwargs...)
    coef(m)[1]
end

function _std(est::AbstractEstimator, x::AbstractVector; corrected::Bool=true, σ0=:mad, kwargs...)
    m = _model_from_univariate(est, x; σ0, kwargs...)
    n = dof_residual(m)
    if !corrected; n += 1 end
    dispersion(m.resp, n, false)
end

function _var(est::AbstractEstimator, x::AbstractVector; corrected::Bool=true, σ0=:mad, kwargs...)
    m = _model_from_univariate(est, x; σ0, kwargs...)
    n = dof_residual(m)
    if !corrected; n += 1 end
    dispersion(m.resp, n, true)
end

function _sem(est::AbstractEstimator, x::AbstractVector; σ0=:mad, kwargs...)
    m = _model_from_univariate(est, x; σ0, kwargs...)
    stderror(m)[1]
end

function _mean_and_std(est::AbstractEstimator, x::AbstractVector; corrected::Bool=true, σ0=:mad, kwargs...)
    m = _model_from_univariate(est, x; σ0, kwargs...)
    n = dof_residual(m)
    if !corrected; n += 1 end
    coef(m)[1], dispersion(m.resp, n, false)
end

function _mean_and_var(est::AbstractEstimator, x::AbstractVector; corrected::Bool=true, σ0=:mad, kwargs...)
    m = _model_from_univariate(est, x; σ0, kwargs...)
    n = dof_residual(m)
    if !corrected; n += 1 end
    coef(m)[1], dispersion(m.resp, n, true)
end

function _mean_and_sem(est::AbstractEstimator, x::AbstractVector; σ0=:mad, kwargs...)
    m = _model_from_univariate(est, x; σ0, kwargs...)
    coef(m)[1], stderror(m)[1]
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
        function $(fun)(est::AbstractEstimator, x::AbstractArray; dims::Dims=:, kwargs...) where Dims
            check_l1loss(est)
            if dims === (:)
                return $(_fun)(est, vec(x); kwargs...)
            else
                dims = compatdims(ndims(x), dims)
                if isnothing(dims)
                    return x .* $(fun)(one(eltype(x)))
                else
                    return mapslices(r->$(_fun)(est, vec(r); kwargs...), x; dims)
                end
            end
        end
    end
end

for fun in (:mean_and_std, :mean_and_var, :mean_and_sem)
    _fun = Symbol("_$(fun)")
    @eval begin
        # `where Dims` to allow Colon
        function $(fun)(est::AbstractEstimator, x::AbstractArray; dims::Dims=:, kwargs...) where Dims
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
                    ret = mapslices(r->$(_fun)(est, vec(r); kwargs...), x; dims)
                    return (first.(ret), last.(ret))
                end
            end
        end
    end
end


## For iterators
for fun in (:mean, :std, :var, :sem, :mean_and_std, :mean_and_var, :mean_and_sem)
    _fun = Symbol("_$(fun)")
    @eval $(fun)(est::AbstractEstimator, itr; kwargs...) = $(_fun)(est, collect(itr); kwargs...)
end
