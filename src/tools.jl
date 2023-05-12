
_missing_omit(x::AbstractArray{T}) where T = copyto!(similar(x, nonmissingtype(T)), x)

function StatsModels.missing_omit(X::AbstractMatrix, y::AbstractVector)
    X_ismissing = eltype(X) >: Missing
    y_ismissing = eltype(y) >: Missing

    # Check that X and y have the same number of observations
    n = size(X, 1)
    if n != size(y, 1)
        throw(DimensionMismatch("number of rows in X and y must match"))
    end

    nonmissings = trues(n)
    if X_ismissing
        for j in axes(X, 2)
            nonmissings .&= .!ismissing.(X[:, j])
        end
    end
    if y_ismissing
        nonmissings .&= .!ismissing.(y)
    end

    if all(nonmissings)
        X_nonmissing = _missing_omit(X)
        y_nonmissing = _missing_omit(y)
    else
        rows = findall(nonmissings)
        y_nonmissing = _missing_omit(view(y, rows))
        X_nonmissing = _missing_omit(view(X, rows, :))
    end

    X_nonmissing, y_nonmissing, nonmissings
end


function _hasintercept(X::AbstractMatrix)
    return any(i -> all(==(1), view(X , :, i)), 1:size(X, 2))
end

function get_intercept_col(X::AbstractMatrix, f::Union{Nothing,FormulaTerm}=nothing)::Union{Nothing, Integer}
    if !isnothing(f) && hasintercept(f)
        return findfirst(isa.(f.rhs.terms, InterceptTerm))
    elseif isnothing(f)
        return findfirst(i->all(==(1), view(X, :, i)), 1:size(X, 2))
    end
    return nothing
end


######
##    TableRegressionModel methods
######

const ModelFrameType =
    Tuple{FormulaTerm,<:AbstractVector,<:AbstractMatrix,NamedTuple}

"""
    modelframe(f::FormulaTerm, data, contrasts::AbstractDict, ::Type{M}; kwargs...) where M

Returns a 4-Tuple with the formula, the response, the model matrix and a NamedTuple with the
extra columns specified by name as keyword arguments. The response, model matrix and extra columns
are extracted from the `data` Table using the formula `f`.

Adapted from GLM.jl
"""
function modelframe(
    f::FormulaTerm,
    data,
    contrasts::AbstractDict,
    dropmissing::Bool,
    ::Type{M};
    kwargs...,
)::ModelFrameType where {M<:AbstractRobustModel}
    # Check is a Table
    Tables.istable(data) ||
        throw(ArgumentError("expected data in a Table, got $(typeof(data))"))
    t = Tables.columntable(data)

    # Check columns exist
    cols = collect(termvars(f))
    msg = ""
    for col in cols
        msg *= checkcol(t, col)
        if msg != ""
            msg *= "\n"
        end
    end
    msg != "" && throw(ArgumentError("Error with formula term names.\n" * msg))
    for val in Base.values(kwargs)
        if isa(val, Symbol)
            msg = checkcol(t, val)
            msg != "" && throw(ArgumentError("Error with extra column name.\n" * msg))
            push!(cols, val)
        end
    end

    if dropmissing
        # Drop rows with missing values
        # Compat with VERSION < v"1.7.0"
        t, _ = missing_omit(NamedTuple{Tuple(cols)}(t))

    else
        # Check columns have no missing or complex values
        msg = ""
        for col in cols
            typ = eltype(t[col])
            if !(typ <: Real)
                msg *= "Column $(col) does not have only Real values: $(typ).\n"
            end
        end
        msg != "" && throw(ArgumentError(msg))
    end

    # Get formula, response and model matrix
    sch = schema(f, t, contrasts)
    f = apply_schema(f, sch, M)
    # response and model matrix
    ## Do not copy the arrays!
    y, X = modelcols(f, t)
    extra_vec = NamedTuple(
        var => (
            if isa(val, Symbol)
                t[val]
            elseif isnothing(val)
                similar(y, 0)
            else
                val
            end
        ) for (var, val) in pairs(kwargs)
    )

    return f, y, X, extra_vec
end

