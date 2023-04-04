

## Adapted from GLM.jl

function Base.getproperty(mm::RobustLinearModel, f::Symbol)
    if f == :model
        Base.depwarn("accessing the `model` field of RobustModels.jl models is deprecated, " *
                     "as they are no longer wrapped in a `TableRegressionModel` " *
                     "and can be used directly now", :getproperty)
        return mm
    elseif f == :mf
        Base.depwarn("accessing the `mf` field of RobustModels.jl models is deprecated, " *
                     "as they are no longer wrapped in a `TableRegressionModel`." *
                     "Use `hasformula(m)` to check if the model has a formula and " *
                     "then call `formula(m)` to access the model formula.", :getproperty)
        form = formula(mm)
        return ModelFrame{Nothing, typeof(mm)}(form, nothing, nothing, typeof(mm))
    else
        return getfield(mm, f)
    end
end
