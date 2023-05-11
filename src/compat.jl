using LinearAlgebra: cholesky!

## Compatibility layers

# https://github.com/JuliaStats/GLM.jl/pull/459
@static if VERSION < v"1.8.0-DEV.1139"
    pivoted_cholesky!(A; kwargs...) = cholesky!(A, Val(true); kwargs...)
else
    pivoted_cholesky!(A; kwargs...) = cholesky!(A, RowMaximum(); kwargs...)
end
