using LinearAlgebra: cholesky!, qr!

function get_pkg_version(m::Module)
    toml = Pkg.TOML.parsefile(joinpath(pkgdir(m), "Project.toml"))
    return VersionNumber(toml["version"])
end


## Compatibility layers

# https://github.com/JuliaStats/GLM.jl/pull/459
@static if VERSION < v"1.8.0-DEV.1139"
    pivoted_cholesky!(A; kwargs...) = cholesky!(A, Val(true); kwargs...)
else
    using LinearAlgebra: RowMaximum
    pivoted_cholesky!(A; kwargs...) = cholesky!(A, RowMaximum(); kwargs...)
end

@static if VERSION < v"1.7.0"
    pivoted_qr!(A; kwargs...) = qr!(A, Val(true); kwargs...)
else
    using LinearAlgebra: ColumnNorm
    pivoted_qr!(A; kwargs...) = qr!(A, ColumnNorm(); kwargs...)
end
