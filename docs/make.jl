using Documenter
using GLM, StatsBase, SparseArrays, LinearAlgebra
using RobustModels

prettyurls = get(ENV, "CI", nothing) == "true"

makedocs(
    format = Documenter.HTML(prettyurls=prettyurls),
    sitename = "RobustModels",
    modules = [RobustModels, GLM, StatsBase],
    pages = [
        "Home" => "index.md",
        "manual.md",
        "examples.md",
        "api.md",
    ],
#    doctest = :fix,
    debug = false,
)

deploydocs(
    repo = "github.com/getzze/RobustModels.jl"
)
