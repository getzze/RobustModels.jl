using Documenter
using GLM, StatsBase, SparseArrays, LinearAlgebra
using RobustModels

DocMeta.setdocmeta!(RobustModels, :DocTestSetup, :(using RobustModels); recursive=true)

prettyurls = get(ENV, "CI", "false") == "true"

makedocs(
    sitename = "RobustModels",
    modules = [RobustModels, GLM, StatsBase],
    authors="Bertrand Lacoste <bertrand.lacoste@gmail.com>",
    repo="https://github.com/getzze/RobustModels.jl/blob/{commit}{path}#{line}",
    format = Documenter.HTML(;
        prettyurls=prettyurls,
        canonical="https://getzze.github.io/RobustModels.jl",
        assets=String[],
    ),
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
