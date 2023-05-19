using Documenter
using GLM, StatsBase, SparseArrays, LinearAlgebra
using RobustModels

DocMeta.setdocmeta!(
    RobustModels, :DocTestSetup, :(using RobustModels, StatsBase, GLM); recursive=true
)

prettyurls = get(ENV, "CI", "false") == "true"

makedocs(;
    modules=[RobustModels, GLM, StatsBase],
    sitename="RobustModels",
    authors="Bertrand Lacoste <bertrand.lacoste@gmail.com>",
    repo="https://github.com/getzze/RobustModels.jl/blob/{commit}{path}#{line}",
    format=Documenter.HTML(;
        prettyurls=prettyurls,
        canonical="https://getzze.github.io/RobustModels.jl",
        assets=["assets/favicon.ico"],
    ),
    pages=[
        "Home" => "index.md", 
        "manual.md", 
        "examples.md", 
        "api.md",
    ],
    # doctest = :fix,
    debug=false,
)

deploydocs(; 
    devbranch="main", 
    repo="github.com/getzze/RobustModels.jl",
)
