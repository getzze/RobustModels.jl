using Documenter
using GLM, StatsBase, StatsAPI, SparseArrays, LinearAlgebra
using RobustModels

DocMeta.setdocmeta!(
    RobustModels,
    :DocTestSetup,
    :(using RobustModels, StatsBase, GLM, StatsAPI);
    recursive=true,
)
DocMeta.setdocmeta!(StatsBase, :DocTestSetup, :(using StatsBase); recursive=true)
DocMeta.setdocmeta!(GLM, :DocTestSetup, :(using GLM, StatsBase); recursive=true)
DocMeta.setdocmeta!(StatsAPI, :DocTestSetup, :(using StatsAPI); recursive=true)

prettyurls = get(ENV, "CI", "false") == "true"

makedocs(;
    modules=[RobustModels, GLM, StatsBase, StatsAPI],
    sitename="RobustModels",
    authors="Bertrand Lacoste <bertrand.lacoste@gmail.com>",
    repo="https://github.com/getzze/RobustModels.jl/blob/{commit}{path}#{line}",
    format=Documenter.HTML(;
        prettyurls=prettyurls,
        canonical="https://getzze.github.io/RobustModels.jl",
        assets=["assets/favicon.ico"],
    ),
    #! format: off
    pages=[
        "Home" => "index.md",
        "manual.md",
        "examples.md",
        "api.md",
    ],
    #! format: on
    ## Uncomment the line below to re-generate doctest outputs
    # doctest = :fix,
    warnonly=[:missing_docs],
    debug=false,
)

deploydocs(; devbranch="main", repo="github.com/getzze/RobustModels.jl")
