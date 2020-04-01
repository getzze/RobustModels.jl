using RobustModels
using Test
using RDatasets
using StatsModels: @formula, coef


data = dataset("robustbase", "Animals2")

@testset "Quantile regression" begin
    X = hcat(ones(size(data, 1)), data.Body)
    y = data.Brain

    βs = hcat(map(τ->interiormethod(X, y, τ)[1], range(0.05, 0.95, step=0.05))...)
    println(size(βs))
    println(βs)
    @test size(βs) == (size(X, 2), 19)
end




