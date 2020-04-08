using RobustModels
using Test
using RDatasets
using StatsModels: @formula, coef


data = dataset("robustbase", "Animals2")
form = @formula(Brain ~ 1 + Body)

@testset "Quantile regression: high-level function" begin
    X = hcat(ones(size(data, 1)), data.Body)
    y = data.Brain

    τs = range(0.1, 0.9, step=0.1)
    βs = hcat(map(τ->interiormethod(X, y, τ)[1], τs)...)
    println("Coefficients: $(vcat(τs', βs))")
    @test size(βs) == (size(X, 2), length(τs))
end

@testset "Quantile regression: fit" begin
    τs = range(0.1, 0.9, step=0.1)
    for τ in τs
        m = fit(QuantileRegression, form, data; quantile=τ, verbose=false)
        println("Quantile $τ:\n$(m)")
        β = coef(m)
        res = residuals(m)
        ## The quantile regression line exactly passes through p points, with p number of columns of X.
        @test count(iszero, res) == length(β)
    end
end


