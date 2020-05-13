using RobustModels
using Test
using RDatasets: dataset
using SparseArrays: SparseMatrixCSC
using StatsModels: @formula, coef


data = dataset("robustbase", "Animals2")
form = @formula(Brain ~ 1 + Body)

@testset "Quantile regression: high-level function" begin
    X = hcat(ones(size(data, 1)), data.Body)
    y = data.Brain

    τs = range(0.1, 0.9, step=0.1)
    βs = hcat(map(τ->RobustModels.interiormethod(X, y, τ)[1], τs)...)
    println("Coefficients: $(vcat(τs', βs))")
    @test size(βs) == (size(X, 2), length(τs))
end


@testset "Quantile regression: fit method" begin
    y = data.Brain
    X = hcat(ones(size(data, 1)), data.Body)
    sX = SparseMatrixCSC(X)

    τ = 0.5
    # Formula, dense and sparse entry  and methods :cg and :chol
    @testset "Argument type: $(typeof(A))" for (A, b) in ((form, data), (X, y), (sX, y))
        m1 = fit(QuantileRegression, A, b; quantile=τ, verbose=false)
        m2 = quantreg(A, b; quantile=τ, verbose=false)
        @test all(coef(m1) .== coef(m2))

        # refit
        β = copy(coef(m2))
        refit!(m2, y; quantile=τ, verbose=false)
        @test all(coef(m2) .== β)
    end
end

@testset "Quantile regression: different quantiles" begin
    τs = range(0.1, 0.9, step=0.1)
    m2 = fit(QuantileRegression, form, data; quantile=0.5, verbose=false)

    @testset "$(τ) quantile" for τ in τs
        m1 = fit(QuantileRegression, form, data; quantile=τ, verbose=false)
        println(m1)
        β = coef(m1)
        res = residuals(m1)
        ## The quantile regression line exactly passes through p points, with p number of columns of X.
        @test count(iszero, res) == length(β)

        # refit with new quantile
        refit!(m2; quantile=τ)
        @test all(coef(m1) .== coef(m2))
    end
end


