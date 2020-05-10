using RobustModels
using Test
using RDatasets
using SparseArrays: SparseMatrixCSC
using StatsModels: @formula, coef
using StatsBase: mad
using GLM: LinearModel

data = dataset("robustbase", "Animals2")
form = @formula(Brain ~ 1 + Body)

λ = mad(data.Brain; normalize=true)
est1 = RobustModels.L2Estimator()
est2 = RobustModels.TukeyEstimator()

@testset "linear: L2 estimator" begin
    println("\n\t\u25CF Estimator: L2")
    y = data.Brain
    X = hcat(ones(size(data, 1)), data.Body)
    sX = SparseMatrixCSC(X)

    # OLS
    m1 = fit(LinearModel, form, data)
    println(m1)
    println(" lm              : ", coef(m1))

    # Formula, dense and sparse entry  and methods :cg and :chol
    for (A, b) in ((form, data), (X, y), (sX, y)), method in (:cg, :chol)
        name  = if A==form; "formula" elseif A==X; "dense  " else "sparse " end
        name *= if method==:cg; ",  cg" else ",chol" end
        m = fit(RobustLinearModel, A, b, est1; method=method, verbose=false, initial_scale_estimate=:mad)
        println("rlm($name): ", coef(m))
        println(m)
        @test all(isapprox.(coef(m1), coef(m); rtol=1.0e-5))
    end
end

@testset "linear: Robust estimators" begin
    m1 = fit(LinearModel, form, data)
    for name in ("L1", "Huber", "L1L2", "Fair", "Logcosh", "Arctan", "Cauchy", "Geman", "Welsch", "Tukey")
        println("\n\t\u25CF Estimator: $(name)")
        est = getproperty(RobustModels, Symbol(name * "Estimator"))()
        m2 = fit(RobustLinearModel, form, data, est; method=:cg, verbose=false, initial_scale_estimate=:mad)
        m3 = fit(RobustLinearModel, form, data, est; method=:chol, initial_scale_estimate=:mad)
#        m4 = fit(RobustLinearModel, form, data, est; method=:qr, verbose=true, initial_scale_estimate=:mad)
        println("\n\t\u25CF Estimator: $(name)")
        println(" lm      : ", coef(m1))
        println("rlm(cg)  : ", coef(m2))
        println("rlm(chol): ", coef(m3))
#        println("rlm(qr)  : ", coef(m4))
        if name != "L1"
            @test all(isapprox.(coef(m2), coef(m3); rtol=1.0e-5))
#            @test all(isapprox.(coef(m3), coef(m4); rtol=1.0e-5))
        end
    end
end


@testset "linear: S-estimator" begin
    println("\n\t\u25CF Estimator type: S-estimators")
    m1 = fit(LinearModel, form, data)
    println(" lm             : ", coef(m1))
    for name in ("Huber", "L1L2", "Fair", "Logcosh", "Arctan", "Cauchy")
        Mest = getproperty(RobustModels, Symbol(name * "Estimator"))()
        @test_throws ErrorException fit(RobustLinearModel, form, data, Mest; method=:cg, verbose=false, initial_scale_estimate=:mad, kind=:Sestimate)
    end

    for name in ("Geman", "Welsch", "Tukey")
        Mest = getproperty(RobustModels, Symbol(name * "Estimator"))()
        m = fit(RobustLinearModel, form, data, Mest; method=:cg, verbose=false, initial_scale_estimate=:mad, kind=:Sestimate)
        println("rlm($name) : ", coef(m))
        ## TODO: find better test
        @test size(coef(m), 1) == 2
    end
end

@testset "linear: MM-estimator" begin
    println("\n\t\u25CF Estimator type: MM-estimators")
    m1 = fit(LinearModel, form, data)
    println(" lm             : ", coef(m1))
    for name in ("Huber", "L1L2", "Fair", "Logcosh", "Arctan", "Cauchy")
        Mest = getproperty(RobustModels, Symbol(name * "Estimator"))()
        for fallback in (nothing, RobustModels.TukeyEstimator)
            m = fit(RobustLinearModel, form, data, Mest; method=:cg, verbose=false, initial_scale_estimate=:mad, kind=:MMestimate, sestimator=fallback)
            println("rlm($name) : ", coef(m))
            ## TODO: find better test
            @test size(coef(m), 1) == 2
        end
    end

    for name in ("Geman", "Welsch", "Tukey")
        Mest = getproperty(RobustModels, Symbol(name * "Estimator"))()
        for fallback in (nothing, RobustModels.TukeyEstimator)
            m = fit(RobustLinearModel, form, data, Mest; method=:cg, verbose=false, initial_scale_estimate=:mad, kind=:MMestimate, sestimator=fallback)
            println("rlm($name) : ", coef(m))
            ## TODO: find better test
            @test size(coef(m), 1) == 2
        end
    end
end


@testset "linear: leverage correction" begin
    m1 = fit(RobustLinearModel, form, data, est2; method=:cg, initial_scale_estimate=:mad, correct_leverage=false)
    m2 = fit(RobustLinearModel, form, data, est2; method=:cg, initial_scale_estimate=:mad, correct_leverage=true)
    println("rlm(without leverage correction) : ", coef(m1))
    println("rlm(   with leverage correction) : ", coef(m2))
    ## TODO: find better test
    @test all(isapprox.(coef(m1), coef(m2); rtol=0.1))
end

