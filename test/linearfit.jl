using RobustModels
using Test
using RDatasets
using StatsModels: @formula, coef
using StatsBase: mad
using GLM: LinearModel

data = dataset("robustbase", "Animals2")
λ = mad(data.Brain; normalize=true)
est1 = RobustModels.L2Estimator()

@testset "linear cg" begin
    m1 = fit(LinearModel, @formula(Brain ~ 1 + Body), data)
    m2 = fit(RobustLinearModel, @formula(Brain ~ 1 + Body), data, est1; method=:cg, verbose=true, estimate_scale=:mad)
    println(" lm: ", coef(m1))
    println("rlm: ", coef(m2))
    @test all(isapprox.(coef(m1), coef(m2); rtol=1.0e-5))
end


@testset "linear chol" begin
    m1 = fit(LinearModel, @formula(Brain ~ 1 + Body), data)
    m2 = fit(RobustLinearModel, @formula(Brain ~ 1 + Body), data, est1; method=:chol, verbose=true, estimate_scale=:mad)
    println(" lm: ", coef(m1))
    println("rlm: ", coef(m2))
    @test all(isapprox.(coef(m1), coef(m2); rtol=1.0e-5))
end

@testset "linear: Robust estimators" begin
    m1 = fit(LinearModel, @formula(Brain ~ 1 + Body), data)
    for name in ("L1", "Huber", "L1L2", "Fair", "Arctan", "Cauchy", "Geman", "Welsch", "Tukey")
        println("\n\t\u25CF Estimator: $(name)")
        est = getproperty(RobustModels, Symbol(name * "Estimator"))()
        m2 = fit(RobustLinearModel, @formula(Brain ~ 1 + Body), data, est; method=:cg, verbose=true, estimate_scale=:mad)
        m3 = fit(RobustLinearModel, @formula(Brain ~ 1 + Body), data, est; method=:chol, estimate_scale=:mad)
        println("\n\t\u25CF Estimator: $(name)")
        println(" lm      : ", coef(m1))
        println("rlm(cg)  : ", coef(m2))
        println("rlm(chol): ", coef(m3))
        if name != "L1"
            @test all(isapprox.(coef(m2), coef(m3); rtol=1.0e-5))
        end
    end
end

@testset "linear: Student's-t estimator" begin
    for ν in range(1, 5, step=1)
        est = RobustModels.StudentEstimator(ν)
        m2 = fit(RobustLinearModel, @formula(Brain ~ 1 + Body), data, est; method=:cg, verbose=true, estimate_scale=:mad)
        m3 = fit(RobustLinearModel, @formula(Brain ~ 1 + Body), data, est; method=:chol, estimate_scale=:mad)
        println("rlm(cg)  : ", coef(m2))
        println("rlm(chol): ", coef(m3))
        @test all(isapprox.(coef(m2), coef(m3); rtol=1.0e-5))
    end
end

@testset "linear: Expectile estimators" begin
    for ν in (0.05, 0.25, 0.5, 0.75, 0.95), name in ("Expectile", )
        println("\n\t\u25CF Estimator: $(name)($ν)")
        est = getproperty(RobustModels, Symbol(name * "Estimator"))(ν)
        m2 = fit(RobustLinearModel, @formula(Brain ~ 1 + Body), data, est; method=:cg, maxiter=50, verbose=true, estimate_scale=:mad)
        m3 = fit(RobustLinearModel, @formula(Brain ~ 1 + Body), data, est; method=:chol, maxiter=50, estimate_scale=:mad)
        println("rlm(cg)  : ", coef(m2))
        println("rlm(chol): ", coef(m3))
        @test all(isapprox.(coef(m2), coef(m3); rtol=1.0e-5))
    end
end


