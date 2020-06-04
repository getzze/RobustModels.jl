
using Random: MersenneTwister

m1 = fit(LinearModel, form, data)

est1 = RobustModels.L2Estimator()
est2 = RobustModels.TukeyEstimator()

@testset "linear: M-estimator $(name)" for name in ("L1", "Huber", "L1L2", "Fair", "Logcosh", "Arctan", "Cauchy", "Geman", "Welsch", "Tukey", "YohaiZamar")
    est = getproperty(RobustModels, Symbol(name * "Estimator"))()
    m2 = fit(RobustLinearModel, form, data, est; method=:cg, verbose=false, initial_scale_estimate=:mad)
    m3 = fit(RobustLinearModel, form, data, est; method=:chol, initial_scale_estimate=:mad)
    m4 = rlm(form, data, est; method=:chol, initial_scale_estimate=:mad)
    println("\n\t\u25CF Estimator: $(name)")
    println(" lm       : ", coef(m1))
    println("rlm(cg)   : ", coef(m2))
    println("rlm(chol) : ", coef(m3))
    println("rlm2(chol): ", coef(m4))
    println(m2)
    if name != "L1"
        @test isapprox(coef(m2), coef(m3); rtol=1e-5)
    end
    
    # refit
    β2 = copy(coef(m2))
    refit!(m2, y; wts=weights(m2), verbose=false, initial_scale_estimate=:mad)
    @test all(coef(m2) .== β2)
end


@testset "linear: S-estimator" begin
    println("\n\t\u25CF Estimator type: S-estimators")
    println(" lm             : ", coef(m1))
    @testset "Not bounded: $(name)" for name in ("Huber", "L1L2", "Fair", "Logcosh", "Arctan", "Cauchy")
        Mest = getproperty(RobustModels, Symbol(name * "Estimator"))()
        @test_throws ErrorException fit(RobustLinearModel, form, data, Mest; method=:cg, verbose=false, initial_scale_estimate=:mad, kind=:Sestimate)
    end

    @testset "Bounded: $(name)" for name in ("Geman", "Welsch", "Tukey", "YohaiZamar")
        Mest = getproperty(RobustModels, Symbol(name * "Estimator"))()
        m = fit(RobustLinearModel, form, data, Mest; method=:cg, verbose=false, initial_scale_estimate=:mad, kind=:Sestimate)
        println("rlm($name) : ", coef(m))
        ## TODO: find better test
        @test size(coef(m), 1) == 2

        # Resampling
        rng = MersenneTwister(1)
        opts = Dict(:Nsteps_β=>3, :rng=>rng)
        m2 = fit(RobustLinearModel, form, data, Mest; method=:cg, verbose=false, initial_scale_estimate=:mad, kind=:Sestimate, resample=true, resampling_options=opts)
        @test (scale(m2) / scale(m) - 1) <= 1e-2
    end
end

@testset "linear: MM-estimator" begin
    println("\n\t\u25CF Estimator type: MM-estimators")
    println(" lm             : ", coef(m1))
    @testset "Not bounded: $(name)" for name in ("Huber", "L1L2", "Fair", "Logcosh", "Arctan", "Cauchy")
        Mest = getproperty(RobustModels, Symbol(name * "Estimator"))()
        @testset "Fallback for S-estimate: $(fallback)" for fallback in (nothing, RobustModels.TukeyEstimator)
            m = fit(RobustLinearModel, form, data, Mest; method=:cg, verbose=false, initial_scale_estimate=:mad, kind=:MMestimate, sestimator=fallback)
            println("rlm($name) : ", coef(m))
            ## TODO: find better test
            @test size(coef(m), 1) == 2
        end
    end

    @testset "Bounded: $(name)" for name in ("Geman", "Welsch", "Tukey", "YohaiZamar")
        Mest = getproperty(RobustModels, Symbol(name * "Estimator"))()
        @testset "Fallback for S-estimate: $(fallback)" for fallback in (nothing, RobustModels.TukeyEstimator)
            m = fit(RobustLinearModel, form, data, Mest; method=:cg, verbose=false, initial_scale_estimate=:mad, kind=:MMestimate, sestimator=fallback)
            println("rlm($name) : ", coef(m))
            ## TODO: find better test
            @test size(coef(m), 1) == 2

            # Resampling
            rng = MersenneTwister(1)
            opts = Dict(:Npoints=>10, :Nsteps_β=>3, :Nsteps_σ=>3, :rng=>rng)
            m2 = fit(RobustLinearModel, form, data, Mest; method=:cg, verbose=false, initial_scale_estimate=:mad, kind=:MMestimate, sestimator=fallback, resample=true, resampling_options=opts)
            @test (scale(m2) / scale(m) - 1) <= 1e-2
        end
    end
end

@testset "linear: τ-estimator" begin
    println("\n\t\u25CF Estimator type: τ-estimators")
    println(" lm             : ", coef(m1))
    @testset "Not bounded: $(name)" for name in ("Huber", "L1L2", "Fair", "Logcosh", "Arctan", "Cauchy")
        @test_throws TypeError TauEstimator(getproperty(RobustModels, Symbol(name * "Estimator")))
    end

    @testset "Bounded: $(name)" for name in ("Geman", "Welsch", "Tukey", "YohaiZamar")
        E = getproperty(RobustModels, Symbol(name * "Estimator"))
        Mest = E()
        τest = TauEstimator(E)

        @test_throws ErrorException fit(RobustLinearModel, form, data, Mest; method=:cg, verbose=false, initial_scale_estimate=:mad, kind=:Tauestimate)

        m = fit(RobustLinearModel, form, data, τest; method=:cg, verbose=false, initial_scale_estimate=:mad, kind=:Tauestimate)
        println("rlm($name) : ", coef(m))
        ## TODO: find better test
        @test size(coef(m), 1) == 2

        # Resampling
        rng = MersenneTwister(1)
        opts = Dict(:Nsteps_β=>3, :Nsteps_σ=>3, :rng=>rng)
        m2 = fit(RobustLinearModel, form, data, τest; method=:cg, verbose=false, initial_scale_estimate=:mad, kind=:Tauestimate, resample=true, resampling_options=opts)
        @test (tauscale(m2) / tauscale(m) - 1) <= 1e-2
    end
end

@testset "linear: leverage correction" begin
    m2 = fit(RobustLinearModel, form, data, est2; method=:cg, initial_scale_estimate=:mad, correct_leverage=false)
    m3 = fit(RobustLinearModel, form, data, est2; method=:cg, initial_scale_estimate=:mad, correct_leverage=true)
    println("rlm(without leverage correction) : ", coef(m2))
    println("rlm(   with leverage correction) : ", coef(m3))
    ## TODO: find better test
    @test isapprox(coef(m2), coef(m3); rtol=0.1)
end

