
using Random: MersenneTwister

m1 = fit(LinearModel, form, data)

loss1 = RobustModels.L2Loss()
loss2 = RobustModels.TukeyLoss()
est1 = MEstimator(loss1)
est2 = MEstimator(loss2)

@testset "linear: M-estimator $(name)" for name in ("L1", "Huber", "L1L2", "Fair", "Logcosh", "Arctan", "Cauchy", "Geman", "Welsch", "Tukey", "YohaiZamar")
    typeloss = getproperty(RobustModels, Symbol(name * "Loss"))
    l = typeloss()
    est = MEstimator(l)
    est3 = MEstimator{typeloss}()
    m2 = fit(RobustLinearModel, form, data, est; method=:cg, verbose=false, initial_scale=:mad)
    m3 = fit(RobustLinearModel, form, data, est; method=:chol, initial_scale=:mad)
    m4 = rlm(form, data, est; method=:chol, initial_scale=:mad)
    m5 = rlm(form, data, est3; method=:chol, initial_scale=:mad)
    println("\n\t\u25CF Estimator: $(name)")
    println(" lm       : ", coef(m1))
    println("rlm(cg)   : ", coef(m2))
    println("rlm(chol) : ", coef(m3))
    println("rlm2(chol): ", coef(m4))
    println("rlm3(chol): ", coef(m5))
    println(m2)
    if name != "L1"
        @test isapprox(coef(m2), coef(m3); rtol=1e-5)
    end

    # refit
    β2 = copy(coef(m2))
    refit!(m2, y; wts=weights(m2), verbose=false, initial_scale=:mad)
    @test all(coef(m2) .== β2)
end


@testset "linear: S-estimator" begin
    println("\n\t\u25CF Estimator type: S-estimators")
    println(" lm             : ", coef(m1))
    @testset "Not bounded: $(name)" for name in ("Huber", "L1L2", "Fair", "Logcosh", "Arctan", "Cauchy")
        typeloss = getproperty(RobustModels, Symbol(name * "Loss"))
        @test_throws TypeError SEstimator{typeloss}()
    end

    @testset "Bounded: $(name)" for name in ("Geman", "Welsch", "Tukey", "YohaiZamar")
        typeloss = getproperty(RobustModels, Symbol(name * "Loss"))
        est = SEstimator{typeloss}()
        m = fit(RobustLinearModel, form, data, est; method=:cg, verbose=false, initial_scale=:mad)
        println("rlm($name) : ", coef(m))
        ## TODO: find better test
        @test size(coef(m), 1) == 2

        # Resampling
        rng = MersenneTwister(1)
        opts = Dict(:Nsteps_β=>3, :rng=>rng)
        m2 = fit(RobustLinearModel, form, data, est; method=:cg, verbose=false, initial_scale=:mad, resample=true, resampling_options=opts)
        @test (scale(m2) / scale(m) - 1) <= 1e-2
    end
end

@testset "linear: MM-estimator" begin
    println("\n\t\u25CF Estimator type: MM-estimators")
    println(" lm             : ", coef(m1))
    @testset "Not bounded: $(name)" for name in ("Huber", "L1L2", "Fair", "Logcosh", "Arctan", "Cauchy")
        typeloss = getproperty(RobustModels, Symbol(name * "Loss"))
        @test_throws TypeError MMEstimator{typeloss}()
    end

    @testset "Bounded: $(name)" for name in ("Geman", "Welsch", "Tukey", "YohaiZamar")
        typeloss = getproperty(RobustModels, Symbol(name * "Loss"))
        est = MMEstimator{typeloss}()
        m = fit(RobustLinearModel, form, data, est; method=:cg, verbose=false, initial_scale=:mad)
        println("rlm($name) : ", coef(m))
        ## TODO: find better test
        @test size(coef(m), 1) == 2

        # Resampling
        rng = MersenneTwister(1)
        opts = Dict(:Npoints=>10, :Nsteps_β=>3, :Nsteps_σ=>3, :rng=>rng)
        m2 = fit(RobustLinearModel, form, data, est; method=:cg, verbose=false, initial_scale=:mad, resample=true, resampling_options=opts)
        @test (scale(m2) / scale(m) - 1) <= 1e-2
    end
end

@testset "linear: τ-estimator" begin
    println("\n\t\u25CF Estimator type: τ-estimators")
    println(" lm             : ", coef(m1))
    @testset "Not bounded: $(name)" for name in ("Huber", "L1L2", "Fair", "Logcosh", "Arctan", "Cauchy")
        typeloss = getproperty(RobustModels, Symbol(name * "Loss"))
        @test_throws TypeError TauEstimator{typeloss}()
    end

    @testset "Bounded: $(name)" for name in ("Geman", "Welsch", "Tukey", "YohaiZamar")
        typeloss = getproperty(RobustModels, Symbol(name * "Loss"))
        est = TauEstimator{typeloss}()

        m = fit(RobustLinearModel, form, data, est; method=:cg, verbose=false, initial_scale=:mad)
        println("rlm($name) : ", coef(m))
        ## TODO: find better test
        @test size(coef(m), 1) == 2

        # Resampling
        rng = MersenneTwister(1)
        opts = Dict(:Nsteps_β=>3, :Nsteps_σ=>3, :rng=>rng)
        m2 = fit(RobustLinearModel, form, data, est; method=:cg, verbose=false, initial_scale=:mad, resample=true, resampling_options=opts)
        @test (tauscale(m2) / tauscale(m) - 1) <= 1e-2
    end
end

@testset "linear: leverage correction" begin
    m2 = fit(RobustLinearModel, form, data, est2; method=:cg, initial_scale=:mad, correct_leverage=false)
    m3 = fit(RobustLinearModel, form, data, est2; method=:cg, initial_scale=:mad, correct_leverage=true)
    println("rlm(without leverage correction) : ", coef(m2))
    println("rlm(   with leverage correction) : ", coef(m3))
    ## TODO: find better test
    @test isapprox(coef(m2), coef(m3); rtol=0.1)
end

