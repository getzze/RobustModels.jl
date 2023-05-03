
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

    @testset "Infinite loop: issue #32" begin
        # Infinite loop: issue #32
        Xs = [0.001481; 0.0017; 0.00133; 0.001853; 0.002086; 0.003189; 0.001161; 0.002441; 0.001133; 0.001308; 0.001; 0.009309; 0.1456; 0.3127; 0.2627; 0.1704; 0.101; 0.06855; 0.02578]
        ys = [1.222, 1.599, 2.238, 2.233, 2.668, 2.637, 3.177, 2.539, 2.339, 1.481, 1.733, 0.04986, 0.0812, 0.1057, 0.1197, 0.1348, 0.1006, 0.1021, 0.08278]
        @test_throws Exception rlm(Xs, ys, SEstimator{TukeyLoss}(), initial_scale=:mad)
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

