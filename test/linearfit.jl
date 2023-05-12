

m1 = fit(LinearModel, form, data)

loss1 = RobustModels.L2Loss()
loss2 = RobustModels.TukeyLoss()
est1 = MEstimator(loss1)
est2 = MEstimator(loss2)

@testset "linear: M-estimator $(name)" for name in losses
    typeloss = getproperty(RobustModels, Symbol(name * "Loss"))
    l = typeloss()
    est = MEstimator(l)
    est3 = MEstimator{typeloss}()
    kwargs = (; initial_scale=:mad)
    if name == "L1"
        kwargs = (; initial_scale=:mad, maxiter=100)
    end

    m2 = fit(RobustLinearModel, form, data, est; verbose=false, kwargs...)
    @test all(isfinite.(coef(m2)))

    if name != "L1"
        @test_nowarn println(m2)
    else
        @test_warn L1_warning println(m2)
    end

    VERBOSE && println("\n\t\u25CF Estimator: $(name)")
    VERBOSE && println(" lm       : ", coef(m1))
    VERBOSE && println("rlm(:auto)   : ", coef(m2))

    @testset "method: $(method)" for method in nopen_methods
        m3 = fit(RobustLinearModel, form, data, est; method=method, kwargs...)
        m4 = rlm(form, data, est; method=method, kwargs...)
        m5 = rlm(form, data, est3; method=method, kwargs...)

        VERBOSE && println("rlm($(method)) : ", coef(m3))
        VERBOSE && println("rlm2($(method)): ", coef(m4))
        VERBOSE && println("rlm3($(method)): ", coef(m5))

        if name != "L1"
            @test isapprox(coef(m2), coef(m3); rtol=1e-5)
        end
    end

    # refit
    β2 = copy(coef(m2))
    refit!(m2, y; wts=weights(m2), verbose=false, kwargs...)
    @test all(coef(m2) .== β2)

    # empty refit
    refit!(m2; kwargs...)
    @test all(coef(m2) .== β2)
end

not_bounded_losses = setdiff(Set(losses), Set(bounded_losses))

@testset "linear: S-estimator" begin
    VERBOSE && println("\n\t\u25CF Estimator type: S-estimators")
    VERBOSE && println(" lm             : ", coef(m1))
    @testset "Not bounded: $(name)" for name in not_bounded_losses
        typeloss = getproperty(RobustModels, Symbol(name * "Loss"))
        @test_throws TypeError SEstimator{typeloss}()
    end

    @testset "Bounded: $(name)" for name in bounded_losses
        typeloss = getproperty(RobustModels, Symbol(name * "Loss"))
        est = SEstimator{typeloss}()
        m = fit(RobustLinearModel, form, data, est; method=:cg, verbose=false, initial_scale=:mad)
        VERBOSE && println("rlm($name) : ", coef(m))
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
        Xs = reshape([0.001481, 0.0017, 0.00133, 0.001853, 0.002086, 0.003189, 0.001161, 0.002441, 0.001133, 0.001308, 0.001, 0.009309, 0.1456, 0.3127, 0.2627, 0.1704, 0.101, 0.06855, 0.02578], :, 1)
        ys = [1.222, 1.599, 2.238, 2.233, 2.668, 2.637, 3.177, 2.539, 2.339, 1.481, 1.733, 0.04986, 0.0812, 0.1057, 0.1197, 0.1348, 0.1006, 0.1021, 0.08278]
        @test_throws Exception rlm(Xs, ys, SEstimator{TukeyLoss}(), initial_scale=:mad)
    end
end

@testset "linear: MM-estimator" begin
    VERBOSE && println("\n\t\u25CF Estimator type: MM-estimators")
    VERBOSE && println(" lm             : ", coef(m1))
    @testset "Not bounded: $(name)" for name in not_bounded_losses
        typeloss = getproperty(RobustModels, Symbol(name * "Loss"))
        @test_throws TypeError MMEstimator{typeloss}()
    end

    @testset "Bounded: $(name)" for name in bounded_losses
        typeloss = getproperty(RobustModels, Symbol(name * "Loss"))
        est = MMEstimator{typeloss}()
        m = fit(RobustLinearModel, form, data, est; method=:cg, verbose=false, initial_scale=:mad)
        VERBOSE && println("rlm($name) : ", coef(m))
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
    VERBOSE && println("\n\t\u25CF Estimator type: τ-estimators")
    VERBOSE && println(" lm             : ", coef(m1))
    @testset "Not bounded: $(name)" for name in not_bounded_losses
        typeloss = getproperty(RobustModels, Symbol(name * "Loss"))
        @test_throws TypeError TauEstimator{typeloss}()
    end

    @testset "Bounded: $(name)" for name in bounded_losses
        typeloss = getproperty(RobustModels, Symbol(name * "Loss"))
        est = TauEstimator{typeloss}()

        m = fit(RobustLinearModel, form, data, est; method=:cg, verbose=false, initial_scale=:mad)
        VERBOSE && println("rlm($name) : ", coef(m))
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
    VERBOSE && println("rlm(without leverage correction) : ", coef(m2))
    VERBOSE && println("rlm(   with leverage correction) : ", coef(m3))
    ## TODO: find better test
    @test isapprox(coef(m2), coef(m3); rtol=0.1)
end

