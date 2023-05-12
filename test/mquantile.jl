

loss1 = RobustModels.L2Loss()
loss2 = RobustModels.TukeyLoss()


@testset "linear: Expectile estimators" begin
    m  = fit(RobustLinearModel, form, data, MEstimator(loss1); method=:cg, maxiter=50, verbose=false, initial_scale=:mad)
    m1 = fit(RobustLinearModel, form, data, GeneralizedQuantileEstimator(loss1, 0.5); method=:cg, maxiter=50, verbose=false, initial_scale=:mad)

    # refit
    # cannot change type to GeneralQuantileEstimator
    @test_throws TypeError refit!(m; quantile=0.5)

    @testset "$(τ) $(name)" for τ in range(0.1, 0.9, step=0.1), name in ("Expectile", "Quantile")
        VERBOSE && println("\n\t\u25CF Estimator: $name($τ)")
        est = getproperty(RobustModels, Symbol(name * "Estimator"))(τ)
        m2 = fit(RobustLinearModel, form, data, est; method=:cg, maxiter=50, rtol=1e-4, initial_scale=:mad)
        m3 = fit(RobustLinearModel, form, data, est; method=:chol, maxiter=50, rtol=1e-4, initial_scale=:mad)
        m4 = fit(RobustLinearModel, form, data, est; method=:qr, maxiter=50, rtol=1e-4, initial_scale=:mad)
        VERBOSE && println("rlm(cg)  : ", coef(m2))
        VERBOSE && println("rlm(chol): ", coef(m3))
        VERBOSE && println("rlm(qr): ", coef(m4))

        if name == "Expectile"
            @test_nowarn println(m2)

            @test isapprox(coef(m2), coef(m3); rtol=1e-4)
            @test isapprox(coef(m2), coef(m4); rtol=1e-4)
            if τ==0.5
                @test isapprox(coef(m), coef(m2); rtol=1e-4)
            end

            # refit with new quantile
            refit!(m1; quantile=τ)
            @test isapprox(coef(m1), coef(m2); rtol=1e-4)
        else
            @test_warn L1_warning println(m2)

            @test isapprox(coef(m2), coef(m3); rtol=1e-2)
            @test isapprox(coef(m2), coef(m4); rtol=1e-2)
        end
    end
end


@testset "linear: M-Quantile estimators" begin
    @testset "$(name) estimator" for name in ("Huber", "L1L2", "Fair", "Logcosh", "Arctan") #, "Cauchy", "Geman", "Welsch", "Tukey", "YohaiZamar")
        VERBOSE && println("\n\t\u25CF M-Quantile Estimator: $name")
        loss = getproperty(RobustModels, Symbol(name * "Loss"))()
        est = MEstimator(loss)
        m1 = fit(RobustLinearModel, form, data, est; method=:cg, maxiter=50, verbose=false, initial_scale=:mad)
        VERBOSE && println("no quant, rlm(cg)  : ", coef(m1))
        est1 = GeneralizedQuantileEstimator(loss)
        @testset "$(τ) quantile" for τ in range(0.1, 0.9, step=0.1)
            est = GeneralizedQuantileEstimator(loss, τ)
            m2 = fit(RobustLinearModel, form, data, est; method=:cg, maxiter=50, verbose=false, initial_scale=:mad)
            m3 = fit(RobustLinearModel, form, data, est; method=:chol, maxiter=50, initial_scale=:mad)
            m4 = fit(RobustLinearModel, form, data, est; method=:qr, maxiter=50, initial_scale=:mad)
            m5 = fit(RobustLinearModel, form, data, est1; quantile=τ, method=:chol, maxiter=50, initial_scale=:mad)
            VERBOSE && println("quant $(τ) rlm(cg)   : ", coef(m2))
            VERBOSE && println("quant $(τ) rlm(chol) : ", coef(m3))
            VERBOSE && println("quant $(τ) rlm(qr)   : ", coef(m4))
            VERBOSE && println("quant $(τ) rlm2(chol): ", coef(m5))
            if τ==0.5
                @test isapprox(coef(m1), coef(m2); rtol=1e-5)
            end
            if name != "L1"
                @test isapprox(coef(m2), coef(m3); rtol=1e-2)
                @test isapprox(coef(m2), coef(m4); rtol=1e-2)
            end
            @test isapprox(coef(m3), coef(m5); rtol=1e-5)
        end
    end
end
