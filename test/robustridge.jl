using Random: MersenneTwister

funcs = ( dof,
          dof_residual,
          confint,
          deviance,
          nulldeviance,
          loglikelihood,
          nullloglikelihood,
          dispersion,
          nobs,
          stderror,
          vcov,
          residuals,
          predict,
          response,
          weights,
          leverage,
          modelmatrix,
          dispersion,
          scale
        )

m1 = fit(LinearModel, form, data)

loss1 = RobustModels.L2Loss()
loss2 = RobustModels.TukeyLoss()
est1 = MEstimator(loss1)
est2 = MEstimator(loss2)

@testset "linear: Ridge M-estimator $(lossname)" for lossname in ("L2", "Huber", "Tukey")
    typeloss = getproperty(RobustModels, Symbol(lossname * "Loss"))
    l = typeloss()
    est = MEstimator(typeloss())

    # Formula, dense and sparse entry  and methods :cg and :chol
    @testset "(type, method): ($(typeof(A)),\t$(method))" for (A, b) in ((form, data), (X, y), (sX, y)), method in (:cg, :chol)
        aspace = if method==:cg; "  " else "    " end
        name  = "MEstimator($(typeloss)),\t"
        name *= if A==form; "formula" elseif A==X; "dense  " else "sparse " end
        name *= if method==:cg; ",  cg" else ",chol" end
        # use the dispersion from GLM to ensure that the loglikelihood is correct
        m2 = fit(RobustLinearModel, A, b, est; method=method, initial_scale=:L1)
        m3 = fit(RobustLinearModel, A, b, est; method=method, initial_scale=:L1, ridgeλ=1)
        m4 = fit(RobustLinearModel, A, b, est; method=method, initial_scale=:L1, ridgeλ=1, ridgeG=float([0 0; 0 1]))
        m5 = fit(RobustLinearModel, A, b, est; method=method, initial_scale=:L1, ridgeλ=0.1, ridgeG=[0, 1])
        m6 = fit(RobustLinearModel, A, b, est; method=method, initial_scale=:L1, ridgeλ=1, ridgeG=[0, 1], βprior=[0.0, 2.0])
        println("\n\t\u25CF Estimator: $(name)")
        println(" lm$(aspace)               : ", coef(m1))
        println("rlm($(method))             : ", coef(m2))
        println("ridge λ=1   rlm3($(method)): ", coef(m3))
        println("ridge λ=1   rlm4($(method)): ", coef(m4))
        println("ridge λ=0.1 rlm5($(method)): ", coef(m5))
        println("ridge λ=1 βprior=[0,2] rlm6($(method)): ", coef(m6))
        @test isapprox(coef(m3), coef(m4); rtol=1e-5)

        # Test printing the model
        @test_nowarn println(m3)
        
        # refit
        refit!(m5; ridgeλ=1, initial_scale=:L1)
        @test isapprox(m5.pred.λ, 1.0; rtol=1e-5)
        @test isapprox(coef(m3), coef(m5); rtol=1e-5)
    end
end

@testset "linear: Ridge L2 estimator methods" begin
    m2 = fit(RobustLinearModel, form, data, est1; method=:chol, initial_scale=:L1)
    m3 = fit(RobustLinearModel, form, data, est1; method=:chol, initial_scale=:L1, ridgeλ=1)

    @testset "method: $(f)" for f in funcs
        # make sure the interfaces for RobustLinearModel are well defined
        @test_nowarn f(m3)
    end

    # Check coefficients and dof change
    λs = vcat([0], 10 .^ range(-2, 1, length=4))

    L = length(coef(m3))
    βs = zeros(L, 5)
    dofs = zeros(5)
    for (i,λ) in enumerate(λs)
        refit!(m3; ridgeλ=λ, initial_scale=:L1)
        βs[:, i] = coef(m3)
        dofs[i] = dof(m2)
    end

    @test isapprox(dofs[1], dof(m2); rtol=1e-5)
    @test isapprox(βs[:, 1], coef(m2); rtol=1e-5)
    @test sort(dofs; rev=true)==dofs
    @test all(sort(βs[r, :]; rev=true)==βs[r, :] for r in 2:L)  # not the intercept
end
