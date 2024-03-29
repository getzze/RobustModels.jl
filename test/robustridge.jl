using Random: MersenneTwister
using LinearAlgebra: inv, Hermitian, I, tr, diag

seed = 123987

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
    @testset "$(typeof(A)),\t$(method)" for (A, b) in data_tuples, method in nopen_methods

        aspace = (method in (:cg, :qr)) ? "  " : "    "
        name = "MEstimator($(typeloss)),\t"
        name *= if (A == form)
            "formula"
        elseif (A == X)
            "dense  "
        else
            "sparse "
        end
        name *= (method in (:cg, :qr)) ? ",  " : ","
        name *= "$(method)"

        kwargs = (; method=method, initial_scale=:L1)
        # use the dispersion from GLM to ensure that the loglikelihood is correct
        m2 = fit(RobustLinearModel, A, b, est; kwargs...)
        m3 = fit(RobustLinearModel, A, b, est; ridgeλ=10, kwargs...)
        m4 = fit(
            RobustLinearModel, A, b, est; ridgeλ=10, ridgeG=float([0 0; 0 1]), kwargs...
        )
        m5 = fit(RobustLinearModel, A, b, est; ridgeλ=0.1, ridgeG=[0, 1], kwargs...)
        m6 = rlm(A, b, est; ridgeλ=0.1, ridgeG=[0, 1], dropcollinear=true, kwargs...)
        m7 = rlm(A, b, est; ridgeλ=10, ridgeG=[0, 1], βprior=[0.0, 2.0], kwargs...)

        VERBOSE && println("\n\t\u25CF Estimator: $(name)")
        VERBOSE && println(" lm$(aspace)               : ", coef(m1))
        VERBOSE && println("rlm($(method))             : ", coef(m2))
        VERBOSE && println("ridge λ=10  rlm3($(method)): ", coef(m3))
        VERBOSE && println("ridge λ=10  rlm4($(method)): ", coef(m4))
        VERBOSE && println("ridge λ=0.1 rlm5($(method)): ", coef(m5))
        VERBOSE && println("ridge λ=0.1 dropcollinear=true rlm6($(method)): ", coef(m6))
        VERBOSE && println("ridge λ=10 βprior=[0,2] rlm7($(method)): ", coef(m7))
        @test isapprox(coef(m3), coef(m4); rtol=1e-5)

        # Test printing the model
        @test_nowarn show(devnull, m3)

        # refit
        refit!(m5; ridgeλ=10, initial_scale=:L1)
        @test isapprox(m5.pred.λ, 10.0; rtol=1e-5)
        @test isapprox(coef(m3), coef(m5); rtol=1e-6)

        refit!(m6; ridgeλ=10, initial_scale=:L1)
        @test isapprox(m6.pred.λ, 10.0; rtol=1e-5)
        @test isapprox(coef(m3), coef(m6); rtol=1e-6)
    end
end

@testset "linear: Ridge L2 estimator methods" begin
    m2 = fit(RobustLinearModel, form, data, est1; method=:chol, initial_scale=:L1)
    m3 = fit(RobustLinearModel, form, data, est1; method=:chol, initial_scale=:L1, ridgeλ=1)

    @testset "method: $(f)" for f in interface_methods
        # make sure the interfaces for RobustLinearModel are well defined
        @test_nowarn f(m3)
    end

    # Check coefficients and dof change
    λs = vcat([0], 10 .^ range(-2, 1; length=4))

    L = length(coef(m3))
    βs = zeros(L, 5)
    dofs = zeros(5)
    for (i, λ) in enumerate(λs)
        refit!(m3; ridgeλ=λ, initial_scale=:L1)
        βs[:, i] = coef(m3)
        dofs[i] = dof(m2)
    end

    @test isapprox(dofs[1], dof(m2); rtol=1e-5)
    @test isapprox(βs[:, 1], coef(m2); rtol=1e-5)
    @test sort(dofs; rev=true) == dofs
    @test all(sort(βs[r, :]; rev=true) == βs[r, :] for r in 2:L)  # not the intercept
end


@testset "linear: Ridge L2 exact solution" begin
    rng = MersenneTwister(seed)

    n = 10_000
    p = 4
    σ = 0.5
    λ = 1000
    Xt = randn(rng, n, p)
    βt = randn(rng, p)
    yt = Xt * βt + σ * randn(rng, n)

    vc = inv(Hermitian(Xt'Xt + λ * I(p)))

    βsol = vc * (Xt' * yt)
    dofsol = tr(Xt * vc * Xt')
    stdβsol = σ * √(n / (n - p)) * .√(diag(vc * Xt'Xt * vc'))

    m = rlm(Xt, yt, est1; method=:chol, ridgeλ=λ)

    @test coef(m) ≈ βsol
    @test dof(m) ≈ dofsol
    @test vcov(m) ≈ vc
    @test_skip stderror(m) ≈ stdβsol
end
