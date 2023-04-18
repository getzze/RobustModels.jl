using QuadGK: quadgk
using RobustModels:
    rho,
    psi,
    psider,
    weight,
    estimator_values,
    estimator_high_breakdown_point_constant,
    estimator_high_efficiency_constant,
    estimator_tau_efficient_constant,
    isbounded,
    isconvex,
    efficiency_tuning_constant,
    breakdown_point_tuning_constant,
    tau_efficiency_tuning_constant,
    nothing  # stopper

estimators = (
    "L2",
    "L1",
    "Huber",
    "L1L2",
    "Fair",
    "Logcosh",
    "Arctan",
    "CatoniWide",
    "CatoniNarrow",
    "Cauchy",
    "Geman",
    "Welsch",
    "Tukey",
    "YohaiZamar",
    "HardThreshold",
    "Hampel",
)


bounded_losses = ("Geman", "Welsch", "Tukey", "YohaiZamar", "HardThreshold", "Hampel")

# norm
emp_norm(l::LossFunction) = 2 * quadgk(x -> exp(-RobustModels.rho(l, x)), 0, Inf)[1]



@testset "Methods loss functions: $(name)" for name in estimators
    typeloss = getproperty(RobustModels, Symbol(name * "Loss"))
    l = typeloss()

    @testset "Methods estimators: $(estimator)" for estimator in (nothing, "M", "S", "MM", "Tau", "GeneralizedQuantile")
        # Check LossFunction methods
        if isnothing(estimator)
            estimator_name = "Loss function"
            typest = typeloss

        # Check AbstractEstimator methods
        else
            estimator_name = "$(estimator) Estimator"
            T = getproperty(RobustModels, Symbol(estimator * "Estimator"))
            if estimator in ("S", "MM", "Tau")
                if !in(name, bounded_losses)
                    @test_throws TypeError T{typeloss}
                    continue
                end
            end
            typest = T{typeloss}
        end
        est = typest()
        @test_nowarn println(est)

        if !isnothing(estimator)
            if estimator == "Tau"
#                @test isa(loss(est), Tuple{BoundedLossFunction, BoundedLossFunction})
                @test isa(loss(est), CompositeLossFunction)
                @test typeof(first(loss(est))) == typeloss
                @test typeof(last(loss(est))) == typeloss
            else
                @test typeof(loss(est)) == typeloss
            end
        end

        @testset "Bounded $(estimator_name): $(name)" begin
            if name in bounded_losses
                @test isbounded(est)
            else
                @test !isbounded(est)
            end
        end

        @testset "Convex $(estimator_name): $(name)" begin
            if name == "Cauchy" || name in bounded_losses
                @test !isconvex(est)
            else
                @test isconvex(est)
            end
        end

        @testset "$(estimator_name) values: $(name)" begin
            ρ  = rho(est, 1)
            ψ  = psi(est, 1)
            ψp = psider(est, 1)
            w  = weight(est, 1)

            vals = estimator_values(est, 1)
            @test length(vals) == 3
            @test vals[1] ≈ ρ  rtol=1e-6
            @test vals[2] ≈ ψ  rtol=1e-6
            @test vals[3] ≈ w  rtol=1e-6
        end

        # Only for LossFunction
        if isnothing(estimator)
            if !isbounded(est)
                @testset "Estimator norm: $(name)" begin
                    @test emp_norm(est) ≈ RobustModels.estimator_norm(est)  rtol=1e-5
                end
            end

            if !in(name, ("L2", "L1"))
                @testset "Estimator high efficiency: $(name)" begin
                    vopt = estimator_high_efficiency_constant(typest)
                    if name != "HardThreshold"
                        v = efficiency_tuning_constant(typest; eff=0.95, c0=0.9*vopt)
                        @test isapprox(v, vopt; rtol=1e-3)
                    end
                end
            end

            if isbounded(est)
                @testset "Estimator high breakdown point: $(name)" begin
                    vopt = estimator_high_breakdown_point_constant(typest)
                    v = breakdown_point_tuning_constant(typest; bp=0.5, c0=1.1*vopt)
                    @test isapprox(v, vopt; rtol=1e-3)
                end

                @testset "τ-Estimator high efficiency: $(name)" begin
                    vopt = estimator_tau_efficient_constant(typest)
                    if name != "HardThreshold"
                        v = tau_efficiency_tuning_constant(typest; eff=0.95, c0=1.1*vopt)
                        @test isapprox(v, vopt; rtol=1e-3)
                    end
                end
            end
        end

        if typest <: AbstractQuantileEstimator
            @testset "MQuantile: $(name)" begin
                τ = 0.5
                qest1 = GeneralizedQuantileEstimator(l, τ)
                qest2 = GeneralizedQuantileEstimator{typeloss}(τ)
                @test qest1 == qest2
                if name == "L2"
                    qest4 = ExpectileEstimator(τ)
                    @test qest1 == qest4
                elseif name == "L1"
                    qest4 = RobustModels.QuantileEstimator(τ)
                    @test qest1 == qest4
                end

                @testset "Pass through of methods" for fun in (rho, psi, psider, weight)
                    ρ1 = fun(est, 1)
                    ρ2 = fun(qest1, 1)
                    @test ρ1 == ρ2
                end

            end
        end
    end

end
