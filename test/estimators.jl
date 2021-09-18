using RobustModels: rho,
                    psi,
                    psider,
                    weight,
                    values,
                    estimator_high_breakdown_point_constant,
                    estimator_high_efficiency_constant,
                    estimator_tau_efficient_constant,
                    isbounded,
                    isconvex,
                    efficiency_tuning_constant,
                    breakdown_point_tuning_constant,
                    tau_efficiency_tuning_constant,
                    nothing  # stopper


@testset "Methods loss functions: $(name)" for name in ("L2", "L1", "Huber", "L1L2", "Fair", "Logcosh", "Arctan", "Cauchy", "Geman", "Welsch", "Tukey", "YohaiZamar")
    typeloss = getproperty(RobustModels, Symbol(name * "Loss"))
    l = typeloss()

    @testset "Methods estimators: $(estimator)" for estimator in (nothing, "M", "S", "MM", "Tau", "GeneralizedQuantile")
        typest = if isnothing(estimator)
            # Check LossFunction methods
            typeloss
        else
            # Check AbstractEstimator methods
            T = getproperty(RobustModels, Symbol(estimator * "Estimator"))
            if estimator in ("S", "MM", "Tau")
                if !in(name, ("Geman", "Welsch", "Tukey", "YohaiZamar"))
                    @test_throws TypeError T{typeloss}
                    continue
                end
            end
            T{typeloss}
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

        @testset "Bounded $(estimator) estimators: $(name)" begin
            if name in ("Geman", "Welsch", "Tukey", "YohaiZamar")
                @test isbounded(est)
            else
                @test !isbounded(est)
            end
        end

        @testset "Convex $(estimator) estimators: $(name)" begin
            if name in ("Cauchy", "Geman", "Welsch", "Tukey", "YohaiZamar")
                @test !isconvex(est)
            else
                @test isconvex(est)
            end
        end

        @testset "$(estimator) Estimator values: $(name)" begin
            ρ  = rho(est, 1)
            ψ  = psi(est, 1)
            ψp = psider(est, 1)
            w  = weight(est, 1)

            vals = values(est, 1)
            @test length(vals) == 3
            @test vals[1] ≈ ρ  rtol=1e-6
            @test vals[2] ≈ ψ  rtol=1e-6
            @test vals[3] ≈ w  rtol=1e-6
        end

        # Only for LossFunction
        if isnothing(estimator)
            if !in(name, ("L2", "L1"))
                @testset "Estimator high efficiency: $(name)" begin
                    vopt = estimator_high_efficiency_constant(typest)
                    v = efficiency_tuning_constant(typest; eff=0.95, c0=0.9*vopt)
                    @test isapprox(v, vopt; rtol=1e-3)
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
                    v = tau_efficiency_tuning_constant(typest; eff=0.95, c0=1.1*vopt)
                    @test isapprox(v, vopt; rtol=1e-3)
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
