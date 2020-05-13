using RobustModels
using RobustModels: estimator_rho, 
                    estimator_psi, 
                    estimator_psider, 
                    estimator_chi, 
                    estimator_weight, 
                    estimator_values,
                    estimator_high_breakdown_point_constant,
                    estimator_high_efficiency_constant,
                    isbounded,
                    isconvex,
                    efficiency_tuning_constant,
                    breakdown_point_tuning_constant,
                    nothing  # stopper
using Test
using RDatasets: dataset
using StatsModels: @formula, coef
using StatsBase: mad
using GLM: LinearModel

data = dataset("robustbase", "Animals2")
form = @formula(Brain ~ 1 + Body)

m1 = fit(LinearModel, form, data)

λ = mad(data.Brain; normalize=true)
est1 = RobustModels.L2Estimator()
est2 = RobustModels.TukeyEstimator()

@testset "Methods estimators: $(name)" for name in ("L2", "L1", "Huber", "L1L2", "Fair", "Logcosh", "Arctan", "Cauchy", "Geman", "Welsch", "Tukey")
    typest = getproperty(RobustModels, Symbol(name * "Estimator"))
    est = typest()
    
    @testset "Bounded estimators: $(name)" begin
        if name in ("Geman", "Welsch", "Tukey")
            @test isbounded(est)
        else
            @test !isbounded(est)
        end
    end

    @testset "Convex estimators: $(name)" begin
        if name in ("Cauchy", "Geman", "Welsch", "Tukey")
            @test !isconvex(est)
        else
            @test isconvex(est)
        end
    end

    @testset "Estimator values: $(name)" begin
        ρ  = estimator_rho(est, 1)
        ψ  = estimator_psi(est, 1)
        ψp = estimator_psider(est, 1)
        w  = estimator_weight(est, 1)
        
        vals = estimator_values(est, 1)
        @test length(vals) == 3
        @test vals[1] ≈ ρ  rtol=1e-6
        @test vals[2] ≈ ψ  rtol=1e-6
        @test vals[3] ≈ w  rtol=1e-6
    end

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
            v = breakdown_point_tuning_constant(typest; bp=0.5, c0=0.9*vopt)
            @test isapprox(v, vopt; rtol=1e-3)
        end
    end

    @testset "MQuantile: $(name)" begin
        τ = 0.5
        qest1 = GeneralQuantileEstimator(est, τ)
        qest2 = GeneralQuantileEstimator{typest}(τ)
        @test qest1 == qest2
        if name == "L2"
            qest3 = ExpectileEstimator(τ)
            @test qest1 == qest3
        elseif name == "L1"
            qest3 = QuantileEstimator(τ)
            @test qest1 == qest3
        end

        @testset "Pass through of methods" for fun in (estimator_rho, estimator_psi, estimator_psider, estimator_weight)
            ρ1 = fun(est, 1)
            ρ2 = fun(qest1, 1)
            @test ρ1 == ρ2
        end
        
    end

end
