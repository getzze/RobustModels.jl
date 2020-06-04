
m1 = fit(LinearModel, form, data)
λlm = dispersion(m1)

est1 = RobustModels.L2Estimator()
est2 = RobustModels.TukeyEstimator()

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
          scale
        )


@testset "linear: L2 estimator" begin
    println("\n\t\u25CF Estimator: L2")

    # OLS
    println(m1)
    println(" lm              : ", coef(m1))

    # Formula, dense and sparse entry  and methods :cg and :chol
    @testset "(type, method): ($(typeof(A)),\t$(method))" for (A, b) in ((form, data), (X, y), (sX, y)), method in (:cg, :chol)
        name  = if A==form; "formula" elseif A==X; "dense  " else "sparse " end
        name *= if method==:cg; ",  cg" else ",chol" end
        # use the dispersion from GLM to ensure that the loglikelihood is correct
        m = fit(RobustLinearModel, A, b, est1; method=method, verbose=false, initial_scale_estimate=λlm)
        β = copy(coef(m))
        println("rlm($name): ", β)
        println(m)
        @test isapprox(coef(m1), β; rtol=1e-5)

        # refit
        refit!(m, y; verbose=false, initial_scale_estimate=:extrema)
        println("$m")
        @test all(coef(m) .== β)
        
        # interface
        @testset "method: $(f)" for f in funcs
            # make sure the interfaces for RobustLinearModel are well defined
            if f in (weights, leverage, modelmatrix, scale)
                # method is not defined in GLM
                robvar = f(m)
            else
                var = f(m1)
                robvar = f(m)
                if isa(var, Union{AbstractArray, Tuple})
                    if f != vcov
                        @test isapprox(var, robvar; rtol=1e-4)
                    end
                else
                    if f == dof
                        @test var == robvar + 1
                    elseif f == dof_residual
                        @test var == robvar
                    elseif f in (deviance, nulldeviance)
                        s = scale(m)
                        ## the deviance in RobustModels is the scaled deviance (divided by 
                        ## the squared dispersion), whereas in GLM it is not.
                        @test isapprox(var, robvar * s^2; rtol=1e-4)
                    elseif f in (loglikelihood, nullloglikelihood)
                        ## TODO: should work
                        @test_broken isapprox(var, robvar; rtol=1e-4)
#                        @test isapprox(var + log(λ), (robvar + RobustModels.fullloglikelihood(r)) * s^2/λ^2 - log(RobustModels.estimator_norm(r.est)); rtol=1e-4)
#                        @test isapprox(var + log(dispersion(m1)), robvar + log(s); rtol=1e-4)
                    else
                        @test isapprox(var, robvar; rtol=1e-4)
                    end
                end
            end
        end
        
        # later fit!
        m2 = fit(RobustLinearModel, A, b, est1; method=method, dofit=false)
        @test all(0 .== coef(m2))
        fit!(m2; verbose=false, initial_scale_estimate=:mad)
        @test all(β .== coef(m2))
    end
end


