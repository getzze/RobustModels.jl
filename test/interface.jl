
using Tables
using Missings
using StatsModels: FormulaTerm

m1 = fit(LinearModel, form, data)
λlm = dispersion(m1)

loss1 = RobustModels.L2Loss()
loss2 = RobustModels.TukeyLoss()
est1 = MEstimator(loss1)
est2 = MEstimator(loss2)

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
          workingweights,
          fitted,
          predict,
          isfitted,
          islinear,
          leverage,
          modelmatrix,
          projectionmatrix,
          wobs,
          Estimator,
          scale,
          hasintercept,
        )


@testset "linear: L2 estimator" begin
    println("\n\t\u25CF Estimator: L2")

    # OLS
    println(m1)
    println(" lm              : ", coef(m1))

    # Formula, dense and sparse entry  and methods :cg and :chol
    @testset "(type, method): ($(typeof(A)),\t$(method))" for (A, b) in ((form, data), (form, nt), (X, y), (sX, y)), method in (:cg, :chol)
        name  = if A==form; "formula" elseif A==X; "dense  " else "sparse " end
        name *= if method==:cg; ",  cg" else ",chol" end
        # use the dispersion from GLM to ensure that the loglikelihood is correct
        m = fit(RobustLinearModel, A, b, est1; method=method, verbose=false, initial_scale=λlm)
        β = copy(coef(m))
        println("rlm($name): ", β)
        @test_nowarn println(m)
        @test isapprox(coef(m1), β; rtol=1e-5)

        # refit
        refit!(m, y; verbose=false, initial_scale=:extrema)
        println("$m")
        @test all(coef(m) .== β)

        # make sure the methods for RobustLinearModel are well defined
        @testset "method: $(f)" for f in funcs
            if f in (Estimator, weights, workingweights, islinear, isfitted,
                     leverage, modelmatrix, projectionmatrix, scale, wobs)
                # method is not defined in GLM
                @test_nowarn f(m)
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

        # TauEstimator interface
        m3 = fit(RobustLinearModel, A, b, TauEstimator{TukeyLoss}(); method=method, initial_scale=λlm)
        @test_nowarn tauscale(m3)

        # later fit!
        m2 = fit(RobustLinearModel, A, b, est1; method=method, dofit=false)
        @test all(0 .== coef(m2))
        fit!(m2; verbose=false, initial_scale=:mad)
        @test all(β .== coef(m2))

        @testset "Handling of $(typemod) values" for (typemod, conv_func) in (("missing", allowmissing), ("complex", complex))
            # check that Missing eltype is routed correctly
            if isa(A, FormulaTerm)
                if isa(b, NamedTuple)
                    b_mod = NamedTuple(k=>conv_func(v) for (k,v) in pairs(b))
                elseif isa(b, DataFrame)
                    b_mod = DataFrame((v = Tables.getcolumn(b, k); k=>(eltype(v) <: Real ? conv_func(v) : v)) for k in Tables.columnnames(b))
                else
                    b_mod = nothing
                end
                @test_throws ArgumentError fit(RobustLinearModel, A, b_mod, est1)
            else
                A_mod = conv_func(A)
                b_mod = conv_func(b)
                @test_throws MethodError fit(RobustLinearModel, A_mod, b, est1)
                @test_throws MethodError fit(RobustLinearModel, A, b_mod, est1)
                @test_throws MethodError fit(RobustLinearModel, A_mod, b_mod, est1)
            end
        end
    end
end


