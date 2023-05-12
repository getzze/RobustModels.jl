
loss1 = RobustModels.L2Loss()
est1 = MEstimator(loss1)
est2 = MMEstimator(TukeyLoss)
pen1 = RobustModels.NoPenalty()
λ = 10_000.0
pen2 = RobustModels.RangedPenalties([2:RobustModels.End()], [RobustModels.SquaredL2Penalty(λ)])

m00 = fit(LinearModel, form, data)
m01 = fit(RobustLinearModel, form, data, est1; method=:chol, initial_scale=1)
m02 = fit(RobustLinearModel, form, data, est1; method=:chol, initial_scale=1, ridgeλ=λ)
m03 = fit(RobustLinearModel, form, data, est2)
σ = scale(m03)

@testset "Θ-IPOD: L2Loss method, no penalty ($(pen))" for (pen, methods) in zip((nothing, pen1), (nopen_methods, pen_methods))
    rtol = isnothing(pen) ? 1e-6 : 1e-4

    @testset "solver method $(method)" for method in methods
        if method === :fista
            rtol = 1e-1
        end

        # Formula, dense and sparse entry
        @testset "data type: $(typeof(A))" for (A, b) in ((form, data), (form, nt), (X, y), (sX, y))
            name  = "Θ-IPOD(L2Loss, $(pen); method=$(method)),\t"
            name *= if A==form; "formula" elseif A==X; "dense  " else "sparse " end

            m1 = fit(IPODRegression, A, b, loss1, pen; method=method, initial_scale=1)
            @test_nowarn ipod(A, b, loss1, pen; method=method, initial_scale=1)

            @test all(iszero, outliers(m1))
            @test all(isfinite.(coef(m1)))

            VERBOSE && println("\n\t\u25CF $(name)")
            VERBOSE && println(" rlm : ", coef(m01))
            VERBOSE && println("ipod($(method)) : ", coef(m1))
            @test isapprox(coef(m1), coef(m01); rtol=rtol)

            # Test printing the model
            @test_nowarn println(m1)

            # make sure that it is not a TableRegressionModel
            @test !isa(m1, TableRegressionModel)

            # refit
            @test_nowarn refit!(m1)

            # interface
            @testset "method: $(f)" for f in interface_methods
                if f == workingweights
                    # Not defined for IPODRegression
                    continue
                end

                # make sure the methods for IPODRegression give the same results as RobustLinearModel
                var0 = f(m01)
                var1 = f(m1)
                if f == hasformula
                    # m01 is defined from a formula
                    @test var1 == (A isa FormulaTerm)
                else
                    @test isapprox(var0, var1; rtol=rtol)
                end
            end
        end
    end
end


@testset "Θ-IPOD: robust loss, no penalty ($(pen))" for (pen, methods) in zip((nothing, pen1), (nopen_methods, pen_methods))
    @testset "solver method $(method)" for method in methods
        @testset "loss $(loss_name)" for loss_name in ("Huber", "Hampel", "Tukey")
            typeloss = getproperty(RobustModels, Symbol(loss_name * "Loss"))
            l = typeloss()
            est = MEstimator(l)

            # Formula, dense and sparse entry
            @testset "data type: $(typeof(A))" for (A, b) in ((X, y), (sX, y))
                name  = "Θ-IPOD($(l), $(pen); method=$(method)),\t"
                name *= if A==form; "formula" elseif A==X; "dense  " else "sparse " end

                m0 = fit(RobustLinearModel, A, b, est; method=:chol, initial_scale=σ)
                m1 = fit(IPODRegression, A, b, l, pen; method=method, initial_scale=σ)
                β1 = copy(coef(m1))
                γ1 = copy(outliers(m1))

                @test any(!iszero, outliers(m1))
                @test all(isfinite.(coef(m1)))

                VERBOSE && println("\n\t\u25CF $(name)")
                VERBOSE && println(" rlm : ", coef(m0))
                VERBOSE && println("ipod($(method)) : ", coef(m1))
                @test coef(m0) ≈ coef(m1)  rtol=0.1
                @test stderror(m0) ≈ stderror(m1)  rtol=0.5

                # Test printing the model
                @test_nowarn println(m1)

                # make sure that it is not a TableRegressionModel
                @test !isa(m1, TableRegressionModel)

                # refit
                @test_nowarn refit!(m1)
                @test coef(m1) ≈ β1
                @test outliers(m1) ≈ γ1
            end
        end
    end
end



@testset "Θ-IPOD: L2Loss method, Ridge penalty" begin
    rtol = 1e-5
    @testset "solver method $(method)" for method in pen_methods
        if method === :fista
            rtol = 1e-2
        end

        # Formula, dense and sparse entry
        @testset "data type: $(typeof(A))" for (A, b) in data_tuples
            name  = "Θ-IPOD(L2Loss, $(pen2); method=$(method)),\t"
            name *= if A==form; "formula" elseif A==X; "dense  " else "sparse " end

            m2 = fit(IPODRegression, A, b, loss1, pen2; method=method, initial_scale=1)

            @test all(iszero, outliers(m2))
            @test all(isfinite.(coef(m2)))

            VERBOSE && println("\n\t\u25CF $(name)")
            VERBOSE && println(" rlm : ", coef(m02))
            VERBOSE && println("ipod($(method)) : ", coef(m2))
            @test isapprox(coef(m2), coef(m02); rtol=rtol)

            # interface
            @testset "method: $(f)" for f in interface_methods
                if f == workingweights
                    # Not defined for IPODRegression
                    continue
                end

                # make sure the methods for IPODRegression give the same results as RobustLinearModel
                var0 = f(m02)
                var1 = f(m2)
                if f == hasformula
                    # m0 is defined from a formula
                    @test var1 == (A isa FormulaTerm)
                elseif f in (dof, dof_residual)
                    # Ridge dof is smaller than the unpenalized regression
                    # IPOD dof is the same as the unpenalized IPOD
                    @test var1 == f(m01)
                elseif f in (dispersion, stderror, vcov, leverage)
                    @test all(abs.(var1) .>= abs.(var0))
                elseif f in (leverage_weights,)
                    @test all(abs.(var1) .<= abs.(var0))
                elseif f in (confint, )
                    @test isapprox(var0, var1; rtol=1e-1)
                elseif f in (projectionmatrix, )
                    continue
                else
                    @test isapprox(var0, var1; rtol=rtol)
                end
            end
        end
    end
end


@testset "Θ-IPOD: L2Loss method, $(pen_name) penalty" for pen_name in penalties
    typepenalty = getproperty(RobustModels, Symbol(pen_name * "Penalty"))
    pen = typepenalty(λ)
    sep_pen1 = RobustModels.RangedPenalties([1:RobustModels.End()], [pen])
    sep_pen2 = RobustModels.RangedPenalties([2:RobustModels.End()], [pen])
    kwargs = (; initial_scale=1, maxiter=1_000)

    m0 = fit(RobustLinearModel, X, y, pen; method=:auto, kwargs...)

    # Formula, dense and sparse entry
    @testset "Omit intercept, data type: $(typeof(A))" for (A, b) in data_tuples
        name  = "Θ-IPOD(L2Loss, $(pen); method=:auto),\t"
        name *= if A==form; "formula" elseif A==X; "dense  " else "sparse " end

        m1 = fit(IPODRegression, A, b, loss1, pen; kwargs...)
        m2 = fit(IPODRegression, A, b, loss1, sep_pen2; kwargs...)
        m3 = fit(IPODRegression, A, b, loss1, sep_pen1; kwargs...)
        m4 = fit(IPODRegression, A, b, loss1, pen; penalty_omit_intercept=false, kwargs...)
        m5 = fit(IPODRegression, A, b, loss1, sep_pen1; penalty_omit_intercept=false, kwargs...)

        @test all(isfinite.(coef(m1)))
        @test all(isfinite.(coef(m2)))
        @test all(isfinite.(coef(m3)))
        @test all(isfinite.(coef(m4)))
        @test all(isfinite.(coef(m5)))
        @test all(iszero, outliers(m1))
        @test all(iszero, outliers(m2))
        @test all(iszero, outliers(m3))
        @test all(iszero, outliers(m4))
        @test all(iszero, outliers(m5))

        @test penalty(m0) == penalty(m1)
        @test penalty(m0) == penalty(m2)
        @test penalty(m0) == penalty(m3)
        @test penalty(m4) != penalty(m5)
        @test penalty(m0) != penalty(m4)

        @test coef(m0) ≈ coef(m1)
        @test coef(m0) ≈ coef(m2)
        @test coef(m0) ≈ coef(m3)
        @test coef(m4) ≈ coef(m5)
        @test coef(m4)[2] ≈ 0  atol=0.1
        @test coef(m5)[2] ≈ 0  atol=0.1

        VERBOSE && println("\n\t\u25CF $(name)")
        VERBOSE && println("ipod($(pen), $(method)) : ", coef(m1))
    end

    @testset "solver method $(method)" for method in pen_methods
        m0 = fit(RobustLinearModel, X, y, pen; method=method, kwargs...)

        # Formula, dense and sparse entry
        @testset "data type: $(typeof(A))" for (A, b) in data_tuples
            name  = "Θ-IPOD(L2Loss, $(pen); method=$(method)),\t"
            name *= if A==form; "formula" elseif A==X; "dense  " else "sparse " end

            m1 = fit(IPODRegression, A, b, loss1, pen; method=method, kwargs...)

            @test all(isfinite.(coef(m1)))
            @test all(iszero, outliers(m1))

            @test penalty(m0) == penalty(m1)
            @test coef(m0) ≈ coef(m1)
            @test coef(m1)[2] ≈ 0  atol=0.1

            VERBOSE && println("\n\t\u25CF $(name)")
            VERBOSE && println("ipod($(pen), $(method)) : ", coef(m1))
        end
    end
end
