
@testset "Quantile regression: low-level function" begin
    τs = range(0.1, 0.9; step=0.1)
    βs = hcat(map(τ -> RobustModels.interiormethod(X, y, τ)[1], τs)...)
    VERBOSE && println("Coefficients: $(vcat(τs', βs))")
    @test size(βs) == (size(X, 2), length(τs))
end


@testset "Quantile regression: fit method" begin
    τ = 0.5
    # Formula, dense and sparse entry  and methods :cg and :chol
    @testset "Argument type: $(typeof(A))" for (A, b) in data_tuples
        m1 = fit(QuantileRegression, A, b; quantile=τ, verbose=false)
        m2 = quantreg(A, b; quantile=τ, verbose=false)
        @test_nowarn println(m2)
        @test all(coef(m1) .== coef(m2))

        # make sure that it is not a TableRegressionModel
        @test !isa(m1, TableRegressionModel)
        @test !isa(m2, TableRegressionModel)

        # refit
        β = copy(coef(m2))
        refit!(m2, y; quantile=τ, verbose=false)
        @test all(coef(m2) .== β)

        # interface
        @testset "interface method: $(f)" for f in interface_methods
            # make sure the method is defined
            @test_nowarn robvar = f(m1)
        end

        # later fit!
        m3 = fit(QuantileRegression, A, b; quantile=τ, dofit=false)
        @test all(0 .== coef(m3))
        fit!(m3; verbose=false)
        @test all(β .== coef(m3))

        # leverage weights
        @test_nowarn refit!(m3; correct_leverage=true)

        # handling of missing values
        @testset "Handling of missing values" begin
            # check that Missing eltype is routed correctly
            if isa(A, FormulaTerm)
                if isa(b, NamedTuple)
                    b_missing = NamedTuple(k => allowmissing(v) for (k, v) in pairs(b))
                elseif isa(b, DataFrame)
                    b_missing = allowmissing(b)
                else
                    b_missing = nothing
                end
                @test_throws ArgumentError fit(QuantileRegression, A, b_missing)
                @test_nowarn fit(QuantileRegression, A, b_missing; dropmissing=true)
            else
                A_missing = allowmissing(A)
                b_missing = allowmissing(b)
                @test_throws ArgumentError fit(QuantileRegression, A_missing, b)
                @test_throws ArgumentError fit(QuantileRegression, A, b_missing)
                @test_throws ArgumentError fit(QuantileRegression, A_missing, b_missing)

                @test_nowarn fit(QuantileRegression, A_missing, b; dropmissing=true)
                @test_nowarn fit(QuantileRegression, A, b_missing; dropmissing=true)
                @test_nowarn fit(QuantileRegression, A_missing, b_missing; dropmissing=true)
            end
        end
    end
end

@testset "Quantile regression: different quantiles" begin
    τs = range(0.1, 0.9; step=0.1)
    m2 = fit(QuantileRegression, form, data; quantile=0.5, verbose=false)

    @testset "$(τ) quantile" for τ in τs
        m1 = fit(QuantileRegression, form, data; quantile=τ, verbose=false)
        @test_nowarn println(m1)
        β = coef(m1)
        res = residuals(m1)
        ## The quantile regression line exactly passes through p points, with p number of columns of X.
        @test count(x -> isapprox(x, 0; atol=1e-7), res) == length(β)
        #        @test count(iszero, res) == length(β)

        # refit with new quantile
        refit!(m2; quantile=τ)
        @test all(coef(m1) .== coef(m2))
    end
end

@testset "Quantile regression: sparsity estimation" begin
    m2 = fit(QuantileRegression, form, data; quantile=0.25, verbose=false)
    s = RobustModels.location_variance(m2, false)

    τs = range(0.25, 0.75; step=0.25)
    @testset "(q, method, kernel): $(τ), $(method), $(kernel)" for τ in τs,
        method in (:jones, :bofinger, :hall_sheather),
        kernel in (:epanechnikov, :triangle, :window)

        if τ != m2.τ
            refit!(m2; quantile=τ)
            s = RobustModels.location_variance(m2, false)
        end

        si = RobustModels.location_variance(
            m2, false; bw_method=method, α=0.05, kernel=kernel
        )
        if method == :jones && kernel == :epanechnikov
            @test isapprox(s, si; rtol=1e-4)
        else
            @test !isapprox(s, si; rtol=1e-4)
        end
    end
end
