
using Random: MersenneTwister
using Statistics: median
using RobustModels: mean_and_sem, compatdims

funcs = (:mean, :std, :var, :sem, :mean_and_std, :mean_and_var, :mean_and_sem)

seed = 306
#seed = rand(1:1000)
#println("Random generator with MersenneTwister(seed=$(seed))")

xorig = randn(MersenneTwister(seed), 100)
x = copy(xorig)
x[1:10] .= 50


@testset "robust univariate statistics: MEstimator{$(lossname)}" for lossname in ("L2Loss", "L1Loss", "HuberLoss", "TukeyLoss")
    typeloss = getproperty(RobustModels, Symbol(lossname))
    est = MEstimator{typeloss}()

    @testset "method: $(name)" for name in funcs
        func = getproperty(RobustModels, Symbol(name))
        if lossname == "L1Loss"
            @test_throws ArgumentError func(est, x)
            continue
        end

        # Compute robust statistics
        s = func(est, x)
        if lossname == "L2Loss"
            @test all(isapprox.(s, func(x); rtol=1e-7))
        elseif lossname == "TukeyLoss"
            @test all(s .<= func(x))
            # the estimate is better than when the outliers are removed...
            @test_skip all(func(xorig) .<= s)
        else
            @test all(func(xorig) .<= s .<= func(x))
        end
    end
end


@testset "robust univariate statistics: Bounded estimator $(typeest)" for typeest in (SEstimator, MMEstimator, TauEstimator)
    est = typeest{TukeyLoss}()

    @testset "method: $(name)" for name in funcs
        func = getproperty(RobustModels, Symbol(name))

        resampling_options = Dict(:rng => MersenneTwister(seed))
        s = func(est, x; resampling_options)
#        println("statistics $(name): $(round.(s; digits=4)) ≈ $(round.(func(xorig); digits=4)) (with outliers removed)")
        @test all(isapprox.(s, func(xorig); rtol=2))
    end
end


########################################################################
##  With iterables
########################################################################

d = Base.values(Dict(:a=>1, :b=>2, :c=>3))
g = (i for i in 1:3)
est = MEstimator{L2Loss}()

@testset "robust univariate statistics: iterable $(typeof(a).name.name)" for a in (d, g)
    @test all(isapprox.(mean(est, a), mean(a); rtol=1e-7))
end


########################################################################
##  Arrays and dims
########################################################################

# use the L2Loss to be directly compared to the non-robust functions
# it should give exactly the same results
est = MEstimator{L2Loss}()

yorig = randn(MersenneTwister(seed), 306)
y1 = reshape(yorig, (306, 1))
y2 = reshape(yorig, (1, 306))
y3 = reshape(yorig, (1, 3, 102))
y4 = reshape(yorig, (17, 18, 1))

@testset "robust univariate statistics: Array size: $(size(a))" for a in (y, y1, y2, y3, y4)
    @testset "dims=$(dims)" for dims in (1, 2, (1,), (1,2), (3,1), 4, (:))
        ## Mean
        m = @test_nowarn mean(est, a; dims=dims)

        # non-robust mean
        res = mean(a; dims=dims)
        @test all(isapprox.(m, res; rtol=1e-7, nans=true))

        ## Dispersion: std, var, sem
        for disp_name in (:std, :var, :sem)
            func = getproperty(RobustModels, Symbol(disp_name))

            ## Test only the dispersion
            s = @test_nowarn func(est, a; dims=dims)

            ## Test `mean_and_<dispersion> == (mean, <dispersion>)`
            func_tup = getproperty(RobustModels, Symbol("mean_and_" * String(disp_name)))
            ms = @test_nowarn func_tup(est, a; dims=dims)
            @test length(ms) == 2
            @test ms[1] ≈ m
            @test ms[2] ≈ s  nans=true

            # non-robust dispersion
            if dims === Colon()
                # apply on a flatten array
                res = func(vec(a))
            elseif disp_name == :sem
                # `StatsBase.sem` does not allow the `dims` keyword
                dims = compatdims(ndims(a), dims)
                if isnothing(dims)
                    continue
                end
                res = mapslices(func, a; dims=dims)
            else
                # call the non-robust version with dims
                res = func(a; dims=dims)
            end

            @test all(isapprox.(s, res; rtol=1e-7, nans=true))
        end
    end
end

