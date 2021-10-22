
using Statistics: median
using RobustModels: mean_and_sem

funcs = (;
          mean=mean,
          std=std,
          var=var,
          sem=sem,
          mean_and_std=mean_and_std,
          mean_and_var=mean_and_var,
          mean_and_sem=mean_and_sem
        )

xorig = [
  0.529,
 -0.921,
 -0.247,
  0.689,
 -0.254,
 -1.087,
  0.067,
  0.679,
  0.089,
 -0.278,
  1.327,
 -0.571,
 -0.559,
 -0.491,
  0.237,
  1.196,
 -0.807,
  0.35 ,
  0.259,
  1.006,
  0.257,
]
#x = randn(100)

x = copy(xorig)
x[1:2] .= 50


@testset "robust univariate statistics: MEstimator{$(lossname)}" for lossname in ("L2Loss", "L1Loss", "HuberLoss", "TukeyLoss")
    typeloss = getproperty(RobustModels, Symbol(lossname))
    est = MEstimator{typeloss}()

    @testset "method: $(name)" for (name, func) in pairs(funcs)
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

    @testset "method: $(name)" for (name, func) in pairs(funcs)
        s = func(est, x)
#        println("statistics $(name): $(round.(s; digits=4)) ≈ $(round.(func(xorig); digits=4)) (with outliers removed)")
        @test all(isapprox.(s, func(xorig); rtol=2))
    end
end

# define the Estimator for the next tests
est = MEstimator{L2Loss}()

x1 = reshape(xorig, (21, 1))
x2 = reshape(xorig, (1, 21))
x3 = reshape(xorig, (1, 3, 7))
x4 = reshape(xorig, (7, 3, 1))

@testset "robust univariate statistics: Array size: $(size(a))" for a in (x, x1, x2, x3, x4)
    @testset "method: $(name)" for (name, func) in pairs(funcs)
        @testset "dims=$(dims)" for dims in (1, 2, (1,), (1,2), (3,1), 4, (:))
            s = func(est, a; dims=dims)
            if name in (:mean, :std, :var)
                res = func(a; dims=dims)
                @test all(isapprox.(s, res; rtol=1e-7, nans=true))
            elseif name in (:mean_and_std, :mean_and_var)
                # tuples are not allowed for the `dims` arg in StatsBase.mean_and_var
                if isa(dims, Tuple) && length(dims) > 1
                    continue
                end
                res = if dims===(:); func(a) else func(a, first(dims)) end
                @test all(isapprox.(s, res; rtol=1e-7, nans=true))
            end
        end
    end
end

d = Base.values(Dict(:a=>1, :b=>2, :c=>3))
g = (i for i in 1:3)

@testset "robust univariate statistics: iterable $(typeof(a).name.name)" for a in (d, g)
    @test all(isapprox.(mean(est, a), mean(a); rtol=1e-7))
end
