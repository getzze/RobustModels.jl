
using Statistics: median

funcs = (;
          mean=mean,
          std=std,
          var=var,
          sem=sem,
          mean_and_std=mean_and_std,
          mean_and_var=mean_and_var,
          mean_and_sem=mean_and_sem
        )

x = [
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


x[1:2] .= 50
xx = x[3:end]

import RobustModels: mean_and_sem
mean_and_sem(x) = (m=mean(x); s=sem(x); (m,s))


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
            @test_skip all(func(xx) .<= s)
        else
            @test all(func(xx) .<= s .<= func(x))
        end
    end
end

@testset "robust univariate statistics: Bounded estimator $(typeest)" for typeest in (SEstimator, MMEstimator, TauEstimator)
    est = typeest{TukeyLoss}()

    @testset "method: $(name)" for (name, func) in pairs(funcs)
        s = func(est, x)
#        println("statistics $(name): $(round.(s; digits=4)) ≈ $(round.(func(xx); digits=4)) (with outliers removed)")
        @test all(isapprox.(s, func(xx); rtol=2))
    end
end
