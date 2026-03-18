# Examples

```@meta
DocTestSetup = quote
    using DataFrames, GLM
    using RobustModels
end
```

## Robust linear regression

Here we show examples fitting simple data with ``Y = 2X`` in the presence of outliers. We first use the [`MEstimator`](@ref) with the [`L2Loss`](@ref) function.

```jldoctest example
julia> using DataFrames, RobustModels

julia> data = DataFrame(X=[1,2,3,4,5,6], Y=[2,4,7,8,9,13])
6×2 DataFrame
 Row │ X      Y
     │ Int64  Int64
─────┼──────────────
   1 │     1      2
   2 │     2      4
   3 │     3      7
   4 │     4      8
   5 │     5      9
   6 │     6     13

julia> ols = rlm(@formula(Y ~ X), data, MEstimator{L2Loss}())
Robust regression with M-Estimator(L2Loss())

Y ~ 1 + X

Coefficients:
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)  0.0666667    0.781533   0.09    0.9361   -2.10322    2.23655
X            2.02857      0.200679  10.11    0.0005    1.4714     2.58575
─────────────────────────────────────────────────────────────────────────

julia> round.(stderror(ols), digits=5)
2-element Vector{Float64}:
 0.78153
 0.20068

julia> round.(predict(ols), digits=5)
6-element Vector{Float64}:
  2.09524
  4.12381
  6.15238
  8.18095
 10.20952
 12.2381
```

Now we add an additional outlier and we estimate the model parameters with the [`MMEstimator`](@ref) using the [`TukeyLoss`](@ref) function.

```jldoctest example
julia> data[5, :Y] = 1; data
6×2 DataFrame
 Row │ X      Y
     │ Int64  Int64
─────┼──────────────
   1 │     1      2
   2 │     2      4
   3 │     3      7
   4 │     4      8
   5 │     5      1
   6 │     6     13

julia> rlm(@formula(Y ~ X), data, MMEstimator{TukeyLoss}(); σ0=:mad)
Robust regression with MM-Estimator(TukeyLoss(1.5476), TukeyLoss(4.685))

Y ~ 1 + X

Coefficients:
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)  -0.184561    0.514249  -0.36    0.7378   -1.61235    1.24322
X             2.18005     0.14112   15.45    0.0001    1.78824    2.57186
─────────────────────────────────────────────────────────────────────────
```

And now with the [`SEstimator`](@ref) using the [`TukeyLoss`](@ref) function.

```jldoctest example
julia> rlm(@formula(Y ~ X), data, SEstimator{TukeyLoss}(); σ0=:mad)
Robust regression with S-Estimator(TukeyLoss(1.5476))

Y ~ 1 + X

Coefficients:
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)  -0.331551    0.393179  -0.84    0.4466   -1.42319   0.760088
X             2.20275     0.105577  20.86    <1e-04    1.90962   2.49588
─────────────────────────────────────────────────────────────────────────
```