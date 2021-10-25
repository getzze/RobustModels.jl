# Robust linear models in Julia

| Documentation | CI Status | Coverage |
|:-------------------:|:------------------:|:-----------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url] | [![][CI-img]][CI-url] | [![][codecov-img]][codecov-url] |

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://getzze.github.io/RobustModels.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://getzze.github.io/RobustModels.jl/stable

[CI-img]: https://github.com/getzze/RobustModels.jl/workflows/CI/badge.svg
[CI-url]: https://github.com/getzze/RobustModels.jl/actions

[codecov-img]: https://img.shields.io/codecov/c/github/getzze/RobustModels.jl?label=Codecov&logo=codecov
[codecov-url]: https://codecov.io/gh/getzze/RobustModels.jl/branch/master


This package defines robust linear models using the interfaces from
[StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl) and
[StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl).
It defines an `AbstractRobustModel` type as a subtype of `RegressionModel` and
it defines the methods from the statistical model API like `fit`/`fit!`.

A _robust model_ is a regression model, meaning it finds a relationship between one or
several _covariates/independent variables_ `X` and a _response/dependent_ variable `y`.
Contrary to the ordinary least squares estimate, robust regression mitigates the influence
of _outliers_ in the data. A standard view for `RobustLinearModel` is to consider that the residuals
are distributed according to a _contaminated_ distribution, that is a _mixture model_
of the distribution of interest `F` and a contaminating distribution `Δ` with higher variance.
Therefore the residuals `r` follow the distribution `r ~ (1-ε) F + ε Δ`, with `0≤ε<1`.

This package implements:

* M-estimators
* S-estimators
* MM-estimators
* τ-estimators
* MQuantile regression (e.g. Expectile regression)
* Robust Ridge regression (using any of the previous estimator)
* Quantile regression using interior point method

## Installation

```julia
julia>] add RobustModels
```

To install the last development version:

```julia
julia>] add RobustModels#main
```

## Usage

The prefered way of performing robust regression is by calling the `rlm` function:

`m = rlm(X, y, MEstimator{TukeyLoss}(); initial_scale=:mad)`

For quantile regression, use `quantreg`:

`m = quantreg(X, y; quantile=0.5)`

For robust version of `mean`, `std`, `var` and `sem` statistics, specify the estimator as first argument.
Use the `dims` keyword for computing the statistics along specific dimensions.
The following functions are also implemented: `mean_and_std`, `mean_and_var` and `mean_and_sem`.

`xm = mean(MEstimator{HuberLoss}(), x; dims=2)`

`(xm, sm) = mean_and_std(TauEstimator{YohaiZamarLoss}(), x)`


## Examples

```julia
using RDatasets: dataset
using StatsModels
using RobustModels

data = dataset("robustbase", "Animals2")
data.logBrain = log.(data.Brain)
data.logBody = log.(data.Body)
form = @formula(logBrain ~ 1 + logBody)

## M-estimator using Tukey estimator
m1 = rlm(form, data, MEstimator{TukeyLoss}(); method=:cg, initial_scale=:mad)

## MM-estimator using Tukey estimator
m2 = rlm(form, data, MMEstimator{TukeyLoss}(); method=:cg, initial_scale=:L1)

## M-estimator using Huber estimator and correcting for covariate outliers using leverage
m3 = rlm(form, data, MEstimator{HuberLoss}(); method=:cg, initial_scale=:L1, correct_leverage=true)

## M-estimator using Huber estimator, providing an initial scale estimate and using Cholesky method of solving.
m4 = rlm(form, data, MEstimator{HuberLoss}(); method=:chol, initial_scale=15.0)

## S-estimator using Tukey estimator
m5 = rlm(form, data, SEstimator{TukeyLoss}(); σ0=:mad)

## τ-estimator using Tukey estimator
m6 = rlm(form, data, TauEstimator{TukeyLoss}(); initial_scale=:L1)

## τ-estimator using YohaiZamar estimator and resampling to find the global minimum
opts = Dict(:Npoints=>10, :Nsteps_β=>3, :Nsteps_σ=>3)
m7 = rlm(form, data, TauEstimator{YohaiZamarLoss}(); initial_scale=:L1, resample=true, resampling_options=opts)

## Expectile regression for τ=0.8
m8 = rlm(form, data, GeneralizedQuantileEstimator{L2Loss}(0.8))
#m8 = rlm(form, data, ExpectileEstimator(0.8))

## Refit with different parameters
refit!(m8; quantile=0.2)

## Robust ridge regression
m9 = rlm(form, data, MEstimator{TukeyLoss}(); initial_scale=:L1, ridgeλ=1.0)

## Quantile regression solved by linear programming interior point method
m10 = quantreg(form, data; quantile=0.2)
refit!(m10; quantile=0.8)
;

# output

```

## Theory

### M-estimators

With ordinary least square (OLS), the objective function is, from maximum likelihood estimation:

`L = ½ Σᵢ (yᵢ - 𝒙ᵢ 𝜷)² = ½ Σᵢ rᵢ²`

where `yᵢ` are the values of the response variable, `𝒙ᵢ` are the covectors of individual covariates
(rows of the model matrix `X`), `𝜷` is the vector of fitted coefficients and `rᵢ` are the individual residuals.

A `RobustLinearModel` solves instead for the following loss function: `L' = Σᵢ ρ(rᵢ)`
(more precisely `L' = Σᵢ ρ(rᵢ/σ)` where `σ` is an (robust) estimate of the standard deviation of the residual).
Several loss functions are implemented:

- `L2Loss`: `ρ(r) = ½ r²`, like ordinary OLS.
- `L1Loss`: `ρ(r) = |r|`, non-differentiable estimator also know as _Least absolute deviations_. Prefer the `QuantileRegression` solver.
- `HuberLoss`: `ρ(r) = if (r<c); ½(r/c)² else |r|/c - ½ end`, convex estimator that behaves as `L2` cost for small residuals and `L1` for large esiduals and outliers.
- `L1L2Loss`: `ρ(r) = √(1 + (r/c)²) - 1`, smooth version of `HuberLoss`.
- `FairLoss`: `ρ(r) = |r|/c - log(1 + |r|/c)`, smooth version of `HuberLoss`.
- `LogcoshLoss`: `ρ(r) = log(cosh(r/c))`, smooth version of `HuberLoss`.
- `ArctanLoss`: `ρ(r) = r/c * atan(r/c) - ½ log(1+(r/c)²)`, smooth version of `HuberLoss`.
- `CauchyLoss`: `ρ(r) = log(1+(r/c)²)`, non-convex estimator, that also corresponds to a Student's-t distribution (with fixed degree of freedom). It suppresses outliers more strongly but it is not sure to converge.
- `GemanLoss`: `ρ(r) = ½ (r/c)²/(1 + (r/c)²)`, non-convex and bounded estimator, it suppresses outliers more strongly.
- `WelschLoss`: `ρ(r) = ½ (1 - exp(-(r/c)²))`, non-convex and bounded estimator, it suppresses outliers more strongly.
- `TukeyLoss`: `ρ(r) = if r<c; ⅙(1 - (1-(r/c)²)³) else ⅙ end`, non-convex and bounded estimator, it suppresses outliers more strongly and it is the prefered estimator for most cases.
- `YohaiZamarLoss`: `ρ(r)` is quadratic for `r/c < 2/3` and is bounded to 1; non-convex estimator, it is optimized to have the lowest bias for a given efficiency.

The value of the tuning constants `c` are optimized for each estimator so the M-estimators have a high efficiency of 0.95. However, these estimators have a low breakdown point.

### S-estimators

Instead of minimizing `Σᵢ ρ(rᵢ/σ)`, S-estimation minimizes the estimate of the squared scale `σ²` with the constraint that: `1/n Σᵢ ρ(rᵢ/σ) = 1/2`.
S-estimators are only defined for bounded estimators, like `TukeyLoss`.
These estimators have low efficiency but a high breakdown point of 1/2, by choosing the tuning constant `c` accordingly.

### MM-estimators

It is a two-pass estimation, 1) Estimate `σ` using an S-estimator with high breakdown point and 2) estimate `𝜷` using an M-estimator with high efficiency.
It results in an estimator with high efficiency and high breakdown point.

### τ-estimators

Like MM-estimators, τ-estimators combine a high efficiency with a high breakdown point.
Similar to S-estimators, it minimizes a scale estimate:
`τ² = σ² (2/n Σᵢρ₂(rᵢ/σ))` where `σ` is an M-scale estimate, solution of `1/n Σᵢ ρ₁(rᵢ/σ) = 1/2`.
Finding the minimum of a τ-estimator is similar to the procedure for an S-estimator with a special weight function
that combines both functions `ρ₁` and `ρ₂`. To ensure a high breakdown point and high efficiency,
the two loss functions should be the same but with different tuning constants.

### MQuantile-estimators

Using an asymmetric variant of the `L1Estimator`, quantile regression is performed
(although the `QuantileRegression` solver should be prefered because it gives an exact solution).
Identically, with an M-estimator using an asymetric version of the loss function,
a generalization of quantiles is obtained. For instance, using an asymetric `L2Loss` results in _Expectile Regression_.

### Robust Ridge regression

This is the robust version of the ridge regression, adding a penalty on the coefficients.
The objective function to solve is `L = Σᵢ ρ(rᵢ/σ) + λ Σⱼ|βⱼ|²`, where the sum of squares of
coefficients does not include the intersect `β₀`.
Robust ridge regression is implemented for all the estimators (not for `quantreg`).
By default, all the coefficients (except the intercept) have the same penalty, which assumes that
all the feature variables have the same scale. If it is not the case, use a robust estimate of scale
to normalize every column of the model matrix `X` before fitting the regression.

### Quantile regression

_Quantile regression_ results from minimizing the following objective function:
`L = Σᵢ wᵢ|yᵢ - 𝒙ᵢ 𝜷| = Σᵢ wᵢ(rᵢ) |rᵢ|`,
where `wᵢ = ifelse(rᵢ>0, τ, 1-τ)` and `τ` is the quantile of interest. `τ=0.5` corresponds to _Least Absolute Deviations_.

This problem is solved exactly using linear programming techniques,
specifically, interior point methods using the internal API of [Tulip](https://github.com/ds4dm/Tulip.jl).
The internal API is considered unstable, but it results in a much lighter dependency than
including the [JuMP](https://github.com/JuliaOpt/JuMP.jl) package with Tulip backend.

## Credits

This package derives from the [RobustLeastSquares](https://github.com/FugroRoames/RobustLeastSquares.jl)
package for the initial implementation, especially for the Conjugate Gradient
solver and the definition of the M-Estimator functions.

Credits to the developpers of the [GLM](https://github.com/JuliaStats/GLM.jl)
and [MixedModels](https://github.com/JuliaStats/MixedModels.jl) packages
for implementing the Iteratively Reweighted Least Square algorithm.

## References

- "Robust Statistics: Theory and Methods (with R)", 2nd Edition, 2019, R. Maronna, R. Martin, V. Yohai, M. Salibián-Barrera
