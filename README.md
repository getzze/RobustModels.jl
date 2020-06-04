# Robust linear models in Julia

| CI Status | Coverage |
|:------------------:|:-----------------:|
| [![][travis-img]][travis-url] | [![][coveralls-img]][coveralls-url] [![][codecov-img]][codecov-url] |

[travis-img]: https://img.shields.io/travis/com/getzze/RobustModels.jl?label=Travis&logo=travis
[travis-url]: https://travis-ci.com/getzze/RobustModels.jl

[coveralls-img]: https://img.shields.io/coveralls/github/getzze/RobustModels.jl?label=Coveralls&logo=coveralls
[coveralls-url]: https://coveralls.io/github/getzze/RobustModels.jl?branch=master

[codecov-img]: https://img.shields.io/codecov/c/github/getzze/RobustModels.jl?label=Codecov&logo=codecov
[codecov-url]: https://codecov.io/gh/getzze/RobustModels.jl/branch/master


This package defines robust linear models using the interfaces from [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl) and [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl). It defines an `AbstractRobustModel` type as a subtype of `RegressionModel` and it defines the methods from the statistical model API like `fit`/`fit!`.

A _robust model_ is a regression model, meaning it finds a relationship between one or several _covariates/independent variables_ `X` and a _response/dependent_ variable `y`. Contrary to the ordinary least squares estimate, robust regression mitigates the influence of _outliers_ in the data.
A standard view for `RobustLinearModel` is to consider that the residuals are distributed according to a _contaminated_ distribution, that is a _mixture model_ of the distribution of interest `F` and a contaminating distribution `Δ` with higher variance. Therefore the residuals `r` follow the distribution `r ~ (1-ε) F + ε Δ`, with `0≤ε<1`.

This package implements:
* M-estimators
* S-estimators
* MM-estimators
* τ-estimators
* MQuantile regression (e.g. Expectile regression)
* Quantile regression using interior point method

## Installation

```julia
]add https://github.com/getzze/RobustModels.jl
```

## Usage

The prefered way of performing robust regression is by calling the `fit` method:

`m = fit(RobustLinearModel, X, y, TukeyEstimator(); initial_scale_estimate=:mad)`

The same for quantile regression:

`m = fit(QuantileRegression, X, y; quantile=0.5)`

## Examples

```julia
using RDatasets: dataset
using StatsModels
using RobustModels
using RobustModels: HuberEstimator, TukeyEstimator, L2Estimator, TauEstimator

data = dataset("robustbase", "Animals2")
data.logBrain = log.(data.Brain)
data.logBody = log.(data.Body)
form = @formula(logBrain ~ 1 + logBody)

## M-estimator using Tukey estimator
m1 = fit(RobustLinearModel, form, data, TukeyEstimator(); method=:cg, initial_scale_estimate=:mad)

## MM-estimator using Tukey estimator
m2 = fit(RobustLinearModel, form, data, TukeyEstimator(); method=:cg, initial_scale_estimate=:mad, kind=:MMestimate)

## M-estimator using Huber estimator and correcting for covariate outliers using leverage
m3 = fit(RobustLinearModel, form, data, HuberEstimator(); method=:cg, initial_scale_estimate=:mad, correct_leverage=true)

## M-estimator using Huber estimator, providing an initial scale estimate and using Cholesky method of solving.
m4 = fit(RobustLinearModel, form, data, HuberEstimator(); method=:chol, initial_scale_estimate=15.0)

## S-estimator using Tukey estimator
m5 = fit(RobustLinearModel, form, data, TukeyEstimator(); initial_scale_estimate=:mad, kind=:Sestimate)

## τ-estimator using Tukey estimator
m6 = fit(RobustLinearModel, form, data, TauEstimator(TukeyEstimator); initial_scale_estimate=:mad, kind=:Tauestimate)

## Expectile regression for τ=0.8
m7 = fit(RobustLinearModel, form, data, L2Estimator(); quantile=0.8)
#m7 = fit(RobustLinearModel, form, data, ExpectileEstimator(0.8))

## Quantile regression solved by linear programming interior point method
m8 = fit(QuantileRegression, form, data; quantile=0.2)

## Refit with different parameters
refit!(m8; quantile=0.8)

```

## Theory

### M-estimators
With ordinary least square (OLS), the objective function is, from maximum likelihood estimation:

`L = ½ Σᵢ (yᵢ - 𝒙ᵢ 𝜷)² = ½ Σᵢ rᵢ²`

where `yᵢ` are the values of the response variable, `𝒙ᵢ` are the covectors of individual covariates (rows of the model matrix `X`), `𝜷` is the vector of fitted coefficients and `rᵢ` are the individual residuals.

A `RobustLinearModel` solves instead for the following objective function: `L' = Σᵢ ρ(rᵢ)` (more precisely `L' = Σᵢ ρ(rᵢ/σ)` where `σ` is an estimate of the standard deviation of the residual). Several M-estimators are implemented:
- `L2Estimator`: `ρ(r) = ½ r²`, like ordinary OLS.
- `L1Estimator`: `ρ(r) = |r|`, non-differentiable estimator also know as _Least absolute deviations_. Prefer the `QuantileRegression` solver.
- `HuberEstimator`: `ρ(r) = if (r<c); ½(r/c)² else |r|/c - ½ end`, convex estimator that behaves as `L2` cost for small residuals and `L1` for large esiduals and outliers.
- `L1L2Estimator`: `ρ(r) = √(1 + (r/c)²) - 1`, smooth version of `HuberEstimator`.
- `FairEstimator`: `ρ(r) = |r|/c - log(1 + |r|/c)`, smooth version of `HuberEstimator`.
- `LogcoshEstimator`: `ρ(r) = log(cosh(r/c))`, smooth version of `HuberEstimator`.
- `ArctanEstimator`: `ρ(r) = r/c * atan(r/c) - ½ log(1+(r/c)²)`, smooth version of `HuberEstimator`.
- `CauchyEstimator`: `ρ(r) = log(1+(r/c)²)`, non-convex estimator, that also corresponds to a Student's-t distribution (with fixed degree of freedom). It suppresses outliers more strongly but it is not sure to converge.
- `GemanEstimator`: `ρ(r) = ½ (r/c)²/(1 + (r/c)²)`, non-convex and bounded estimator, it suppresses outliers more strongly.
- `WelschEstimator`: `ρ(r) = ½ (1 - exp(-(r/c)²))`, non-convex and bounded estimator, it suppresses outliers more strongly.
- `TukeyEstimator`: `ρ(r) = if r<c; ⅙(1 - (1-(r/c)²)³) else ⅙ end`, non-convex and bounded estimator, it suppresses outliers more strongly and it is the prefered estimator for most cases.
- `YohaiZamarEstimator`: `ρ(r)` is quadratic for `r/c < 2/3` and is bounded to 1; non-convex estimator, it is optimized to have the lowest bias for a given efficiency.

The value of the tuning constants `c` are optimized for each estimator so the M-estimators have a high efficiency of 0.95. However, these estimators have a low breakdown point.

### S-estimators
Instead of minimizing `Σᵢ ρ(rᵢ/σ)`, S-estimation minimizes the estimate of the standard deviation `σ` with the constraint that: `Σᵢ ρ(rᵢ/σ) = 1/2`.
S-estimators are only defined for bounded estimators, like `TukeyEstimator`.
These estimators have low efficiency but a high breakdown point of 1/2, by changing the tuning constant `c`.

### MM-estimators
It is a two-pass estimation, 1) Estimate `σ` using an S-estimator with high breakdown point and 2) estimate `𝜷` using an M-estimator with high efficiency.
It results in an estimator with high efficiency and high breakdown point.

### τ-estimators
Like MM-estimators, τ-estimators combine a high efficiency with a high breakdown point. Similar to S-estimators, it minimize a scale estimate:
`τ² = σ² (2/n Σᵢρ₂(rᵢ/σ))` where `σ` is an M-scale estimate solution of `Σᵢ ρ₁(rᵢ/σ) = 1/2`.
Finding the minimum of a τ-estimator is similar to the procedure for an S-estimator with a special weight function
that combines both functions `ρ₁` and `ρ₂`. They should be of the same kind with different tuning constants.

### MQuantile-estimators
Using an asymetric variant of the `L1Estimator`, quantile regression is performed (although the `QuantileRegression` solver should be prefered because it gives an exact solution). Identically, using an asymetric version of each M-estimator, a generalization of quantiles is obtained. For instance, using an asymetric `L2Estimator` results in _Expectile Regression_.

### Quantile regression
_Quantile regression_ results from minimizing the following objective function:
`L = Σᵢ wᵢ|yᵢ - 𝒙ᵢ 𝜷| = Σᵢ wᵢ(rᵢ) |rᵢ|`,
where `wᵢ = ifelse(rᵢ>0, τ, 1-τ)` and `τ` is the quantile of interest. `τ=½` corresponds to _Least Absolute Deviations_.

This problem can be solved exactly using linear programming techniques like interior point methods using the [JuMP](https://github.com/JuliaOpt/JuMP.jl) package with the [GLPK](https://github.com/JuliaOpt/GLPK.jl) backend.


## Credits

This package derives from the [RobustLeastSquares](https://github.com/FugroRoames/RobustLeastSquares.jl) package for the initial implementation, especially for the Conjugate Gradient solver and the definition of the M-Estimator functions.

Credits to the developpers of the [GLM](https://github.com/JuliaStats/GLM.jl) and [MixedModels](https://github.com/JuliaStats/MixedModels.jl) packages for implementing the Iteratively Reweighted Least Square algorithm.

