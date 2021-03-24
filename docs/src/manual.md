# Manual

## Installation

```julia
Pkg.add("RobustModels")
```

## Fitting robust models

The API is consistent with [GLM package](https://github.com/JuliaStats/GLM.jl).
To fit a robust model, use the function, `rlm(formula, data, estimator; initial_scale=:mad)`, where,
- `formula`: uses column symbols from the DataFrame data, for example, if `propertynames(data)=[:Y,:X1,:X2]`,then a valid formula is `@formula(Y ~ X1 + X2)`. An intercept is included by default.
- `data`: a DataFrame which may contain missing values
- `estimator`: chosen from

    - [`MEstimator`](@ref)
    - [`SEstimator`](@ref)
    - [`MMEstimator`](@ref)
    - [`TauEstimator`](@ref)
    - [`GeneralizedQuantileEstimator`](@ref)

Supported loss functions are:

- [`L2Loss`](@ref): `ρ(r) = ½ r²`, like ordinary OLS.
- [`L1Loss`](@ref): `ρ(r) = |r|`, non-differentiable estimator also know as _Least absolute deviations_. Prefer the `QuantileRegression` solver.
- [`HuberLoss`](@ref): `ρ(r) = if (r<c); ½(r/c)² else |r|/c - ½ end`, convex estimator that behaves as `L2` cost for small residuals and `L1` for large esiduals and outliers.
- [`L1L2Loss`](@ref): `ρ(r) = √(1 + (r/c)²) - 1`, smooth version of `HuberLoss`.
- [`FairLoss`](@ref): `ρ(r) = |r|/c - log(1 + |r|/c)`, smooth version of `HuberLoss`.
- [`LogcoshLoss`](@ref): `ρ(r) = log(cosh(r/c))`, smooth version of `HuberLoss`.
- [`ArctanLoss`](@ref): `ρ(r) = r/c * atan(r/c) - ½ log(1+(r/c)²)`, smooth version of `HuberLoss`.
- [`CauchyLoss`](@ref): `ρ(r) = log(1+(r/c)²)`, non-convex estimator, that also corresponds to a Student's-t distribution (with fixed degree of freedom). It suppresses outliers more strongly but it is not sure to converge.
- [`GemanLoss`](@ref): `ρ(r) = ½ (r/c)²/(1 + (r/c)²)`, non-convex and bounded estimator, it suppresses outliers more strongly.
- [`WelschLoss`](@ref): `ρ(r) = ½ (1 - exp(-(r/c)²))`, non-convex and bounded estimator, it suppresses outliers more strongly.
- [`TukeyLoss`](@ref): `ρ(r) = if r<c; ⅙(1 - (1-(r/c)²)³) else ⅙ end`, non-convex and bounded estimator, it suppresses outliers more strongly and it is the prefered estimator for most cases.
- [`YohaiZamarLoss`](@ref): `ρ(r)` is quadratic for `r/c < 2/3` and is bounded to 1; non-convex estimator, it is optimized to have the lowest bias for a given efficiency.

An estimator is constructed from an estimator type and a loss, e.g. `MEstimator{TukeyLoss}()`.

For [`GeneralizedQuantileEstimator`](@ref), the quantile should be specified with `τ` (0.5 by default), e.g. `GeneralizedQuantileEstimator{HuberLoss}(0.2)`.



## Methods applied to fitted models

Many of the methods are consistent with [GLM](https://github.com/JuliaStats/GLM.jl).
- [`nobs`](@ref StatsBase.nobs): number of observations
- [`dof_residual`](@ref): degrees of freedom for residuals
- [`dof`](@ref StatsBase.dof): degrees of freedom of the model, defined by `nobs(m) - dof_residual(m)`
- [`coef`](@ref StatsBase.coef): estimate of the coefficients in the model
- [`predict`](@ref StatsBase.predict) : obtain predicted values of the dependent variable from the fitted model
- [`deviance`](@ref StatsBase.deviance)/[`nulldeviance`](@ref StatsBase.nulldeviance): measure of the model (null model, respectively) fit
- [`stderror`](@ref StatsBase.stderror): standard errors of the coefficients
- [`confint`](@ref StatsBase.confint): confidence intervals for the fitted coefficients
- [`scale`](@ref): the scale estimate from the model
- [`workingweights`](@ref): the weights for each observation from the robust estimate. Outliers have low weights
- [`leverage`](@ref StatsBase.leverage): the vector of leverage score for each observation
- [`vcov`](@ref StatsBase.vcov): estimated variance-covariance matrix of the coefficient estimates


## Separation of response object and predictor object

Building upon [GLM](https://github.com/JuliaStats/GLM.jl) separation of the response and predictor objects,
this package implements a new `RobustLinResp` object to compute the residuals.
There are currently two available predictor objects: `DensePredChol`/`SparsePredChol`
(imported from [GLM](https://github.com/JuliaStats/GLM.jl)) and `DensePredCG`/`SparsePredCG`
that use the iterative Conjugate Gradient methods, `cg!` and `lsqr!`
from the [IterativeSolvers package](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl)
that is faster and more accurate than Cholesky method for very large matrices.
The predictor that is used depends on the model matrix type and the `method` argument of the `fit`/`fit!`/`rlm` methods.


