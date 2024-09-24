# API

## Types defined in the package

```@docs
AbstractRobustModel
AbstractEstimator
AbstractQuantileEstimator
LossFunction
RobustLinearModel
RobustModels.RobustLinResp
GLM.LinPred
RobustModels.DensePredCG
RobustModels.SparsePredCG
GLM.DensePredChol
GLM.SparsePredChol
GLM.DensePredQR
RobustModels.RidgePred
RobustModels.AbstractRegularizedPred
QuantileRegression
```

## Constructors for models

```@docs
fit(::Type{M}, ::Union{AbstractMatrix{T}}, ::AbstractVector{T}, ::AbstractMEstimator) where {T<:AbstractFloat, M<:RobustLinearModel}
fit(::Type{M}, ::Union{AbstractMatrix{T}}, ::AbstractVector{T}) where {T<:AbstractFloat, M<:QuantileRegression}
```

```@docs
rlm
quantreg
fit!
refit!
```

## Model methods
```@docs
StatsModels.coef
StatsAPI.coeftable
StatsAPI.coefnames
StatsModels.responsename
StatsAPI.confint
StatsBase.deviance
StatsBase.nulldeviance
StatsAPI.dof
StatsAPI.dof_residual
StatsBase.nobs
wobs
StatsAPI.isfitted
StatsAPI.islinear
StatsAPI.loglikelihood
StatsAPI.nullloglikelihood
StatsAPI.stderror
StatsBase.vcov
StatsBase.weights
workingweights
StatsAPI.fitted
StatsBase.predict
StatsModels.leverage
leverage_weights
StatsAPI.modelmatrix
projectionmatrix
dispersion(::RobustLinearModel, ::Bool)
StatsAPI.response
StatsAPI.residuals
StatsModels.hasintercept
hasformula
formula
scale
tauscale
RobustModels.location_variance
Estimator
GLM.linpred!
RobustModels.pirls!
RobustModels.pirls_Sestimate!
RobustModels.pirls_τestimate!
```

## Estimators
```@docs
MEstimator
RobustModels.L1Estimator
L2Estimator
SEstimator
MMEstimator
TauEstimator
GeneralizedQuantileEstimator
ExpectileEstimator
RobustModels.QuantileEstimator
```

## Loss functions
```@docs
BoundedLossFunction
L2Loss
L1Loss
HuberLoss
L1L2Loss
FairLoss
LogcoshLoss
ArctanLoss
CatoniWideLoss
CatoniNarrowLoss
CauchyLoss
GemanLoss
WelschLoss
TukeyLoss
YohaiZamarLoss
HardThresholdLoss
HampelLoss
```

## Estimator and Loss functions methods
```@docs
RobustModels.rho
RobustModels.psi
RobustModels.psider
RobustModels.weight
RobustModels.estimator_values
RobustModels.estimator_norm
RobustModels.estimator_bound
tuning_constant
RobustModels.isconvex
RobustModels.isbounded
RobustModels.estimator_high_breakdown_point_constant
RobustModels.estimator_high_efficiency_constant
RobustModels.efficient_loss
RobustModels.robust_loss
RobustModels.efficiency_tuning_constant
RobustModels.mscale_loss
RobustModels.breakdown_point_tuning_constant
RobustModels.scale_estimate
RobustModels.tau_efficiency_tuning_constant
RobustModels.estimator_tau_efficient_constant
loss
RobustModels.set_SEstimator
RobustModels.set_MEstimator
RobustModels.update_weight!
RobustModels.tau_scale_estimate
RobustModels.quantile_weight
```
