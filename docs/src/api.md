# API

## Types defined in the package

```@docs
AbstractRobustModel
AbstractEstimator
AbstractQuantileEstimator
LossFunction
RobustLinearModel
RobustModels.DensePredCG
RobustModels.SparsePredCG
GLM.DensePredChol
GLM.SparsePredChol
RobustModels.RidgePred
RobustModels.RobustLinResp
QuantileRegression
```

## Constructors for models

```@docs
fit(::Type{M}, ::Union{AbstractMatrix{T},SparseMatrixCSC{T}}, ::AbstractVector{T}, ::AbstractEstimator) where {T<:AbstractFloat, M<:RobustLinearModel}
fit(::Type{M}, ::Union{AbstractMatrix{T},SparseMatrixCSC{T}}, ::AbstractVector{T}) where {T<:AbstractFloat, M<:QuantileRegression}
```

```@docs
rlm
quantreg
fit!
refit!
```

## Model methods
```@docs
StatsBase.coef
StatsBase.coeftable
StatsBase.confint
StatsBase.deviance
StatsBase.nulldeviance
StatsBase.dof
StatsBase.dof_residual
nobs(::StatisticalModel)
StatsBase.isfitted
StatsBase.islinear
StatsBase.loglikelihood
StatsBase.nullloglikelihood
StatsBase.stderror
StatsBase.vcov
StatsBase.weights
workingweights
StatsBase.fitted
StatsBase.predict
StatsBase.leverage
StatsBase.modelmatrix
projectionmatrix
GLM.dispersion(::RobustLinearModel, ::Bool)
StatsBase.response
StatsBase.residuals
scale
tauscale
RobustModels.location_variance
Estimator
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
L2Loss
L1Loss
HuberLoss
L1L2Loss
FairLoss
LogcoshLoss
ArctanLoss
CauchyLoss
GemanLoss
WelschLoss
TukeyLoss
YohaiZamarLoss
```

## Estimator and Loss functions methods
```@docs
RobustModels.rho
RobustModels.psi
RobustModels.psider
RobustModels.weight
RobustModels.values
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


