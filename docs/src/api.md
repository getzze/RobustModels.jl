# API

## Types defined in the package

```@docs
AbstractRobustModel
AbstractEstimator
AbstractQuantileEstimator
LossFunction
RobustLinearModel
RobustModels.RobustLinResp
RobustModels.IPODResp
GLM.LinPred
RobustModels.DensePredCG
RobustModels.SparsePredCG
GLM.DensePredChol
GLM.SparsePredChol
GLM.DensePredQR
RobustModels.RidgePred
RobustModels.AbstractRegularizedPred
RobustModels.CGDPred
RobustModels.FISTAPred
RobustModels.AMAPred
RobustModels.ADMMPred
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
ipod
fit!
refit!
```

## Model methods
```@docs
StatsModels.coef
StatsModels.coeftable
StatsModels.coefnames
StatsModels.responsename
StatsBase.confint
StatsBase.deviance
StatsBase.nulldeviance
StatsBase.dof
StatsBase.dof_residual
StatsBase.nobs
wobs
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
StatsModels.leverage
leverage_weights
StatsModels.modelmatrix
projectionmatrix
GLM.dispersion(::AbstractRobustModel, ::Bool)
StatsBase.response
StatsBase.residuals
StatsModels.hasintercept
hasformula
formula
haspenalty
penalty
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

## Penalty functions
```@docs
NoPenalty
SquaredL2Penalty
EuclideanPenalty
L1Penalty
ElasticNetPenalty
RangedPenalties
```

## Estimator, Loss and Penalty functions methods
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
RobustModels.cost
RobustModels.proximal!
RobustModels.proximal
RobustModels.isconcrete
RobustModels.concrete!
RobustModels.concrete
```


