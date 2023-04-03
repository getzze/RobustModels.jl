module RobustModels

include("compat.jl")

# Use README as the docstring of the module and doctest README
@doc let path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    replace(read(path, String), "```julia" => "```jldoctest; output = false")
end RobustModels


using Distributions: ccdf, pdf, quantile, Normal, Chisq, TDist, FDist
using SparseArrays: SparseMatrixCSC, spdiagm
using LinearAlgebra: diag, dot, tr, I, UniformScaling, rmul!, lmul!
using Random: AbstractRNG, GLOBAL_RNG
using Printf: @printf, @sprintf
using GLM: Link, canonicallink, FPVector, lm, SparsePredChol, DensePredChol
using StatsBase: mean, mad, ConvergenceException, sample, quantile
using IterativeSolvers: lsqr!, cg!
#using Roots: find_zero, Order1, ConvergenceFailed
#using QuadGK: quadgk
#import Tulip


import Base: ==, show, broadcastable
import GLM:
    dispersion, LinPred, DensePred, ModResp, delbeta!, linpred!, installbeta!, cholpred
import StatsBase: fit, fit!, deviance, nulldeviance, nobs, weights, Weights, confint,
                  dof, dof_residual, loglikelihood, nullloglikelihood, stderror,
                  vcov, residuals, predict, response, islinear, fitted, isfitted,
                  mean, var, std, sem, mean_and_std, mean_and_var
import StatsModels:
    @delegate, @formula, RegressionModel, coef, coeftable, CoefTable,
    leverage, modelmatrix, TableRegressionModel, hasintercept

## Reexports
export coef,
       coeftable,
       confint,
       deviance,
       nulldeviance,
       dof,
       dof_residual,
       loglikelihood,
       nullloglikelihood,
       nobs,
       stderror,
       vcov,
       residuals,
       predict,
       fit,
       fit!,
       response,
       modelmatrix,
       dispersion,
       islinear,
       isfitted,
       hasintercept,
       fitted,
       weights,
       leverage,
       quantile,
       @formula,
       mean,
       var,
       std,
       sem,
       mean_and_std,
       mean_and_var,
       nothing  # stopper


export LossFunction,
       BoundedLossFunction,
       ConvexLossFunction,
       CompositeLossFunction,
       L2Loss,
       L1Loss,
       HuberLoss,
       L1L2Loss,
       FairLoss,
       LogcoshLoss,
       ArctanLoss,
       CauchyLoss,
       GemanLoss,
       WelschLoss,
       TukeyLoss,
       YohaiZamarLoss,
       AbstractEstimator,
       AbstractMEstimator,
       AbstractQuantileEstimator,
       MEstimator,
       SEstimator,
       MMEstimator,
       TauEstimator,
       GeneralizedQuantileEstimator,
       ExpectileEstimator,
       L2Estimator,
       DensePredCG,
       SparsePredCG,
       RidgePred,
       RobustResp,
       AbstractRobustModel,
       RobustLinearModel,
       QuantileRegression,
       Estimator,
       rlm,
       quantreg,
       loss,
       tuning_constant,
       refit!,
       projectionmatrix,
       workingweights,
       scale,
       tauscale,
       mean_and_sem,
       nothing  # stopper


"""
An estimator needs a cost/loss function for the modified (weighted) least squares
problems of the form:

```math
\\min \\sum_i \\rho\\left(\\dfrac{r_i}{\\hat{\\sigma}}\right)
```
"""
abstract type LossFunction end

"Bounded loss function type for hard rejection of outlier."
abstract type BoundedLossFunction <: LossFunction end

"Convex loss function type with no local minimum solution."
abstract type ConvexLossFunction <: LossFunction end


"General location or scale estimator"
abstract type AbstractEstimator end

"Generalized M-estimator: location or scale estimator associated to one (or more) loss function and
solved using Iteratively Reweighted Least Square."
abstract type AbstractMEstimator <: AbstractEstimator end

"Generalized M-Quantile estimator"
abstract type AbstractQuantileEstimator <: AbstractMEstimator end


"""
    AbstractRobustModel

Abstract type for robust models.

`RobustModels.jl` implements one subtype: [`RobustLinearModel`](@ref).
See the documentation for each for more details.
"""
abstract type AbstractRobustModel{T} <: RegressionModel end

abstract type RobustResp{T} <: ModResp end

abstract type AbstractRegularizedPred{T} <: LinPred end

Base.broadcastable(m::T) where {T<:AbstractEstimator} = Ref(m)
Base.broadcastable(m::T) where {T<:LossFunction} = Ref(m)

include("estimators.jl")
include("robustlinearmodel.jl")
include("linpred.jl")
include("linresp.jl")
include("univariate.jl")
include("quantileregression.jl")

end # module
