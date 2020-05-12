module RobustModels

import Base: ==, show

# https://github.com/JuliaLang/julia/pull/29679
@static if VERSION < v"1.1.0-DEV.472"
    isnothing(::Any) = false
    isnothing(::Nothing) = true
end

# https://github.com/JuliaLang/julia/pull/32148
@static if VERSION < v"1.3.0"
    show(io::IO, ::Nothing) = print(io, "nothing")
end

using Distributions: ccdf, pdf, quantile, Normal, Chisq
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: diag, dot
using Printf: @printf, @sprintf
using GLM: Link, canonicallink, FPVector, cholpred
using StatsBase: mean, mad, ConvergenceException
using IterativeSolvers: lsqr!, cg!
#using Roots: find_zero, Order1, ConvergenceFailed
#using QuadGK: quadgk
#using JuMP: Model, @variable, @constraint, @objective, optimize!, value
#import GLPK


import GLM: dispersion, LinPred, DensePred, ModResp, delbeta!, linpred!, installbeta!
import StatsBase: fit, fit!, deviance, nobs, weights
import StatsModels: RegressionModel, coef, coeftable, CoefTable, leverage, TableRegressionModel

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
       model_response,
       response,
       modelmatrix,
       r2,
       r²,
       adjr2,
       adjr²,
       dispersion,
       weights,
       leverage,
       nothing  # stopper


export Estimator,
       SimpleEstimator,
       ConvexEstimator,
       BoundedEstimator,
       AbstractQuantileEstimator,
       GeneralQuantileEstimator,
       RobustResp,
       AbstractRobustModel,
       RobustLinearModel,
       QuantileRegression,
       SEstimator,
       rlm,
       quantreg,
       nothing  # stopper



"""
An M-estimator is a cost/loss function used in modified (weighted) least squares
problems of the form:
    min ∑ᵢ ½ ρ(rᵢ²)
"""
abstract type Estimator end

abstract type SimpleEstimator <: Estimator end
abstract type BoundedEstimator <: SimpleEstimator end
abstract type ConvexEstimator <: SimpleEstimator end

abstract type AbstractQuantileEstimator <: Estimator end


"""
    AbstractRobustModel
Abstract type for robust models.  RobustModels.jl implements two subtypes:
`RobustLinearModel` and `RobustGeneralizedLinearModel`.  See the documentation for
each for more details.
This type is primarily used for dispatch in `fit`.  Without a distribution and
link function specified, a `RobustLinearModel` will be fit.  When a
distribution/link function is provided, a `RobustGeneralizedLinearModel` is fit.
"""
abstract type AbstractRobustModel{T} <: RegressionModel end


abstract type RobustResp{T} <: ModResp end



include("estimators.jl")
include("robustlinearmodel.jl")
include("linpred.jl")
include("linresp.jl")
include("quantileregression.jl")

end # module
