module RobustModels


#import Tables
#import NLopt
#using MixedModels: OptSummary

using Distributions: ccdf, Normal
#using StatsModels: @formula, FormulaTerm, coefnames, modelcols, apply_schema, schema
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: BlasReal, Diagonal, Hermitian, transpose, mul!, inv, diag
using Printf: @printf, @sprintf
using GLM: Link, canonicallink, FPVector, cholpred, linpred, linpred!, installbeta!
using StatsBase: mean, mad, ConvergenceException
using IterativeSolvers: lsqr!, cg!
using Roots: find_zero


import Base: ==, *
import GLM: dispersion, dispersion_parameter, LinPred, DensePred, ModResp, delbeta!
import StatsBase: fit, fit!, deviance, nobs
import StatsModels: RegressionModel, coef, coeftable, CoefTable

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
       dispersion_parameter,
       dispersion,
       nothing  # stopper


export Estimator,
       MEstimator,
       RobustModel,
       RobustGeneralizedLinearModel,
       RobustLinearModel,
       interiormethod,
       nothing  # stopper
       
       

abstract type Estimator end

"""
An m-estimator is a cost/loss function used in modified (weighted) least squares
problems of the form:
    min ∑ᵢ ½ ρ(rᵢ²)
"""
abstract type MEstimator <: Estimator end

abstract type AbstractQuantileEstimator <: Estimator end

abstract type AbstractExpectileEstimator <: Estimator end


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



include("MEstimators.jl")
include("interiorpoint.jl")
include("linpred.jl")
include("pirls.jl")
include("robustlinearmodel.jl")

end # module
