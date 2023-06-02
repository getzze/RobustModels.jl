using StatsBase
using StatsModels
using GLM
using SparseArrays
using DataFrames
using Test
using Random: MersenneTwister

using RobustModels


## Delegate methods from TableRegressionModel for GLM models
using StatsModels: @delegate, FormulaTerm, TableRegressionModel
import RobustModels:
    dispersion,
    weights,
    islinear,
    fitted,
    isfitted,
    leverage,
    hasintercept,
    nulldeviance,
    deviance

@delegate TableRegressionModel.model [
    leverage,
    weights,
    dispersion,
    deviance,
    nulldeviance,
    fitted,
    isfitted,
    islinear,
    hasintercept,
]

# To run test with verbose output use:
# Pkg.test(RobustModels; test_args=["verbose"])
VERBOSE = "verbose" in ARGS

L1_warning = "Warning: coefficient variance is not well defined for L1Estimator.\n"

## Losses
convex_losses = (
    "L2", "L1", "Huber", "L1L2", "Fair", "Logcosh", "Arctan", "CatoniWide", "CatoniNarrow"
)
other_losses = ("Cauchy",)
bounded_losses = ("Geman", "Welsch", "Tukey", "YohaiZamar", "HardThreshold", "Hampel")
losses = (convex_losses..., other_losses..., bounded_losses...)

## Solving methods
nopen_methods = (:auto, :chol, :cholesky, :qr, :cg)

## Interface methods
interface_methods = (
    dof,
    dof_residual,
    confint,
    deviance,
    nulldeviance,
    loglikelihood,
    nullloglikelihood,
    dispersion,
    nobs,
    stderror,
    vcov,
    residuals,
    response,
    weights,
    workingweights,
    fitted,
    predict,
    isfitted,
    islinear,
    leverage,
    leverage_weights,
    modelmatrix,
    projectionmatrix,
    wobs,
    scale,
    hasintercept,
    hasformula,
)

# Import data
include("data/Animals2.jl")

# Include tests
include("estimators.jl")
include("interface.jl")
include("linearfit.jl")
include("mquantile.jl")
include("robustridge.jl")
include("qreg.jl")
include("weights.jl")
include("univariate.jl")
include("underdetermined.jl")
