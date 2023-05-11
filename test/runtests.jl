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
import RobustModels: dispersion, weights, islinear, fitted, isfitted, leverage,
    hasintercept, nulldeviance, deviance

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
convex_losses = ("L2", "L1", "Huber", "L1L2", "Fair", "Logcosh", "Arctan", "CatoniWide", "CatoniNarrow")
other_losses = ("Cauchy",)
bounded_losses = ("Geman", "Welsch", "Tukey", "YohaiZamar", "HardThreshold", "Hampel")
losses = (convex_losses..., other_losses..., bounded_losses...)

## Penalties
penalties = ("SquaredL2", "Euclidean", "L1", "ElasticNet")

## Solving methods
pen_methods = (:auto, :cgd, :fista, :ama, :admm)
nopen_methods = (:auto, :chol, :qr, :cg)

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
#using RDatasets: dataset
#data = dataset("robustbase", "Animals2")

Animal = ["Lesser short-tailed shrew", "Little brown bat", "Big brown bat", "Mouse", "Musk shrew", "Star-nosed mole", "E. American mole", "Ground squirrel", "Tree shrew", "Golden hamster", "Mole", "Galago", "Rat", "Chinchilla", "Owl monkey", "Desert hedgehog", "Rock hyrax-a", "European hedgehog", "Tenrec", "Artic ground squirrel", "African giant pouched rat", "Guinea pig", "Mountain beaver", "Slow loris", "Genet", "Phalanger", "N.A. opossum", "Tree hyrax", "Rabbit", "Echidna", "Cat", "Artic fox", "Water opossum", "Nine-banded armadillo", "Rock hyrax-b", "Yellow-bellied marmot", "Verbet", "Red fox", "Raccoon", "Rhesus monkey", "Potar monkey", "Baboon", "Roe deer", "Goat", "Kangaroo", "Grey wolf", "Chimpanzee", "Sheep", "Giant armadillo", "Human", "Grey seal", "Jaguar", "Brazilian tapir", "Donkey", "Pig", "Gorilla", "Okapi", "Cow", "Horse", "Giraffe", "Asian elephant", "African elephant", "Triceratops", "Dipliodocus", "Brachiosaurus"]
Brain = [0.14, 0.25, 0.3, 0.4, 0.33, 1.0, 1.2, 4.0, 2.5, 1.0, 3.0, 5.0, 1.9, 6.4, 15.5, 2.4, 12.3, 3.5, 2.6, 5.7, 6.6, 5.5, 8.1, 12.5, 17.5, 11.4, 6.3, 12.3, 12.1, 25.0, 25.6, 44.5, 3.9, 10.8, 21.0, 17.0, 58.0, 50.4, 39.2, 179.0, 115.0, 179.5, 98.2, 115.0, 56.0, 119.5, 440.0, 175.0, 81.0, 1320.0, 325.0, 157.0, 169.0, 419.0, 180.0, 406.0, 490.0, 423.0, 655.0, 680.0, 4603.0, 5712.0, 70.0, 50.0, 154.5]
Body = [0.005, 0.01, 0.023, 0.023, 0.048, 0.06, 0.075, 0.101, 0.104, 0.12, 0.122, 0.2, 0.28, 0.425, 0.48, 0.55, 0.75, 0.785, 0.9, 0.92, 1.0, 1.04, 1.35, 1.4, 1.41, 1.62, 1.7, 2.0, 2.5, 3.0, 3.3, 3.385, 3.5, 3.5, 3.6, 4.05, 4.19, 4.235, 4.288, 6.8, 10.0, 10.55, 14.83, 27.66, 35.0, 36.33, 52.16, 55.5, 60.0, 62.0, 85.0, 100.0, 160.0, 187.1, 192.0, 207.0, 250.0, 465.0, 521.0, 529.0, 2547.0, 6654.0, 9400.0, 11700.0, 87000.0]
data = DataFrame(; Animal=Animal, Brain=Brain, Body=Body)

data.logBrain = log.(data.Brain)
data.logBody = log.(data.Body)
form = @formula(logBrain ~ 1 + logBody)
X = hcat(ones(size(data, 1)), data.logBody)
sX = SparseMatrixCSC(X)
y = data.logBrain
nt = (; logBrain=data.logBrain, logBody=data.logBody)

data_tuples = ((form, data), (form, nt), (X, y), (sX, y))

# Include tests
include("estimators.jl")
include("interface.jl")
include("linearfit.jl")
include("mquantile.jl")
include("robustridge.jl")
include("qreg.jl")
include("weights.jl")
include("univariate.jl")
