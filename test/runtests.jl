using StatsBase
using StatsModels
using GLM
using SparseArrays
using RDatasets: dataset
using Test

using RobustModels

# Import data
data = dataset("robustbase", "Animals2")
data.logBrain = log.(data.Brain)
data.logBody = log.(data.Body)
form = @formula(logBrain ~ 1 + logBody)
X = hcat(ones(size(data, 1)), data.logBody)
sX = SparseMatrixCSC(X)
y = data.logBrain


# Include tests
include("estimators.jl")
include("interface.jl")
include("linearfit.jl")
include("mquantile.jl")
include("robustridge.jl")
include("qreg.jl")
include("univariate.jl")
