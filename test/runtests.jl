using StatsBase
using StatsModels
using GLM
using SparseArrays
using DataFrames
#using RDatasets: dataset
using Test

using RobustModels

# Import data
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


# Include tests
include("estimators.jl")
include("interface.jl")
include("weights.jl")
include("linearfit.jl")
include("mquantile.jl")
include("robustridge.jl")
include("qreg.jl")
include("univariate.jl")
