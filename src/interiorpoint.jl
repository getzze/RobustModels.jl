

using LinearAlgebra: dot
using JuMP: Model, @variable, @constraint, @objective, optimize!, value
import GLPK



function interiormethod(X, y, τ; wts=[],verbose=false)
    model = Model(GLPK.Optimizer)
   
    n, p = size(X)

    @variable(model, β[1:p])
    @variable(model, u[1:n] >= 0)
    @variable(model, v[1:n] >= 0)

    e = ones(n)

    @objective(model, Min, τ*dot(e, u) + (1-τ)*dot(e, v) )
    
    Wy, WX = if isempty(wts)
        y, X
    else
        (wts .* y), (wts .* X) 
    end
    
    @constraint(model, resid, Wy .== WX * β + u - v)
    
    optimize!(model)
    
    βval = value.(β)
    rval = value.(u) - value.(v)
    if verbose
        println("coef: ", βval)
        println("res: ", rval)
    end
    βval, rval
end
