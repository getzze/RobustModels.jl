# Weight functions
# ----------------
# A good overview of these can be found at:
# http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html



## Threshold to avoid numerical overflow of the weight function of L1Estimator and ArctanEstimator
DELTA = 1e-8    # chosen because it minimizes the error between (1-ATWDELTA)/DELTA and 1/3 for the linear approximation
L1WDELTA = 1/(DELTA)
ATWDELTA = atan(DELTA)/DELTA
#DELTA = 1e-8    # chosen because it minimizes the error between (1-ATWDELTA)/DELTA and 1/3 for the linear approximation
#L1WDELTA = 1/(2*sqrt(DELTA))
#ATWDELTA = atan(sqrt(DELTA))*2*L1WDELTA


# Default implementations from definintion of cost-function rho
"The cost (a.k.a. loss) function ρ for the M-estimator"
function estimator_rho end
"The derivative of the cost (a.k.a. loss) function ψ for the M-estimator"
function estimator_psi end
"The derivative of ψ for the M-estimator"
function estimator_psider end
"The derivative of the loss function divided by r for the M-estimator"
function estimator_weight end

"""
The function derived from the estimator for M-estimation of scale.
It is bounded with lim_{t->∞} χ = 1
It can be proportional to ρ or t.ψ(t) depending on the estimator.
"""
function estimator_chi(::M, r) where M<:Estimator
    error("This estimator cannot be used for scale estimation: $(M)")
end


"""
The weight function, w, for the M-estimator, to be used for modifying the normal
equations of a least-square problem
"""
estimator_weight(est::Estimator, r) = estimator_psi(est, r) / r
isconvex(::Estimator) = false

isbounded(::Type{Estimator}) = false
function MScaleEstimator(::Type{M}) where M<:MEstimator
    isbounded(M) || error("This estimator cannot be used for scale estimation: $(M)")
    
    est = M(estimator_low_breakpoint_constant(M))
    return r -> estimator_chi(est, r) - 1/2
end


"""
The tuning constant c is computed so the efficiency for Normally distributed
residuals is 0.95. The efficiency of the mean estimate μ is defined by:
eff_μ = (E[ψ'])²/E[ψ²]

```
using QuadGK: quadgk
using Roots: find_zero

function optimal_tuning_constant(::Type{M}; eff=0.95, c0=1.0) where M<:MEstimator
    psi(x, c)  = RobustModels.estimator_psi(M(c), x)
    psip(x, c) = RobustModels.estimator_psider(M(c), x)

    I1(c) = quadgk(x->(psi(x, c))^2*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    I2(c) = quadgk(x->psip(x, c)*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    fun_eff(c) = (I2(c))^2/I1(c)
    copt = find_zero(c->fun_eff(c) - eff, c0, Order1())
end
```

The efficiency of the variance estimate σ² is defined by:
eff_σ² = E[rψ]/(2*Var[ρ])

"""

"""
The M-estimate of scale is computed by solving:

1/n Σ χ(r/ŝ) = δ

with χ a bounded function with χ(∞) = 1 and δ = E[χ]/χ(∞) with expectation w.r.t. Normal density.
The parameter `c` of χ should be chosen such that δ = 1/2, which 
corresponds to a breakpoint of 50%.
The function χ can be directly the pseudo-negloglikelihood ρ or `t.ψ(t)`.
`estimator_chi` returns this function when it is define, but it is better to call
`MScaleEstimator(::Type{Estimator})` that returns the function r->(χ(r) - 1/2) to be called directly to find ŝ by solving:
Σ MScaleEstimator(ri/ŝ) = 0  with ri = yi - μi


```
using QuadGK: quadgk
using Roots: find_zero

function lowbreakpoint_tuning_constant(::Type{M}; bp=1/2, c0=1.0) where M<:MEstimator
    (0 < bp <= 1/2) || error("breakpoint should be between 0 and 1/2")

    try
        RobustModels.estimator_chi(M(c0))
    catch
        error("optimizing the tuning constant for low breakpoint is only defined for bounded estimators.")
    end

    I(c) = quadgk(x->RobustModels.estimator_chi(M(c), x)*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    copt = find_zero(c->I(c) - bp, c0, Order1())
end
```

"""






"The (convex) L2 M-estimator is that of the standard least squares problem."
struct L2Estimator <: MEstimator; end
L2Estimator(c) = L2Estimator()
estimator_rho(   ::L2Estimator, r) = r^2 / 2
estimator_psi(   ::L2Estimator, r) = r
estimator_psider(::L2Estimator, r) = oftype(r, 1)
estimator_weight(::L2Estimator, r) = oftype(r, 1)
estimator_values(::L2Estimator, r) = (r^2/2, r, oftype(r, 1))
isconvex(::L2Estimator) = true


"""
The standard L1 M-estimator takes the absolute value of the residual, and is
convex but non-smooth. It is not a real L1 M-estimator but a Huber M-estimator
with very small tuning constant.
"""
struct L1Estimator <: MEstimator; end
L1Estimator(c) = L1Estimator()
estimator_rho(   ::L1Estimator, r) = abs(r)
estimator_psi(   ::L1Estimator, r) = sign(r)
estimator_psider(::L1Estimator, r) = if (abs(r)<DELTA); oftype(r, 1) else oftype(r, 0) end
estimator_weight(::L1Estimator, r) = if (abs(r)<DELTA); L1WDELTA else 1/abs(r) end
function estimator_values(est::L1Estimator, r)
    rr = abs(r)
    return (rr, sign(r), (if (rr<DELTA); L1WDELTA else 1/rr end) )
end
isconvex(::L1Estimator) = true


"""
The convex Huber estimator switches from between quadratic and linear cost/loss
function at a certain cutoff.
Tuning constant c is taken to give an efficiency of 95% for Gaussian residuals
ρc(r) = c^2 ρ(r/c) so ρc(ε) ≈ ε^2/2 for ε<<1
"""
struct HuberEstimator <: MEstimator
    c::Float64

    HuberEstimator(c::Real) = new(c)
    HuberEstimator() = new(1.345)
end

estimator_rho(est::HuberEstimator, r)    = if (abs(r)<=est.c); r^2/2        else (est.c*abs(r) - est.c^2/2) end
estimator_psi(est::HuberEstimator, r)    = if (abs(r)<=est.c); r            else est.c*sign(r) end
estimator_psider(est::HuberEstimator, r) = if (abs(r)<=est.c); oftype(r, 1) else oftype(r, 0) end
estimator_weight(est::HuberEstimator, r) = if (abs(r)<=est.c); oftype(r, 1) else est.c/abs(r) end
function estimator_values(est::HuberEstimator, r)
    rr = abs(r)
    if rr <= est.c
        return (rr^2/2 , r , oftype(r, 1) )
    else
        return (est.c*rr - est.c^2/2 , est.c*sign(r) , est.c/rr )
    end
end
isconvex(::HuberEstimator) = true



"""
The convex L1-L2 estimator interpolates smoothly between L2 behaviour for small
residuals and L1 for outliers.
"""
struct L1L2Estimator <: MEstimator
    c::Float64

    L1L2Estimator(c::Real) = new(c)
    L1L2Estimator() = new(1.287)
end
estimator_rho(est::L1L2Estimator, r)    = est.c^2*(sqrt(1 + (r/est.c)^2) - 1)
estimator_psi(est::L1L2Estimator, r)    = r / sqrt(1 + (r/est.c)^2)
estimator_psider(est::L1L2Estimator, r) = 1 / (1 + (r/est.c)^2)^(3/2)
estimator_weight(est::L1L2Estimator, r) = 1 / sqrt(1 + (r/est.c)^2)
function estimator_values(est::L1L2Estimator, r)
    sqr = sqrt(1 + (r/est.c)^2)
    return (est.c^2*(sqr - 1), r/sqr, 1/sqr)
end
isconvex(::L1L2Estimator) = true




"""
The (convex) "fair" estimator switches from between quadratic and linear
cost/loss function at a certain cutoff, and is C3 but non-analytic.
"""
struct FairEstimator <: MEstimator
    c::Float64

    FairEstimator(c::Real) = new(c)
    FairEstimator() = new(1.400)
end
estimator_rho(est::FairEstimator, r)    = est.c*abs(r) - est.c^2*log(1 + abs(r/est.c))
estimator_psi(est::FairEstimator, r)    = r / (1 + abs(r)/est.c)
estimator_psider(est::FairEstimator, r) = 1 / (1 + abs(r)/est.c)^2
estimator_weight(est::FairEstimator, r) = 1 / (1 + abs(r)/est.c)
function estimator_values(est::FairEstimator, r)
    ir = 1/(1 + abs(r/est.c))
    return (est.c*abs(r) + est.c^2*log(ir), r*ir, ir)
end
isconvex(::FairEstimator) = true



"""
The convex Arctan estimator 
r * arctan(r) - 1/2*log(1 + r^2)
"""
struct ArctanEstimator <: MEstimator
    c::Float64

    ArctanEstimator(c::Real) = new(c)
    ArctanEstimator() = new(0.919)
end
estimator_rho(est::ArctanEstimator, r)    = est.c * r * atan(r/est.c) - est.c^2/2*log(1 + (r/est.c)^2)
estimator_psi(est::ArctanEstimator, r)    = est.c * atan(r/est.c)
estimator_psider(est::ArctanEstimator, r) = 1 / (1 + (r/est.c)^2)
estimator_weight(est::ArctanEstimator, r) = if (abs(r/est.c)<DELTA); (1 - (r/est.c)^2/3) else est.c * atan(r/est.c) / r end
function estimator_values(est::ArctanEstimator, r)
    ar = est.c * atan(r/est.c)
    return ( r*ar - est.c^2/2*log(1 + (r/est.c)^2), ar, (if (abs(r/est.c)<DELTA); (1 - (r/est.c)^2/3) else ar/r end) )
end
isconvex(::ArctanEstimator) = true



"""
The non-convex Cauchy estimator switches from between quadratic behaviour to
logarithmic tails. This rejects outliers but may result in multiple minima.
"""
struct CauchyEstimator <: MEstimator
    c::Float64

    CauchyEstimator(c::Real) = new(c)
    CauchyEstimator() = new(2.385)
end
estimator_rho(est::CauchyEstimator, r)    = (est.c)^2/2 * log(1 + (r/est.c)^2)
estimator_psi(est::CauchyEstimator, r)    = r / (1 + (r/est.c)^2)
estimator_psider(est::CauchyEstimator, r) = (1 - (r/est.c)^2) / (1 + (r/est.c)^2)^2
estimator_weight(est::CauchyEstimator, r) = 1 / (1 + (r/est.c)^2)
function estimator_values(est::CauchyEstimator, r)
    ir = 1/(1 + (r/est.c)^2)
    return ( -(est.c)^2/2 * log(ir), r*ir, ir )
end
isconvex(::CauchyEstimator) = false

isbounded(::Type{CauchyEstimator}) = true
estimator_low_breakpoint_constant(::Type{CauchyEstimator}) = 0.61200
estimator_chi(est::CauchyEstimator, r) = r*estimator_psi(est, r)/(est.c)^2



"""
The non-convex Geman-McClure for strong supression of outliers and does not guarantee a unique solution
"""
struct GemanEstimator <: MEstimator
    c::Float64

    GemanEstimator(c::Real) = new(c)
    GemanEstimator() = new(3.787)
end
estimator_rho(est::GemanEstimator, r)    = 1/2 * r^2 / (1 + (r/est.c)^2)
estimator_psi(est::GemanEstimator, r)    = r / (1 + (r/est.c)^2)^2
estimator_psider(est::GemanEstimator, r) = (1 - 3*(r/est.c)^2) / (1 + (r/est.c)^2)^3
estimator_weight(est::GemanEstimator, r) = 1 / (1 + (r/est.c)^2)^2
function estimator_values(est::GemanEstimator, r)
    ir = 1/(1 + (r/est.c)^2)
    return ( 1/2 * r^2 *ir, r*ir^2, ir^2 )
end
isconvex(::GemanEstimator) = false

isbounded(::Type{GemanEstimator}) = true
estimator_low_breakpoint_constant(::Type{GemanEstimator}) = 0.61200
estimator_chi(est::GemanEstimator, r) = estimator_rho(est, r)/((est.c)^2/2)


"""
The non-convex Welsch for strong supression of ourliers and does not guarantee a unique solution
"""
struct WelschEstimator <: MEstimator
    c::Float64

    WelschEstimator(c::Real) = new(c)
    WelschEstimator() = new(2.985)
end
estimator_rho(est::WelschEstimator, r)    = -(est.c)^2/2 * Base.expm1(-(r/est.c)^2)
estimator_psi(est::WelschEstimator, r)    = r * exp(-(r/est.c)^2)
estimator_psider(est::WelschEstimator, r) = (1 - 2*(r/est.c)^2)*exp(-(r/est.c)^2)
estimator_weight(est::WelschEstimator, r) = exp(-(r/est.c)^2)
function estimator_values(est::WelschEstimator, r)
    er = exp(-(r/est.c)^2)
    return ( -(est.c)^2/2 * Base.expm1(-(r/est.c)^2), r*er, er )
end
isconvex(::WelschEstimator) = false

isbounded(::Type{WelschEstimator}) = true
estimator_low_breakpoint_constant(::Type{WelschEstimator}) = 0.8165
estimator_chi(est::WelschEstimator, r) = estimator_rho(est, r)/((est.c)^2/2)



"""
The non-convex Tukey biweight estimator which completely suppresses the outliers,
and does not guaranty a unique solution
"""
struct TukeyEstimator <: MEstimator
    c::Float64

    TukeyEstimator(c::Real) = new(c)
    TukeyEstimator() = new(4.685)
end
estimator_rho(est::TukeyEstimator, r)    = if (abs(r)<=est.c); (est.c)^2/6 * (1 - ( 1 - (r/est.c)^2 )^3) else (est.c)^2/6  end
estimator_psi(est::TukeyEstimator, r)    = if (abs(r)<=est.c); r*(1 - (r/est.c)^2)^2                     else oftype(r, 0) end
estimator_psider(est::TukeyEstimator, r) = if (abs(r)<=est.c); 1 - 6*(r/est.c)^2 + 5*(r/est.c)^4         else oftype(r, 0) end
estimator_weight(est::TukeyEstimator, r) = if (abs(r)<=est.c); (1 - (r/est.c)^2)^2                       else oftype(r, 0) end
function estimator_values(est::TukeyEstimator, r)
    pr = (abs(r)<=est.c) * (1 - (r/est.c)^2)
    return ( (est.c)^2/6*(1 - pr^3), r*pr^2, pr^2 )
end
isconvex(::TukeyEstimator) = false

isbounded(::Type{TukeyEstimator}) = true
estimator_low_breakpoint_constant(::Type{TukeyEstimator}) = 1.5476
estimator_chi(est::TukeyEstimator, r) = estimator_rho(est, r)/((est.c)^2/6)


"""
The non-convex Student's-t estimator. This rejects outliers but may result in multiple minima.

Estimation of c(ν) from fitting:

```
using QuadGK
using Roots
using GLM
function optimal_tuning_constant_Student(; eff=0.95, c0=1.0, ν=1)
    psi(x, c)  = RobustModels.estimator_psi(RobustModels.StudentEstimator(c, ν), x)
    psip(x, c) = RobustModels.estimator_psider(RobustModels.StudentEstimator(c, ν), x)

    I1(c) = quadgk(x->(psi(x, c))^2*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    I2(c) = quadgk(x->psip(x, c)*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    fun_eff(c) = (I2(c))^2/I1(c)
    copt = find_zero(c->fun_eff(c) - eff, c0, Order1())
end

res = [optimal_tuning_constant_Student(eff=0.95, c0=0.1, ν=ν) for ν in 1:200]
m = lm(reshape(hcat(ones(200), log.(1:200)), (200, 2)), log.(res))
println(coef(m))
```

Estimation of c(ν) for low breakpoint from fitting:

```
using QuadGK
using Roots
using GLM
function lowbreakpoint_tuning_constant_Student(; bp=1/2, c0=1, ν=1)
    (0 < bp <= 1/2) || error("breakpoint should be between 0 and 1/2")

    I(c) = quadgk(x->RobustModels.estimator_chi(RobustModels.StudentEstimator(c, ν), x)*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    copt = find_zero(c->I(c) - bp, c0, Order1())
end

res = [lowbreakpoint_tuning_constant_Student(bp=1/2, c0=0.1, ν=ν) for ν in 1:200]
m = lm(reshape(hcat(ones(200), log.(1:200)), (200, 2)), log.(res))
println(coef(m))
```
"""
struct StudentEstimator <: MEstimator
    c::Float64
    ν::Float64
    νp1::Float64

    StudentEstimator(c::Real, ν::Real) = new(c, ν, (ν+1)/(2*ν))
    StudentEstimator(ν::Real) = new(2.385/√ν, ν, (ν+1)/(2*ν))
    StudentEstimator() = new(2.385, 1, 1)
end
estimator_rho(est::StudentEstimator, r)    = (est.c)^2 * (est.ν + 1)/4 * log(1 + (r/est.c)^2 / est.ν)
estimator_psi(est::StudentEstimator, r)    = est.νp1 * r / (1 + (r/est.c)^2 / est.ν)
estimator_psider(est::StudentEstimator, r) = est.νp1 * (1 - (r/est.c)^2 / est.ν) / (1 + (r/est.c)^2 / est.ν)^2
estimator_weight(est::StudentEstimator, r) = est.νp1 / (1 + (r/est.c)^2 / est.ν)
function estimator_values(est::StudentEstimator, r)
    ir = est.νp1 /(1 + (r/est.c)^2 / est.ν)
    return ( -(est.c)^2 * (est.ν + 1)/4 * log(ir / est.νp1), r*ir, ir )
end
isconvex(::StudentEstimator) = false
estimator_limit_rho(::StudentEstimator) = Inf
estimator_limit_tpsi(est::StudentEstimator) = (est.c)^2*(est.ν+1)/2

isbounded(::Type{StudentEstimator}) = true
estimator_low_breakpoint_constant(::Type{StudentEstimator}, ν) = 0.8165/√ν
estimator_chi(est::StudentEstimator, r) = r*estimator_psi(est, r)/((est.c)^2*(est.ν+1)/2)
function MScaleEstimator(T::Type{StudentEstimator}, ν)
    est = T(estimator_low_breakpoint_constant(T, ν), ν)
    return r-> estimator_chi(est, r) - 1/2
end
estimator_low_breakpoint_constant(::Type{StudentEstimator}) = error("this function should be called with ν as an extra last argument for StudentEstimator")
MScaleEstimator(::Type{StudentEstimator}) = error("this function should be called with ν as an extra last argument for StudentEstimator")


"""
The quantile estimator is a generalization of the L1 estimator,
that correspond to a median estimator, for any quantile value τ.
For the 0.5-quantile estimator, the loss if proportional to the L1 norm
such that loss(QuantileEstimator(0.5)) = 1/2 * loss(L1Estimator)

[1] Schnabel, Eilers - Computational Statistics and Data Analysis 53 (2009) 4168–4177 - Optimal expectile smoothing
doi:10.1016/j.csda.2009.05.002
"""
struct QuantileEstimator <: MEstimator
    τ::Float64

    QuantileEstimator(τ::Real) = new(τ)
    QuantileEstimator() = new(0.5)
end

_weight(est::QuantileEstimator, r) = oftype(r, ifelse(r>0, est.τ, 1 - est.τ))
estimator_rho(est::QuantileEstimator, r) = _weight(est, r) * abs(r)
estimator_psi(est::QuantileEstimator, r) = _weight(est, r) * sign(r)
estimator_psider(est::QuantileEstimator, r) = if (abs(r)<DELTA); oftype(r, 1) else oftype(r, 0) end
estimator_weight(est::QuantileEstimator, r) = if (abs(r)<DELTA); r*(est.τ - 1/2)*L1WDELTA^2 + L1WDELTA/2 else _weight(est, r)/abs(r) end
function estimator_values(est::QuantileEstimator, r)
    w = _weight(est, r)
    ww = if (abs(r)<DELTA); r*(est.τ - 1/2)*L1WDELTA^2 + L1WDELTA/2 else w/abs(r) end
    return (w*abs(r), w*sign(r), ww)
end
isconvex(::QuantileEstimator) = true


"""
The expectile estimator is a generalization of the L2 estimator,
that correspond to a mean estimator, for any value τ ∈ [0,1].

[1] Schnabel, Eilers - Computational Statistics and Data Analysis 53 (2009) 4168–4177 - Optimal expectile smoothing
doi:10.1016/j.csda.2009.05.002
"""
struct ExpectileEstimator <: MEstimator
    τ::Float64

    ExpectileEstimator(τ::Real) = new(τ)
    ExpectileEstimator() = new(0.5)
end

_weight(est::ExpectileEstimator, r) = oftype(r, ifelse(r>0, est.τ, 1 - est.τ))
estimator_rho(est::ExpectileEstimator, r) = _weight(est, r) * r^2
estimator_psi(est::ExpectileEstimator, r) = 2 * _weight(est, r) * r
estimator_psider(est::ExpectileEstimator, r) = 2 * _weight(est, r)
estimator_weight(est::ExpectileEstimator, r) = 2 * _weight(est, r)
function estimator_values(est::ExpectileEstimator, r)
    w = _weight(est, r)
    return (w*r^2, 2*w*r, 2*w)
end
isconvex(::ExpectileEstimator) = true

