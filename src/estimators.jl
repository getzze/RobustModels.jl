

## Threshold to avoid numerical overflow of the weight function of L1Estimator and ArctanEstimator
DELTA = 1e-8    # chosen because it minimizes the error between (1-ATWDELTA)/DELTA and 1/3 for the linear approximation
L1WDELTA = 1/(DELTA)
ATWDELTA = atan(DELTA)/DELTA
#DELTA = 1e-8    # chosen because it minimizes the error between (1-ATWDELTA)/DELTA and 1/3 for the linear approximation
#L1WDELTA = 1/(2*sqrt(DELTA))
#ATWDELTA = atan(sqrt(DELTA))*2*L1WDELTA


"The cost (a.k.a. loss) function ρ for the M-estimator"
function estimator_rho end
"The derivative of the cost (a.k.a. loss) function ψ for the M-estimator"
function estimator_psi end
"The derivative of ψ for the M-estimator"
function estimator_psider end
"The derivative of the loss function divided by r for the M-estimator"
function estimator_weight end


isconvex( e::SimpleEstimator) = isa(e, ConvexEstimator)
isbounded(e::SimpleEstimator) = isa(e, BoundedEstimator)

estimator_low_breakpoint_constant( ::SimpleEstimator) = 1
estimator_high_efficiency_constant(::SimpleEstimator) = 1


"""
The function derived from the estimator for M-estimation of scale.
It is bounded with lim_{t->∞} χ = 1
It can be proportional to ρ or t.ψ(t) depending on the estimator.
"""
function estimator_chi(::M, r) where M<:SimpleEstimator
    error("This estimator cannot be used for scale estimation: $(M)")
end

function MScaleEstimator(::Type{M}) where M<:SimpleEstimator
    error("This estimator cannot be used for scale estimation: $(M)")
end

function MScaleEstimator(::Type{M}) where M<:BoundedEstimator
    est = M(estimator_low_breakpoint_constant(M))
    return r -> estimator_chi(est, r) - 1/2
end



"""
The tuning constant c is computed so the efficiency for Normally distributed
residuals is 0.95. The efficiency of the mean estimate μ is defined by:
eff_μ = (E[ψ'])²/E[ψ²]

```
using QuadGK: quadgk
using Roots: find_zero, Order1

function efficiency_tuning_constant(::Type{M}; eff=0.95, c0=1.0) where M<:SimpleEstimator
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
`MScaleEstimator(::Type{SimpleEstimator})` that returns the function r->(χ(r) - 1/2) to be called directly to find ŝ by solving:
Σ MScaleEstimator(ri/ŝ) = 0  with ri = yi - μi


```
using QuadGK: quadgk
using Roots: find_zero, Order1

function breakpoint_tuning_constant(::Type{M}; bp=1/2, c0=1.0) where M<:SimpleEstimator
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
struct L2Estimator <: ConvexEstimator; end
L2Estimator(c) = L2Estimator()
estimator_rho(   ::L2Estimator, r) = r^2 / 2
estimator_psi(   ::L2Estimator, r) = r
estimator_psider(::L2Estimator, r) = oftype(r, 1)
estimator_weight(::L2Estimator, r) = oftype(r, 1)
estimator_values(::L2Estimator, r) = (r^2/2, r, oftype(r, 1))

rho(   ::L2Estimator, r) = r^2 / 2
psi(   ::L2Estimator, r) = r
psider(::L2Estimator, r) = oftype(r, 1)
weight(::L2Estimator, r) = oftype(r, 1)



"""
The standard L1 M-estimator takes the absolute value of the residual, and is
convex but non-smooth. It is not a real L1 M-estimator but a Huber M-estimator
with very small tuning constant.
"""
struct L1Estimator <: ConvexEstimator; end
L1Estimator(c) = L1Estimator()
estimator_rho(   ::L1Estimator, r) = abs(r)
estimator_psi(   ::L1Estimator, r) = sign(r)
estimator_psider(::L1Estimator, r) = if (abs(r)<DELTA); oftype(r, 1) else oftype(r, 0) end
estimator_weight(::L1Estimator, r) = if (abs(r)<DELTA); L1WDELTA else 1/abs(r) end
function estimator_values(est::L1Estimator, r)
    rr = abs(r)
    return (rr, sign(r), (if (rr<DELTA); L1WDELTA else 1/rr end) )
end

rho(   ::L1Estimator, r) = abs(r)
psi(   ::L1Estimator, r) = sign(r)
psider(::L1Estimator, r) = if (abs(r)<DELTA); oftype(r, 1) else oftype(r, 0) end
weight(::L1Estimator, r) = if (abs(r)<DELTA); L1WDELTA else 1/abs(r) end



"""
The convex Huber estimator switches from between quadratic and linear cost/loss
function at a certain cutoff.
Tuning constant c is taken to give an efficiency of 95% for Gaussian residuals
ρc(r) = c^2 ρ(r/c) so ρc(ε) ≈ ε^2/2 for ε<<1
"""
struct HuberEstimator <: ConvexEstimator
    c::Float64

    HuberEstimator(c::Real) = new(c)
    HuberEstimator() = new(1.345)
end

estimator_rho(   est::HuberEstimator, r) = if (abs(r)<=est.c); r^2/2        else (est.c*abs(r) - est.c^2/2) end
estimator_psi(   est::HuberEstimator, r) = if (abs(r)<=est.c); r            else est.c*sign(r) end
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

estimator_high_efficiency_constant(::Type{HuberEstimator}) = 1.345
rho(   est::HuberEstimator, r) = if (abs(r)<=1); r^2/2        else (abs(r) - 1/2) end
psi(   est::HuberEstimator, r) = if (abs(r)<=1); r            else sign(r) end
psider(est::HuberEstimator, r) = if (abs(r)<=1); oftype(r, 1) else oftype(r, 0) end
weight(est::HuberEstimator, r) = if (abs(r)<=1); oftype(r, 1) else 1/abs(r) end



"""
The convex L1-L2 estimator interpolates smoothly between L2 behaviour for small
residuals and L1 for outliers.
"""
struct L1L2Estimator <: ConvexEstimator
    c::Float64

    L1L2Estimator(c::Real) = new(c)
    L1L2Estimator() = new(1.287)
end
estimator_rho(   est::L1L2Estimator, r) = est.c^2*(sqrt(1 + (r/est.c)^2) - 1)
estimator_psi(   est::L1L2Estimator, r) = r / sqrt(1 + (r/est.c)^2)
estimator_psider(est::L1L2Estimator, r) = 1 / (1 + (r/est.c)^2)^(3/2)
estimator_weight(est::L1L2Estimator, r) = 1 / sqrt(1 + (r/est.c)^2)
function estimator_values(est::L1L2Estimator, r)
    sqr = sqrt(1 + (r/est.c)^2)
    return (est.c^2*(sqr - 1), r/sqr, 1/sqr)
end

estimator_high_efficiency_constant(::Type{L1L2Estimator}) = 1.287
rho(   est::L1L2Estimator, r) = sqrt(1 + r^2) - 1
psi(   est::L1L2Estimator, r) = r / sqrt(1 + r^2)
psider(est::L1L2Estimator, r) = 1 / (1 + r^2)^(3/2)
weight(est::L1L2Estimator, r) = 1 / sqrt(1 + r^2)



"""
The (convex) "fair" estimator switches from between quadratic and linear
cost/loss function at a certain cutoff, and is C3 but non-analytic.
"""
struct FairEstimator <: ConvexEstimator
    c::Float64

    FairEstimator(c::Real) = new(c)
    FairEstimator() = new(1.400)
end
estimator_rho(   est::FairEstimator, r) = est.c*abs(r) - est.c^2*log(1 + abs(r/est.c))
estimator_psi(   est::FairEstimator, r) = r / (1 + abs(r)/est.c)
estimator_psider(est::FairEstimator, r) = 1 / (1 + abs(r)/est.c)^2
estimator_weight(est::FairEstimator, r) = 1 / (1 + abs(r)/est.c)
function estimator_values(est::FairEstimator, r)
    ir = 1/(1 + abs(r/est.c))
    return (est.c*abs(r) + est.c^2*log(ir), r*ir, ir)
end

estimator_high_efficiency_constant(::Type{FairEstimator}) = 1.400
rho(   est::FairEstimator, r) = abs(r) - log(1 + abs(r))
psi(   est::FairEstimator, r) = r / (1 + abs(r))
psider(est::FairEstimator, r) = 1 / (1 + abs(r))^2
weight(est::FairEstimator, r) = 1 / (1 + abs(r))


"""
The convex Log-Cosh estimator
log(cosh(r))
r * arctan(r) - 1/2*log(1 + r^2)
"""
struct LogcoshEstimator <: ConvexEstimator
    c::Float64

    LogcoshEstimator(c::Real) = new(c)
    LogcoshEstimator() = new(1.2047)
end
estimator_rho(   est::LogcoshEstimator, r) = (est.c)^2 * log(cosh(r/est.c))
estimator_psi(   est::LogcoshEstimator, r) = est.c * tanh(r/est.c)
estimator_psider(est::LogcoshEstimator, r) = 1 / (cosh(r/est.c))^2
estimator_weight(est::LogcoshEstimator, r) = if (abs(r/est.c)<DELTA); (1 - (r/est.c)^2/3) else est.c * tanh(r/est.c) / r end
function estimator_values(est::LogcoshEstimator, r)
    tr = est.c * tanh(r/est.c)
    rr = abs(r/est.c)
    return ( est.c^2*log(cosh(rr)), tr, (if (rr<DELTA); (1 - rr^2/3) else tr/r end) )
end

estimator_high_efficiency_constant(::Type{LogcoshEstimator}) = 1.2047
rho(   est::LogcoshEstimator, r) = log(cosh(r))
psi(   est::LogcoshEstimator, r) = tanh(r)
psider(est::LogcoshEstimator, r) = 1 / (cosh(r))^2
weight(est::LogcoshEstimator, r) = if (abs(r)<DELTA); (1 - r^2/3) else tanh(r) / r end


"""
The convex Arctan estimator
r * arctan(r) - 1/2*log(1 + r^2)
"""
struct ArctanEstimator <: ConvexEstimator
    c::Float64

    ArctanEstimator(c::Real) = new(c)
    ArctanEstimator() = new(0.919)
end
estimator_rho(   est::ArctanEstimator, r) = est.c * r * atan(r/est.c) - est.c^2/2*log(1 + (r/est.c)^2)
estimator_psi(   est::ArctanEstimator, r) = est.c * atan(r/est.c)
estimator_psider(est::ArctanEstimator, r) = 1 / (1 + (r/est.c)^2)
estimator_weight(est::ArctanEstimator, r) = if (abs(r/est.c)<DELTA); (1 - (r/est.c)^2/3) else est.c * atan(r/est.c) / r end
function estimator_values(est::ArctanEstimator, r)
    ar = est.c * atan(r/est.c)
    rr = abs(r/est.c)
    return ( r*ar - est.c^2/2*log(1 + rr^2), ar, (if (rr<DELTA); (1 - rr^2/3) else ar/r end) )
end

estimator_high_efficiency_constant(::Type{ArctanEstimator}) = 0.919
rho(   est::ArctanEstimator, r) = r * atan(r) - 1/2*log(1 + r^2)
psi(   est::ArctanEstimator, r) = atan(r)
psider(est::ArctanEstimator, r) = 1 / (1 + r^2)
weight(est::ArctanEstimator, r) = if (abs(r)<DELTA); (1 - r^2/3) else atan(r) / r end


"""
The non-convex Cauchy estimator switches from between quadratic behaviour to
logarithmic tails. This rejects outliers but may result in multiple minima.
"""
struct CauchyEstimator <: SimpleEstimator
    c::Float64

    CauchyEstimator(c::Real) = new(c)
    CauchyEstimator() = new(2.385)
end
estimator_rho(   est::CauchyEstimator, r) = (est.c)^2/2 * log(1 + (r/est.c)^2)
estimator_psi(   est::CauchyEstimator, r) = r / (1 + (r/est.c)^2)
estimator_psider(est::CauchyEstimator, r) = (1 - (r/est.c)^2) / (1 + (r/est.c)^2)^2
estimator_weight(est::CauchyEstimator, r) = 1 / (1 + (r/est.c)^2)
function estimator_values(est::CauchyEstimator, r)
    ir = 1/(1 + (r/est.c)^2)
    return ( -(est.c)^2/2 * log(ir), r*ir, ir )
end
isconvex( ::CauchyEstimator) = false
isbounded(::CauchyEstimator) = false

estimator_high_efficiency_constant(::Type{CauchyEstimator}) = 2.385
estimator_low_breakpoint_constant( ::Type{CauchyEstimator}) = 0.61200
estimator_chi(est::CauchyEstimator, r) = r*estimator_psi(est, r)/(est.c)^2

rho(   est::CauchyEstimator, r) = 1/2 * log(1 + r^2)
psi(   est::CauchyEstimator, r) = r / (1 + r^2)
psider(est::CauchyEstimator, r) = (1 - r^2) / (1 + r^2)^2
weight(est::CauchyEstimator, r) = 1 / (1 + r^2)


"""
The non-convex Geman-McClure for strong supression of outliers and does not guarantee a unique solution.
For S-Estimation, it is equivalent to the Cauchy estimator.
"""
struct GemanEstimator <: BoundedEstimator
    c::Float64

    GemanEstimator(c::Real) = new(c)
    GemanEstimator() = new(3.787)
end
estimator_rho(   est::GemanEstimator, r) = 1/2 * r^2 / (1 + (r/est.c)^2)
estimator_psi(   est::GemanEstimator, r) = r / (1 + (r/est.c)^2)^2
estimator_psider(est::GemanEstimator, r) = (1 - 3*(r/est.c)^2) / (1 + (r/est.c)^2)^3
estimator_weight(est::GemanEstimator, r) = 1 / (1 + (r/est.c)^2)^2
function estimator_values(est::GemanEstimator, r)
    ir = 1/(1 + (r/est.c)^2)
    return ( 1/2 * r^2 *ir, r*ir^2, ir^2 )
end
isconvex( ::GemanEstimator) = false
isbounded(::GemanEstimator) = true

estimator_high_efficiency_constant(::Type{GemanEstimator}) = 3.787
estimator_low_breakpoint_constant( ::Type{GemanEstimator}) = 0.61200
estimator_chi(est::GemanEstimator, r) = estimator_rho(est, r)/((est.c)^2/2)

rho(   est::GemanEstimator, r) = 1/2 * r^2 / (1 + r^2)
psi(   est::GemanEstimator, r) = r / (1 + r^2)^2
psider(est::GemanEstimator, r) = (1 - 3*r^2) / (1 + r^2)^3
weight(est::GemanEstimator, r) = 1 / (1 + r^2)^2



"""
The non-convex Welsch for strong supression of ourliers and does not guarantee a unique solution
"""
struct WelschEstimator <: BoundedEstimator
    c::Float64

    WelschEstimator(c::Real) = new(c)
    WelschEstimator() = new(2.985)
end
estimator_rho(   est::WelschEstimator, r) = -(est.c)^2/2 * Base.expm1(-(r/est.c)^2)
estimator_psi(   est::WelschEstimator, r) = r * exp(-(r/est.c)^2)
estimator_psider(est::WelschEstimator, r) = (1 - 2*(r/est.c)^2)*exp(-(r/est.c)^2)
estimator_weight(est::WelschEstimator, r) = exp(-(r/est.c)^2)
function estimator_values(est::WelschEstimator, r)
    er = exp(-(r/est.c)^2)
    return ( -(est.c)^2/2 * Base.expm1(-(r/est.c)^2), r*er, er )
end
isconvex( ::WelschEstimator) = false
isbounded(::WelschEstimator) = true

estimator_high_efficiency_constant(::Type{WelschEstimator}) = 2.985
estimator_low_breakpoint_constant( ::Type{WelschEstimator}) = 0.8165
estimator_chi(est::WelschEstimator, r) = estimator_rho(est, r)/((est.c)^2/2)

rho(   est::WelschEstimator, r) = -1/2 * Base.expm1(-r^2)
psi(   est::WelschEstimator, r) = r * exp(-r^2)
psider(est::WelschEstimator, r) = (1 - 2*r^2)*exp(-r^2)
weight(est::WelschEstimator, r) = exp(-r^2)



"""
The non-convex Tukey biweight estimator which completely suppresses the outliers,
and does not guaranty a unique solution
"""
struct TukeyEstimator <: BoundedEstimator
    c::Float64

    TukeyEstimator(c::Real) = new(c)
    TukeyEstimator() = new(4.685)
end
estimator_rho(   est::TukeyEstimator, r) = if (abs(r)<=est.c); (est.c)^2/6 * (1 - ( 1 - (r/est.c)^2 )^3) else (est.c)^2/6  end
estimator_psi(   est::TukeyEstimator, r) = if (abs(r)<=est.c); r*(1 - (r/est.c)^2)^2                     else oftype(r, 0) end
estimator_psider(est::TukeyEstimator, r) = if (abs(r)<=est.c); 1 - 6*(r/est.c)^2 + 5*(r/est.c)^4         else oftype(r, 0) end
estimator_weight(est::TukeyEstimator, r) = if (abs(r)<=est.c); (1 - (r/est.c)^2)^2                       else oftype(r, 0) end
function estimator_values(est::TukeyEstimator, r)
    pr = (abs(r)<=est.c) * (1 - (r/est.c)^2)
    return ( (est.c)^2/6*(1 - pr^3), r*pr^2, pr^2 )
end
isconvex( ::TukeyEstimator) = false
isbounded(::TukeyEstimator) = true

estimator_high_efficiency_constant(::Type{TukeyEstimator}) = 4.685
estimator_low_breakpoint_constant( ::Type{TukeyEstimator}) = 1.5476
estimator_chi(est::TukeyEstimator, r) = estimator_rho(est, r)/((est.c)^2/6)




######
###   MQuantile Estimators
######
quantile_weight(τ::Real, r::AbstractFloat) = oftype(r, 2*ifelse(r>0, τ, 1 - τ))


struct GeneralQuantileEstimator{E<:SimpleEstimator} <: AbstractQuantileEstimator
    est::E
    τ::Float64
end
GeneralQuantileEstimator{E}(τ::Real) where E<:SimpleEstimator = GeneralQuantileEstimator(E(), float(τ))

function show(io::IO, obj::GeneralQuantileEstimator)
    println(io, "$(MQuantile(obj.τ, obj))")
end

# Forward all methods to the `est` field
estimator_rho(   E::GeneralQuantileEstimator, r) = quantile_weight(E.τ, r) * estimator_rho(   E.est, r)
estimator_psi(   E::GeneralQuantileEstimator, r) = quantile_weight(E.τ, r) * estimator_psi(   E.est, r)
estimator_psider(E::GeneralQuantileEstimator, r) = quantile_weight(E.τ, r) * estimator_psider(E.est, r)
estimator_weight(E::GeneralQuantileEstimator, r) = quantile_weight(E.τ, r) * estimator_weight(E.est, r)
function estimator_values(E::GeneralQuantileEstimator, r)
    w = quantile_weight(E.τ, r)
    vals = estimator_values(E.est, r)
    Tuple([x * w for x in vals])
end
estimator_chi(   E::GeneralQuantileEstimator, r) = quantile_weight(E.τ, r) * estimator_chi(   E.est, r)
isbounded(E::GeneralQuantileEstimator) = isbounded(E.est)
isconvex( E::GeneralQuantileEstimator) = isconvex( E.est)
estimator_low_breakpoint_constant( E::GeneralQuantileEstimator) = estimator_low_breakpoint_constant( E.est)
estimator_high_efficiency_constant(E::GeneralQuantileEstimator) = estimator_high_efficiency_constant(E.est)


"""
The expectile estimator is a generalization of the L2 estimator,
that correspond to a mean estimator, for any value τ ∈ [0,1].

[1] Schnabel, Eilers - Computational Statistics and Data Analysis 53 (2009) 4168–4177 - Optimal expectile smoothing
doi:10.1016/j.csda.2009.05.002
"""
const ExpectileEstimator = GeneralQuantileEstimator{L2Estimator}

const QuantileEstimator = GeneralQuantileEstimator{L1Estimator}


#"""
#The quantile estimator is a generalization of the L1 estimator,
#that correspond to a median estimator, for any quantile value τ.

#[1] Schnabel, Eilers - Computational Statistics and Data Analysis 53 (2009) 4168–4177 - Optimal expectile smoothing
#doi:10.1016/j.csda.2009.05.002
#"""
#struct QuantileEstimator <: ConvexEstimator
#    τ::Float64

#    QuantileEstimator(τ::Real) = new(τ)
#    QuantileEstimator() = new(0.5)
#end

#estimator_rho(   est::QuantileEstimator, r) = quantile_weight(est.τ, r) * abs(r)
#estimator_psi(   est::QuantileEstimator, r) = quantile_weight(est.τ, r) * sign(r)
#estimator_psider(est::QuantileEstimator, r) = if (abs(r)<DELTA); oftype(r, 1) else oftype(r, 0) end
#estimator_weight(est::QuantileEstimator, r) = if (abs(r)<DELTA); r*(est.τ - 1/2)*L1WDELTA^2 + L1WDELTA/2 else quantile_weight(est.τ, r)/abs(r) end
#function estimator_values(est::QuantileEstimator, r)
#    w = quantile_weight(est.τ, r)
#    ww = if (abs(r)<DELTA); r*(est.τ - 1/2)*L1WDELTA^2 + L1WDELTA/2 else w/abs(r) end
#    return (w*abs(r), w*sign(r), ww)
#end


#"""
#The expectile estimator is a generalization of the L2 estimator,
#that correspond to a mean estimator, for any value τ ∈ [0,1].

#[1] Schnabel, Eilers - Computational Statistics and Data Analysis 53 (2009) 4168–4177 - Optimal expectile smoothing
#doi:10.1016/j.csda.2009.05.002
#"""
#struct ExpectileEstimator <: ConvexEstimator
#    τ::Float64

#    ExpectileEstimator(τ::Real) = new(τ)
#    ExpectileEstimator() = new(0.5)
#end

#estimator_rho(   est::ExpectileEstimator, r) = quantile_weight(est.τ, r) * r^2 / 2
#estimator_psi(   est::ExpectileEstimator, r) = quantile_weight(est.τ, r) * r
#estimator_psider(est::ExpectileEstimator, r) = quantile_weight(est.τ, r)
#estimator_weight(est::ExpectileEstimator, r) = quantile_weight(est.τ, r)
#function estimator_values(est::ExpectileEstimator, r)
#    w = quantile_weight(est.τ, r)
#    return (w*r^2/2, w*r, w)
#end

