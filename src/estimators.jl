
using QuadGK: quadgk
using Roots: find_zero, Order1, ConvergenceFailed

## Threshold to avoid numerical overflow of the weight function of L1Estimator and ArctanEstimator
DELTA = 1e-8    # chosen because it minimizes the error between (1-ATWDELTA)/DELTA and 1/3 for the linear approximation
L1WDELTA = 1/(DELTA)
ATWDELTA = atan(DELTA)/DELTA
#DELTA = 1e-8    # chosen because it minimizes the error between (1-ATWDELTA)/DELTA and 1/3 for the linear approximation
#L1WDELTA = 1/(2*sqrt(DELTA))
#ATWDELTA = atan(sqrt(DELTA))*2*L1WDELTA




"The loss function ρ for the M-estimator."
function estimator_rho end

"""
The derivative ψ of the loss function for the M-estimator, multiplied by the square of 
the tuning constant. ψ is proportional to the influence function.
"""
function estimator_psi end

"The derivative of ψ for the M-estimator"
function estimator_psider end

"The function ψ divided by r for the M-estimator"
function estimator_weight end

"The integral of exp(-ρ) used for calculating the full-loglikelihood for the M-estimator"
function estimator_norm end


isconvex( e::SimpleEstimator) = isa(e, ConvexEstimator)
isbounded(e::SimpleEstimator) = isa(e, BoundedEstimator)

estimator_high_breakdown_point_constant(::Type{E}) where {E<:SimpleEstimator} = 1
estimator_high_efficiency_constant(::Type{E}) where {E<:SimpleEstimator} = 1


"""
    estimator_chi(::M, r) where M<:SimpleEstimator
The function derived from the estimator for M-estimation of scale.
It is bounded with lim_{t->∞} χ = 1
It can be proportional to ρ or t.ψ(t) depending on the estimator.
"""
function estimator_chi(::M, r::Real) where M<:SimpleEstimator
    error("This estimator cannot be used for scale estimation: $(M)")
end

"""
    scale_estimate(::Estimator, args...; kwargs...)
Estimate the scale using the function χ, it is the solution of a non-linear equation.
If the estimator is not bounded, it gives an error.
"""
function scale_estimate(::Estimator, args...; kwargs...)
    error("scale estimate is only defined for bounded estimators.")
end



"""
The M-estimator norm is computed with:
     +∞                    +∞
Z = ∫  exp(-ρ(r))dr = c . ∫  exp(-ρ_1(r))dr    with ρ_1 the function for c=1
    -∞                    -∞
"""
function estimator_norm(est::E) where {E<:Estimator}
    2*quadgk(x->exp(-estimator_rho(est, x)), 0, Inf)[1]
end


"""
The tuning constant c is computed so the efficiency for Normally distributed
residuals is 0.95. The efficiency of the mean estimate μ is defined by:
eff_μ = (E[ψ'])²/E[ψ²]
"""
function efficiency_tuning_constant(::Type{M}; eff::Real=0.95, c0::Real=1.0) where M<:SimpleEstimator
    psi(x, c)  = RobustModels.estimator_psi(M(c), x)
    psip(x, c) = RobustModels.estimator_psider(M(c), x)

    I1(c) = quadgk(x->(psi(x, c))^2*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    I2(c) = quadgk(x->psip(x, c)*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    fun_eff(c) = (I2(c))^2/I1(c)
    copt = find_zero(c->fun_eff(c) - eff, c0, Order1())
end

"""
The M-estimate of scale is computed by solving:

1/n Σ χ(r/ŝ) = δ

with χ a bounded function with χ(∞) = 1 and δ = E[χ]/χ(∞) with expectation w.r.t. Normal density.
The parameter `c` of χ should be chosen such that δ = 1/2, which
corresponds to a breakdown-point of 50%.
The function χ can be directly the pseudo-negloglikelihood ρ or `t.ψ(t)`.
`estimator_chi` returns this function when it is define, but it is better to call
`MScaleEstimator(::Type{SimpleEstimator})` that returns the function r->(χ(r) - 1/2) to be called directly to find ŝ by solving:
Σ MScaleEstimator(ri/ŝ) = 0  with ri = yi - μi
"""
function breakdown_point_tuning_constant(::Type{M}; bp::Real=1/2, c0::Real=1.0) where M<:SimpleEstimator
    (0 < bp <= 1/2) || error("breakdown-point should be between 0 and 1/2")

    if !(M <: BoundedEstimator)
        error("optimizing the tuning constant for high breakdown-point is only defined for bounded estimators.")
    end

    I(c) = quadgk(x->RobustModels.estimator_chi(M(c), x)*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    copt = find_zero(c->I(c) - bp, c0, Order1())
end


########
###     Simple Estimators    
########

"The (convex) L2 M-estimator is that of the standard least squares problem."
struct L2Estimator <: ConvexEstimator; end
L2Estimator(c) = L2Estimator()
estimator_rho(   ::L2Estimator, r::Real) = r^2 / 2
estimator_psi(   ::L2Estimator, r::Real) = r
estimator_psider(::L2Estimator, r::Real) = oftype(r, 1)
estimator_weight(::L2Estimator, r::Real) = oftype(r, 1)
estimator_values(::L2Estimator, r::Real) = (r^2/2, r, oftype(r, 1))
estimator_norm(::L2Estimator, ln=false) = √(2*π)

rho(   ::L2Estimator, r::Real) = r^2 / 2
psi(   ::L2Estimator, r::Real) = r
psider(::L2Estimator, r::Real) = oftype(r, 1)
weight(::L2Estimator, r::Real) = oftype(r, 1)



"""
The standard L1 M-estimator takes the absolute value of the residual, and is
convex but non-smooth. It is not a real L1 M-estimator but a Huber M-estimator
with very small tuning constant.
"""
struct L1Estimator <: ConvexEstimator; end
L1Estimator(c) = L1Estimator()
estimator_rho(   ::L1Estimator, r::Real) = abs(r)
estimator_psi(   ::L1Estimator, r::Real) = sign(r)
estimator_psider(::L1Estimator, r::Real) = if (abs(r)<DELTA); oftype(r, 1) else oftype(r, 0) end
estimator_weight(::L1Estimator, r::Real) = if (abs(r)<DELTA); L1WDELTA else 1/abs(r) end
function estimator_values(est::L1Estimator, r::Real)
    rr = abs(r)
    return (rr, sign(r), (if (rr<DELTA); L1WDELTA else 1/rr end) )
end
estimator_norm(::L1Estimator) = 2

rho(   ::L1Estimator, r::Real) = abs(r)
psi(   ::L1Estimator, r::Real) = sign(r)
psider(::L1Estimator, r::Real) = if (abs(r)<DELTA); oftype(r, 1) else oftype(r, 0) end
weight(::L1Estimator, r::Real) = if (abs(r)<DELTA); L1WDELTA else 1/abs(r) end



"""
The convex Huber estimator switches from between quadratic and linear cost/loss
function at a certain cutoff.
"""
struct HuberEstimator <: ConvexEstimator
    c::Float64

    HuberEstimator(c::Real) = new(c)
    HuberEstimator() = new(1.345)
end

#estimator_rho(   est::HuberEstimator, r::Real) = if (abs(r)<=est.c); r^2/2         else (est.c*abs(r) - est.c^2/2) end
estimator_rho(   est::HuberEstimator, r::Real) = if (abs(r)<=est.c); (r/est.c)^2/2 else (abs(r)/est.c - 1/2) end
estimator_psi(   est::HuberEstimator, r::Real) = if (abs(r)<=est.c); r             else est.c*sign(r) end
estimator_psider(est::HuberEstimator, r::Real) = if (abs(r)<=est.c); oftype(r, 1)  else oftype(r, 0) end
estimator_weight(est::HuberEstimator, r::Real) = if (abs(r)<=est.c); oftype(r, 1)  else est.c/abs(r) end
function estimator_values(est::HuberEstimator, r::Real)
    rr = abs(r)
    if rr <= est.c
        return ((rr/est.c)^2/2 , r , oftype(r, 1) )
    else
        return (rr/est.c - 1/2 , est.c*sign(r) , est.c/rr )
    end
end
#estimator_norm(est::HuberEstimator) = est.c * (√(2π)*erf(1/√2) + 2*exp(-1/2))
estimator_norm(est::HuberEstimator) = est.c * 2.92431

estimator_high_efficiency_constant(::Type{HuberEstimator}) = 1.345
rho(   est::HuberEstimator, r::Real) = if (abs(r)<=1); r^2/2        else (abs(r) - 1/2) end
psi(   est::HuberEstimator, r::Real) = if (abs(r)<=1); r            else sign(r) end
psider(est::HuberEstimator, r::Real) = if (abs(r)<=1); oftype(r, 1) else oftype(r, 0) end
weight(est::HuberEstimator, r::Real) = if (abs(r)<=1); oftype(r, 1) else 1/abs(r) end



"""
The convex L1-L2 estimator interpolates smoothly between L2 behaviour for small
residuals and L1 for outliers.
"""
struct L1L2Estimator <: ConvexEstimator
    c::Float64

    L1L2Estimator(c::Real) = new(c)
    L1L2Estimator() = new(1.287)
end
estimator_rho(   est::L1L2Estimator, r::Real) = (sqrt(1 + (r/est.c)^2) - 1)
estimator_psi(   est::L1L2Estimator, r::Real) = r / sqrt(1 + (r/est.c)^2)
estimator_psider(est::L1L2Estimator, r::Real) = 1 / (1 + (r/est.c)^2)^(3/2)
estimator_weight(est::L1L2Estimator, r::Real) = 1 / sqrt(1 + (r/est.c)^2)
function estimator_values(est::L1L2Estimator, r::Real)
    sqr = sqrt(1 + (r/est.c)^2)
    return ((sqr - 1), r/sqr, 1/sqr)
end
estimator_norm(est::L1L2Estimator) = est.c * 3.2723

estimator_high_efficiency_constant(::Type{L1L2Estimator}) = 1.287
rho(   est::L1L2Estimator, r::Real) = sqrt(1 + r^2) - 1
psi(   est::L1L2Estimator, r::Real) = r / sqrt(1 + r^2)
psider(est::L1L2Estimator, r::Real) = 1 / (1 + r^2)^(3/2)
weight(est::L1L2Estimator, r::Real) = 1 / sqrt(1 + r^2)



"""
The (convex) "fair" estimator switches from between quadratic and linear
cost/loss function at a certain cutoff, and is C3 but non-analytic.
"""
struct FairEstimator <: ConvexEstimator
    c::Float64

    FairEstimator(c::Real) = new(c)
    FairEstimator() = new(1.400)
end
estimator_rho(   est::FairEstimator, r::Real) = abs(r)/est.c - log(1 + abs(r/est.c))
estimator_psi(   est::FairEstimator, r::Real) = r / (1 + abs(r)/est.c)
estimator_psider(est::FairEstimator, r::Real) = 1 / (1 + abs(r)/est.c)^2
estimator_weight(est::FairEstimator, r::Real) = 1 / (1 + abs(r)/est.c)
function estimator_values(est::FairEstimator, r::Real)
    ir = 1/(1 + abs(r/est.c))
    return (abs(r)/est.c + log(ir), r*ir, ir)
end
estimator_norm(est::FairEstimator) = est.c * 4

estimator_high_efficiency_constant(::Type{FairEstimator}) = 1.400
rho(   est::FairEstimator, r::Real) = abs(r) - log(1 + abs(r))
psi(   est::FairEstimator, r::Real) = r / (1 + abs(r))
psider(est::FairEstimator, r::Real) = 1 / (1 + abs(r))^2
weight(est::FairEstimator, r::Real) = 1 / (1 + abs(r))


"""
The convex Log-Cosh estimator
log(cosh(r))
"""
struct LogcoshEstimator <: ConvexEstimator
    c::Float64

    LogcoshEstimator(c::Real) = new(c)
    LogcoshEstimator() = new(1.2047)
end
estimator_rho(   est::LogcoshEstimator, r::Real) = log(cosh(r/est.c))
estimator_psi(   est::LogcoshEstimator, r::Real) = est.c * tanh(r/est.c)
estimator_psider(est::LogcoshEstimator, r::Real) = 1 / (cosh(r/est.c))^2
estimator_weight(est::LogcoshEstimator, r::Real) = if (abs(r/est.c)<DELTA); (1 - (r/est.c)^2/3) else est.c * tanh(r/est.c) / r end
function estimator_values(est::LogcoshEstimator, r::Real)
    tr = est.c * tanh(r/est.c)
    rr = abs(r/est.c)
    return ( log(cosh(rr)), tr, (if (rr<DELTA); (1 - rr^2/3) else tr/r end) )
end
estimator_norm(est::LogcoshEstimator) = est.c * π

estimator_high_efficiency_constant(::Type{LogcoshEstimator}) = 1.2047
rho(   est::LogcoshEstimator, r::Real) = log(cosh(r))
psi(   est::LogcoshEstimator, r::Real) = tanh(r)
psider(est::LogcoshEstimator, r::Real) = 1 / (cosh(r))^2
weight(est::LogcoshEstimator, r::Real) = if (abs(r)<DELTA); (1 - r^2/3) else tanh(r) / r end


"""
The convex Arctan estimator
r * arctan(r) - 1/2*log(1 + r^2)
"""
struct ArctanEstimator <: ConvexEstimator
    c::Float64

    ArctanEstimator(c::Real) = new(c)
    ArctanEstimator() = new(0.919)
end
estimator_rho(   est::ArctanEstimator, r::Real) =  r / est.c * atan(r/est.c) - 1/2*log(1 + (r/est.c)^2)
estimator_psi(   est::ArctanEstimator, r::Real) = est.c * atan(r/est.c)
estimator_psider(est::ArctanEstimator, r::Real) = 1 / (1 + (r/est.c)^2)
estimator_weight(est::ArctanEstimator, r::Real) = if (abs(r/est.c)<DELTA); (1 - (r/est.c)^2/3) else est.c * atan(r/est.c) / r end
function estimator_values(est::ArctanEstimator, r::Real)
    ar = atan(r/est.c)
    rr = abs(r/est.c)
    return ( r*ar/est.c - 1/2*log(1 + rr^2), est.c*ar, (if (rr<DELTA); (1 - rr^2/3) else est.c*ar/r end) )
end
estimator_norm(est::ArctanEstimator) = est.c * 2.98151

estimator_high_efficiency_constant(::Type{ArctanEstimator}) = 0.919
rho(   est::ArctanEstimator, r::Real) = r * atan(r) - 1/2*log(1 + r^2)
psi(   est::ArctanEstimator, r::Real) = atan(r)
psider(est::ArctanEstimator, r::Real) = 1 / (1 + r^2)
weight(est::ArctanEstimator, r::Real) = if (abs(r)<DELTA); (1 - r^2/3) else atan(r) / r end


"""
The non-convex Cauchy estimator switches from between quadratic behaviour to
logarithmic tails. This rejects outliers but may result in multiple minima.
"""
struct CauchyEstimator <: SimpleEstimator
    c::Float64

    CauchyEstimator(c::Real) = new(c)
    CauchyEstimator() = new(2.385)
end
estimator_rho(   est::CauchyEstimator, r::Real) = log(1 + (r/est.c)^2) # * 1/2  # remove factor 1/2 so the estimator has a norm
estimator_psi(   est::CauchyEstimator, r::Real) = r / (1 + (r/est.c)^2)
estimator_psider(est::CauchyEstimator, r::Real) = (1 - (r/est.c)^2) / (1 + (r/est.c)^2)^2
estimator_weight(est::CauchyEstimator, r::Real) = 1 / (1 + (r/est.c)^2)
function estimator_values(est::CauchyEstimator, r::Real)
    ir = 1/(1 + (r/est.c)^2)
    return ( - log(ir), r*ir, ir )
end
estimator_norm(est::CauchyEstimator) = est.c * π
isconvex( ::CauchyEstimator) = false
isbounded(::CauchyEstimator) = false

estimator_high_efficiency_constant(::Type{CauchyEstimator}) = 2.385
estimator_high_breakdown_point_constant( ::Type{CauchyEstimator}) = 0.61200
estimator_chi(est::CauchyEstimator, r::Real) = r*estimator_psi(est, r)/(est.c)^2

rho(   est::CauchyEstimator, r::Real) = 1/2 * log(1 + r^2)
psi(   est::CauchyEstimator, r::Real) = r / (1 + r^2)
psider(est::CauchyEstimator, r::Real) = (1 - r^2) / (1 + r^2)^2
weight(est::CauchyEstimator, r::Real) = 1 / (1 + r^2)


"""
The non-convex Geman-McClure for strong supression of outliers and does not guarantee a unique solution.
For S-Estimation, it is equivalent to the Cauchy estimator.
"""
struct GemanEstimator <: BoundedEstimator
    c::Float64

    GemanEstimator(c::Real) = new(c)
    GemanEstimator() = new(3.787)
end
estimator_rho(   est::GemanEstimator, r::Real) = 1/2 * (r/est.c)^2 / (1 + (r/est.c)^2)
estimator_psi(   est::GemanEstimator, r::Real) = r / (1 + (r/est.c)^2)^2
estimator_psider(est::GemanEstimator, r::Real) = (1 - 3*(r/est.c)^2) / (1 + (r/est.c)^2)^3
estimator_weight(est::GemanEstimator, r::Real) = 1 / (1 + (r/est.c)^2)^2
function estimator_values(est::GemanEstimator, r::Real)
    ir = 1/(1 + (r/est.c)^2)
    return ( 1/2 * (r/est.c)^2 *ir, r*ir^2, ir^2 )
end
estimator_norm(est::GemanEstimator) = Inf
isconvex( ::GemanEstimator) = false
isbounded(::GemanEstimator) = true

estimator_high_efficiency_constant(::Type{GemanEstimator}) = 3.787
estimator_high_breakdown_point_constant( ::Type{GemanEstimator}) = 0.61200
estimator_chi(est::GemanEstimator, r::Real) = estimator_rho(est, r)*2

rho(   est::GemanEstimator, r::Real) = 1/2 * r^2 / (1 + r^2)
psi(   est::GemanEstimator, r::Real) = r / (1 + r^2)^2
psider(est::GemanEstimator, r::Real) = (1 - 3*r^2) / (1 + r^2)^3
weight(est::GemanEstimator, r::Real) = 1 / (1 + r^2)^2



"""
The non-convex Welsch for strong supression of ourliers and does not guarantee a unique solution
"""
struct WelschEstimator <: BoundedEstimator
    c::Float64

    WelschEstimator(c::Real) = new(c)
    WelschEstimator() = new(2.985)
end
estimator_rho(   est::WelschEstimator, r::Real) = -1/2 * Base.expm1(-(r/est.c)^2)
estimator_psi(   est::WelschEstimator, r::Real) = r * exp(-(r/est.c)^2)
estimator_psider(est::WelschEstimator, r::Real) = (1 - 2*(r/est.c)^2)*exp(-(r/est.c)^2)
estimator_weight(est::WelschEstimator, r::Real) = exp(-(r/est.c)^2)
function estimator_values(est::WelschEstimator, r::Real)
    er = exp(-(r/est.c)^2)
    return ( -1/2 * Base.expm1(-(r/est.c)^2), r*er, er )
end
estimator_norm(est::WelschEstimator) = Inf
isconvex( ::WelschEstimator) = false
isbounded(::WelschEstimator) = true

estimator_high_efficiency_constant(::Type{WelschEstimator}) = 2.985
estimator_high_breakdown_point_constant( ::Type{WelschEstimator}) = 0.8165
estimator_chi(est::WelschEstimator, r::Real) = estimator_rho(est, r)*2

rho(   est::WelschEstimator, r::Real) = -1/2 * Base.expm1(-r^2)
psi(   est::WelschEstimator, r::Real) = r * exp(-r^2)
psider(est::WelschEstimator, r::Real) = (1 - 2*r^2)*exp(-r^2)
weight(est::WelschEstimator, r::Real) = exp(-r^2)



"""
The non-convex Tukey biweight estimator which completely suppresses the outliers,
and does not guaranty a unique solution
"""
struct TukeyEstimator <: BoundedEstimator
    c::Float64

    TukeyEstimator(c::Real) = new(c)
    TukeyEstimator() = new(4.685)
end
estimator_rho(   est::TukeyEstimator, r::Real) = if (abs(r)<=est.c); 1/6 * (1 - ( 1 - (r/est.c)^2 )^3) else 1/6  end
estimator_psi(   est::TukeyEstimator, r::Real) = if (abs(r)<=est.c); r*(1 - (r/est.c)^2)^2             else oftype(r, 0) end
estimator_psider(est::TukeyEstimator, r::Real) = if (abs(r)<=est.c); 1 - 6*(r/est.c)^2 + 5*(r/est.c)^4 else oftype(r, 0) end
estimator_weight(est::TukeyEstimator, r::Real) = if (abs(r)<=est.c); (1 - (r/est.c)^2)^2               else oftype(r, 0) end
function estimator_values(est::TukeyEstimator, r::Real)
    pr = (abs(r)<=est.c) * (1 - (r/est.c)^2)
    return ( 1/6*(1 - pr^3), r*pr^2, pr^2 )
end
estimator_norm(est::TukeyEstimator) = Inf
isconvex( ::TukeyEstimator) = false
isbounded(::TukeyEstimator) = true

estimator_high_efficiency_constant(::Type{TukeyEstimator}) = 4.685
estimator_high_breakdown_point_constant( ::Type{TukeyEstimator}) = 1.5476
estimator_chi(est::TukeyEstimator, r::Real) = estimator_rho(est, r)*6




######
###   MQuantile Estimators
######
quantile_weight(τ::Real, r::Real) = oftype(r, 2*ifelse(r>0, τ, 1 - τ))


struct GeneralQuantileEstimator{E<:SimpleEstimator} <: AbstractQuantileEstimator
    est::E
    τ::Float64
end
GeneralQuantileEstimator{E}(τ::Real) where E<:SimpleEstimator = GeneralQuantileEstimator(E(), float(τ))

function show(io::IO, obj::GeneralQuantileEstimator)
    print(io, "MQuantile($(obj.τ), $(obj.est))")
end

# Forward all methods to the `est` field
estimator_rho(   E::GeneralQuantileEstimator, r::Real) = quantile_weight(E.τ, r) * estimator_rho(   E.est, r)
estimator_psi(   E::GeneralQuantileEstimator, r::Real) = quantile_weight(E.τ, r) * estimator_psi(   E.est, r)
estimator_psider(E::GeneralQuantileEstimator, r::Real) = quantile_weight(E.τ, r) * estimator_psider(E.est, r)
estimator_weight(E::GeneralQuantileEstimator, r::Real) = quantile_weight(E.τ, r) * estimator_weight(E.est, r)
function estimator_values(E::GeneralQuantileEstimator, r)
    w = quantile_weight(E.τ, r)
    vals = estimator_values(E.est, r)
    Tuple([x * w for x in vals])
end
estimator_norm(E::GeneralQuantileEstimator, args...) = estimator_norm(E.est, args...)
estimator_chi(   E::GeneralQuantileEstimator, r::Real) = quantile_weight(E.τ, r) * estimator_chi(   E.est, r)
isbounded(E::GeneralQuantileEstimator) = isbounded(E.est)
isconvex( E::GeneralQuantileEstimator) = isconvex( E.est)
estimator_high_breakdown_point_constant( E::GeneralQuantileEstimator) = estimator_high_breakdown_point_constant( E.est)
estimator_high_efficiency_constant(E::GeneralQuantileEstimator) = estimator_high_efficiency_constant(E.est)


"""
The expectile estimator is a generalization of the L2 estimator,
that correspond to a mean estimator, for any value τ ∈ [0,1].

[1] Schnabel, Eilers - Computational Statistics and Data Analysis 53 (2009) 4168–4177 - Optimal expectile smoothing
doi:10.1016/j.csda.2009.05.002
"""
const ExpectileEstimator = GeneralQuantileEstimator{L2Estimator}

const QuantileEstimator = GeneralQuantileEstimator{L1Estimator}

const UnionL1 = Union{L1Estimator, GeneralQuantileEstimator{L1Estimator}}


"""
    SEstimator(est::M; fallback::Type{BoundedEstimator}=TukeyEstimator, force=false)
Return an S-Estimator based on an M-Estimator. If the M-Estimator is not bounded,
return an S-Estimator of the same kind as the fallback.
If force is true, the S-Estimator will be of the fallback kind.
"""
function SEstimator(est::SimpleEstimator; fallback::Type{E}=TukeyEstimator,
            force::Bool=false)::SimpleEstimator where {E<:BoundedEstimator}
    ## Change the estimator to the same estimator with a tuning constant that gives a low breakdown point
    typ = typeof(est)
    if force || !isbounded(est)
        typ = fallback
    end
    typ(estimator_high_breakdown_point_constant(typ))
end

function SEstimator(est::GeneralQuantileEstimator{E}; kwargs...)::GeneralQuantileEstimator where E<:SimpleEstimator
    simple_est = SEstimator(est.est; kwargs...)
    GeneralQuantileEstimator(simple_est, est.τ)
end


function scale_estimate(est::Union{E, GeneralQuantileEstimator{E}}, res::AbstractArray{T};
            σ0::T=1.0, wts::AbstractArray{T}=T[], use_reciprocal::Bool=true) where {T<:AbstractFloat, E<:BoundedEstimator}
    
    σest(x) = estimator_chi(est, x) - 1/2

    if isempty(wts)
        if use_reciprocal
            1/find_zero(s->sum(σest.(res .* s)), 1/σ0, Order1())
        else
            find_zero(s->sum(σest.(res ./ s)), σ0, Order1())
        end
    else
        if use_reciprocal
            1/find_zero(s->sum(r.wts .* σest.(res .* s)), 1/σ0, Order1())
        else
            find_zero(s->sum(r.wts .* σest.(res ./ s)), σ0, Order1())
        end
    end
end
function scale_estimate(est::E, r::AbstractArray{T};
        σ0::Real=1, wts=[], kwargs...) where {T<:Real, E<:Estimator}
    scale_estimate(r; σ0=float(σ0), wts=float(wts), kwargs...)
end

