
using QuadGK: quadgk
using Roots: find_zero, Order1, ConvergenceFailed, Newton

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

"The integral of exp(-ρ) used for calculating the full-loglikelihood for the M-estimator"
function estimator_values(est::Estimator, r::Real)
    return (estimator_rho(est, r), estimator_psi(est, r), estimator_weight(est, r))
end


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
    estimator_rchider(::M, r) where M<:SimpleEstimator
The function `r . χ'(r)`, to use to estimate the variance of the asymptotic scale.
"""
function estimator_rchider(::M, r::Real) where M<:SimpleEstimator
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



"""
The convex Huber estimator switches from between quadratic and linear cost/loss
function at a certain cutoff.
"""
struct HuberEstimator <: ConvexEstimator
    c::Float64

    HuberEstimator(c::Real) = new(c)
    HuberEstimator() = new(1.345)
end

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
estimator_norm(est::HuberEstimator) = est.c * 2.92431
estimator_high_efficiency_constant(::Type{HuberEstimator}) = 1.345



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
estimator_rchider(est::GemanEstimator, r::Real) = 2*r*estimator_psi(est, r)



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
estimator_rchider(est::WelschEstimator, r::Real) = 2*r*estimator_psi(est, r)


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
estimator_rchider(est::TukeyEstimator, r::Real) = 6*r*estimator_psi(est, r)


"""
The non-convex (and bounded) optimal Yohai-Zamar estimator that
minimizes the estimator bias. It was originally introduced in 
Optimal locally robust M-estimates of regression (1997) by Yohai and Zamar
with a slightly different formula.
"""
struct YohaiZamarEstimator <: BoundedEstimator
    c::Float64

    YohaiZamarEstimator(c::Real) = new(c)
    YohaiZamarEstimator() = new(3.1806)
end
function estimator_rho(est::YohaiZamarEstimator, r::Real)
    z = (r/est.c)^2
    if (z<=4/9)
        1.3846 * z
    elseif (z<=1)
#        evalpoly(z, (0.55, -2.69, 10.76, -11.66, 4.04))
        min(1.0, 0.5514 - 2.6917*z + 10.7668*z^2 - 11.6640*z^3 + 4.0375*z^4)
#        0.55 - 2.69*z + 10.76*z^2 - 11.66*z^3 + 4.04*z^4
    else
        oftype(r, 1)
    end
end
function estimator_psi(est::YohaiZamarEstimator, r::Real)
    z = (r/est.c)^2
    if (z<=4/9)
        2.7692 * r
    elseif (z<=1)
#        r*evalpoly(z, (-5.38, 43.04, -69.96, 32.32))
        r*max(0, -5.3834 + 43.0672*z - 69.984*z^2 + 32.3*z^3)
    else
        oftype(r, 0)
    end
end
function estimator_psider(est::YohaiZamarEstimator, r::Real)
    z = (r/est.c)^2
    if (z<=4/9)
        2.7692
#    elseif (z<=1)
    elseif (z<=0.997284)  # from the root of ψ expression
#        evalpoly(z, (-5.3834, 129.2016, -349.92, 226.1))
        -5.3834 + 129.2016*z - 349.92*z^2 + 226.1*z^3
    else
        oftype(r, 0)
    end
end
function estimator_weight(est::YohaiZamarEstimator, r::Real)
    z = (r/est.c)^2
    if (z<=4/9)
        2.7692
    elseif (z<=1)
#        evalpoly(z, (-5.38, 43.04, -69.96, 32.32))
        max(0, -5.3834 + 43.0672*z - 69.984*z^2 + 32.3*z^3)
    else
        oftype(r, 0)
    end
end
function estimator_values(est::YohaiZamarEstimator, r::Real)
    z = (r/est.c)^2
    if (z<=4/9)
        return (1.3846*z, 2.7692*r, 2.7692)
    elseif (z<=1)
        ρ = min(1, 0.5514 - 2.6917*z + 10.7668*z^2 - 11.6640*z^3 + 4.0375*z^4)
        w = max(0, -5.3834 + 43.0672*z - 69.984*z^2 + 32.3*z^3)
        return (ρ, r*w, w)
    else
        return (oftype(r, 1), oftype(r, 0), oftype(r, 0))
    end
end
estimator_norm(est::YohaiZamarEstimator) = Inf
isconvex( ::YohaiZamarEstimator) = false
isbounded(::YohaiZamarEstimator) = true

estimator_high_efficiency_constant(::Type{YohaiZamarEstimator}) = 3.1806
estimator_high_breakdown_point_constant( ::Type{YohaiZamarEstimator}) = 1.2139
estimator_chi(est::YohaiZamarEstimator, r::Real) = estimator_rho(est, r)
estimator_rchider(est::YohaiZamarEstimator, r::Real) = r*estimator_psi(est, r)


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
#estimator_high_breakdown_point_constant( E::GeneralQuantileEstimator) = estimator_high_breakdown_point_constant( E.est)
#estimator_high_efficiency_constant(E::GeneralQuantileEstimator) = estimator_high_efficiency_constant(E.est)


"""
The expectile estimator is a generalization of the L2 estimator,
that correspond to a mean estimator, for any value τ ∈ [0,1].

[1] Schnabel, Eilers - Computational Statistics and Data Analysis 53 (2009) 4168–4177 - Optimal expectile smoothing
doi:10.1016/j.csda.2009.05.002
"""
const ExpectileEstimator = GeneralQuantileEstimator{L2Estimator}

const QuantileEstimator = GeneralQuantileEstimator{L1Estimator}

const UnionL1 = Union{L1Estimator, GeneralQuantileEstimator{L1Estimator}}


######
###   S-Estimators
######


#mutable struct SEstimator{E1<:BoundedEstimator, E2<:SimpleEstimator} <: Estimator
#    est1::E1
#    est2::E2
#end

#function SEstimator(::Type{E}) where E<:BoundedEstimator
#    est1 = E(estimator_high_breakdown_point_constant(E))
#    est2 = E(estimator_high_efficiency_constant(E))
#    SEstimator(est1, est2)
#end

#function SEstimator(::Type{E1}, ::Type{E2}) where {E1<:BoundedEstimator, E2<:SimpleEstimator}
#    est1 = E1(estimator_high_breakdown_point_constant(E1))
#    est2 = E2(estimator_high_efficiency_constant(E2))
#    SEstimator(est1, est2)
#end

#function SEstimator(est::SimpleEstimator; fallback::Type{E}=TukeyEstimator,
#            force::Bool=false) where {E<:BoundedEstimator}
#    ## Change the estimator to the same estimator with a tuning constant that gives a low breakdown point
#    typ = typeof(est)
#    if force || !isbounded(est)
#        typ = fallback
#    end
#    SEstimator(typ)
#end

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
            bound::T=0.5, σ0::T=1.0, wts::AbstractArray{T}=T[], verbose::Bool=false,
            order::Int=1, approx::Bool=false, nmax::Int=30, 
            rtol::Real=1e-4, atol::Real=0.1) where {T<:AbstractFloat, E<:BoundedEstimator}
    
    Nz = sum(iszero, res)
    if Nz > length(res)*(1-bound)
        # The M-scale cannot be estimated because too many residuals are zeros.
        verbose && println("there are too many zero residuals for M-scale estimation: #(r=0) > n*(1-b), $(Nz) > $(length(res)*(1-bound))")
        throw(ConvergenceFailed("the M-scale cannot be estimated because too many residuals are zeros: $(res)"))
#            return zero(T)
    end

    # Approximate the solution with `nmax` iterations
    σn = σ0
    converged = false
    verbose && println("Initial M-scale estimate: $(σn)")
    for n in 1:nmax
        ε = if length(wts) == length(res)
            mean( estimator_chi.(est, res ./ σn), weights(wts) ) / bound
        else
            mean(x->estimator_chi(est, x / σn), res) / bound
        end
        verbose && println("M-scale 1st order update: $(ε)")
        
        ## Implemented, but it gives worst results than 1st order...
        if !approx && order>=2
            εp = if length(wts) == length(res)
                mean( estimator_rchider.(est, res ./ σn), weights(wts) ) / bound
            else
                mean(x->estimator_rchider(est, x / σn), res) / bound
            end
            # Use the gradient only if the scale is not too small
            if isnan(εp) || !isfinite(εp) || εp<=atol
                verbose && println("M-scale order is set to 1 for this iteration")
            else
                ε = exp(ε*log(ε)/εp)
            end
        end
        
        σnp1 = σn * ε
        verbose && println("M-scale update: $(ε) : $(σn) ->  $(σnp1)")

        σn = σnp1
        if abs(ε-1) < rtol
            verbose && println("M-scale converged after $(n) steps.")
            converged = true
            break
        end
    end

    if !approx
        converged || @warn "the M-scale did not converge, consider increasing the maximum number
                of iterations nmax=$(nmax) or starting with a better initial value σ0=$(σ0). Return the current estimate: $(σn)"
    end
    return σn
end
function scale_estimate(est::E, res::AbstractArray{T};
        bound::Real=0.5, σ0::Real=1, wts=[], kwargs...) where {T<:Real, E<:Estimator}
    scale_estimate(res; bound=float(bound), σ0=float(σ0), wts=float(wts), kwargs...)
end

    
function old_scale_estimate(est::Union{E, GeneralQuantileEstimator{E}}, res::AbstractArray{T};
            bound::T=0.5, σ0::T=1.0, wts::AbstractArray{T}=T[], use_reciprocal::Bool=true,
            verbose::Bool=true, method::Symbol=:approx, nmax::Int=30, factor::Real=1.2,
            rtol::Real=1e-4) where {T<:AbstractFloat, E<:BoundedEstimator}


    if method == :roots
        # Use Roots.jl to find the solution to the non-linear equation
        σest(x) = estimator_chi(est, x) - bound
        rσest(x) = estimator_rchider(est, x)
    
        # f and Df should by anonymous otherwise julia complains
        σ = if isempty(wts)
            if use_reciprocal
                f = s -> sum(σest.(res .* s))
#                Df = s -> 1/s*sum(rσest.(res .* s))
#                1/find_zero((f, Df), 1/σ0, Newton())
                1/find_zero(f, 1/σ0, Order1())
            else
                f = s -> sum(σest.(res ./ s))
#                Df = s -> -1/s*sum(rσest.(res ./ s))
#                find_zero((f, Df), σ0, Newton())
                find_zero(f, σ0, Order1())
            end
        else
            if use_reciprocal
                f = s -> sum(r.wts .* σest.(res .* s))
#                Df = s -> 1/s*sum(r.wts .* rσest.(res .* s))
#                1/find_zero((f, Df), 1/σ0, Newton())
                1/find_zero(f, 1/σ0, Order1())
            else
                f = s -> sum(r.wts .* σest.(res ./ s))
#                Df = s -> -1/s*sum(r.wts .* rσest.(res ./ s))
#                find_zero((f, Df), σ0, Newton())
                find_zero(f, σ0, Order1())
            end
        end
        if σ <= 0
            throw(ConvergenceFailed("the resulting scale is non-positive"))
        end
        return σ
    elseif method == :newton
        ## compile f(s) and and -s*Df(s)
        f, Df = if (length(res)==length(wts))
            _fDf_scale_estimate(est, res, wts; bound=bound)
        else
            _fDf_scale_estimate(est, res; bound=bound)
        end

        # In case the scale becomes too small, use a lower bound
        minσ = minimum(abs(r) for r in res if r != 0)

        converged = false
        sn = σ0
        verbose && println("Initial M-scale estimate: $(sn)")

        
        sn = mad(res, normalize=true)

        for n in 1:nmax
            fn = f(sn)
            Dfn = Df(sn)
            if Dfn <= 0
                verbose && println("M-scale set to its minimum value: s_n = $(minσ*factor)")
                Dfn = Df(minσ*factor)
            end
            @assert Dfn > 0

            # Newton update step
            ε = fn/Dfn
            
            snp1 = sn*(1 + ε)
            verbose && println("M-scale update: $(ε)  ->  $(snp1)")
            
            # Ensure that the scale stays positive
            if snp1 <= 0
                snp1 = minσ
                verbose && println("scale cannot be non-positive, set to minimum value: $(snp1)")
            end

            sn = snp1
            if abs(ε) < rtol
                verbose && println("M-scale converged after $(n) steps.")
                converged = true
                break
            end
        end
        converged || @warn "the M-scale did not converge, consider increasing the maximum number
                of iterations nmax=$(nmax) or starting with a better initial value σ0=$(σ0). Return the current estimate: $(sn)"
        return sn
    else
        error("only :approx, :roots and :newton methods are allowed to compute the M-scale estimate")
    end
end


function _fDf_scale_estimate(est::Union{E, GeneralQuantileEstimator{E}}, res::AbstractArray{T};
            bound::T=0.5) where {T<:AbstractFloat, E<:BoundedEstimator}
    
        σest(x) = estimator_chi(est, x) - bound
        rσest(x) = estimator_rchider(est, x)

        f(s) = sum(σest.(res ./ s))
        Df(s) = sum(rσest.(res ./ s)) # *(-1/s)

        return f, Df
end

function _fDf_scale_estimate(est::Union{E, GeneralQuantileEstimator{E}}, res::AbstractArray{T},
            wts::AbstractArray{T}; bound::T=0.5) where {T<:AbstractFloat, E<:BoundedEstimator}
    
        σest(x) = estimator_chi(est, x) - bound
        rσest(x) = estimator_rchider(est, x)

        f(s) = sum(wts .* σest.(res ./ s))
        Df(s) = sum(wts .* rσest.(res ./ s)) # *(-1/s)
        return f, Df
end

######
###   τ-Estimators
######

mutable struct TauEstimator{E<:BoundedEstimator} <: Estimator
    est1::E
    est2::E
    w::Float64
    
    TauEstimator{E}(est1::E, est2::E, w::Real) where {E<:BoundedEstimator} = new(est1, est2, float(w))
    TauEstimator{E}(est1::E, est2::E) where {E<:BoundedEstimator} = new(est1, est2, 0.0)
end
TauEstimator(est1::E, est2::E, w::Real) where {E<:BoundedEstimator} = TauEstimator{E}(est1, est2, w)
TauEstimator(est1::E, est2::E) where {E<:BoundedEstimator} = TauEstimator{E}(est1, est2)

function TauEstimator(::Type{E}) where E<:BoundedEstimator
    est1 = E(estimator_high_breakdown_point_constant(E))
    est2 = E(estimator_tau_efficient_constant(E))
    TauEstimator(est1, est2)
end

function TauEstimator(::Type{E}) where E<:Estimator
    throw(TypeError(:TauEstimator, "RobustModels", BoundedEstimator, E))
end

"""
    TauEstimator(est::M; fallback::Type{BoundedEstimator}=TukeyEstimator, force=false)
Return a τ-Estimator based on an M-Estimator. If the M-Estimator is not bounded,
return a τ-Estimator of the same kind as the fallback.
If force is true, the τ-Estimator will be of the fallback kind.
"""
function TauEstimator(est::SimpleEstimator; fallback::Type{E}=TukeyEstimator,
            force::Bool=false)::SimpleEstimator where {E<:BoundedEstimator}
    ## Change the estimator to the same estimator with a tuning constant that gives a low breakdown point
    typ = typeof(est)
    if force || !isbounded(est)
        typ = fallback
    end
    TauEstimator(typ)
end


function show(io::IO, obj::TauEstimator)
    print(io, "τ-Estimator($(obj.est1), $(obj.est2))")
end

function tau_efficiency_tuning_constant(::Type{M}; eff::Real=0.95, c0::Real=1.0) where M<:BoundedEstimator
    est1 = M(estimator_high_breakdown_point_constant(M))
    w1 = quadgk(x->x * estimator_psi(est1, x)*2*exp(-x^2/2)/√(2π), 0, Inf)[1]

    function τest(c)
        est2 = M(c)
        w2 = quadgk(x->(2*estimator_rho(est2, x)*(est2.c)^2 - x * estimator_psi(est2, x))*
                        2*exp(-x^2/2)/√(2π), 0, Inf)[1]
        TauEstimator{M}(est1, est2, w2 / w1)
    end

    psi(x, c)  = estimator_psi(τest(c), x)
    psip(x, c) = estimator_psider(τest(c), x)

    I1(c) = quadgk(x->(psi(x, c))^2*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    I2(c) = quadgk(x->psip(x, c)*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    fun_eff(c) = (I2(c))^2/I1(c)
    copt = find_zero(c->fun_eff(c) - eff, c0, Order1())
end


estimator_tau_efficient_constant(::Type{GemanEstimator})      = 5.632
estimator_tau_efficient_constant(::Type{WelschEstimator})     = 4.043
estimator_tau_efficient_constant(::Type{TukeyEstimator})      = 6.040
estimator_tau_efficient_constant(::Type{YohaiZamarEstimator}) = 3.270


"""
    update_weight!(E::TauEstimator, res::AbstractArray{T}; wts::AbstractArray{T}=T[])
Update the weight between the two estimators of a τ-estimator using the scaled residual.
"""
function update_weight!(E::TauEstimator, res::AbstractArray{T};
            wts::AbstractArray{T}=T[]) where {T<:AbstractFloat}
    w = if length(wts) == length(res)
        w2 = sum(wts .* (2 .* estimator_rho.(E.est2, res) .* (E.est2.c)^2 .-
                                    res .* estimator_psi.(E.est2, res)))
        w1 = sum(wts .* res .* estimator_psi.(E.est1, res))
        w2 / w1
    else
        w2 = sum(2*estimator_rho(E.est2, r)*(E.est2.c)^2 - r*estimator_psi(E.est2, r) for r in res)
        w1 = sum(r*estimator_psi(E.est1, r) for r in res)
        w2 / w1
    end
    E.w = w
end
update_weight!(E::TauEstimator, res::AbstractArray; wts::AbstractArray=[]) = update_weight!(E, float(res); wts=float(wts))
update_weight!(E::TauEstimator, w::Real) = (E.w = w; )

# Forward all methods to the `est` fields
estimator_rho(   E::TauEstimator, r::Real) = E.w * (E.est1.c)^2 / (E.est2.c)^2 * estimator_rho(E.est1, r) + estimator_rho(E.est2, r)
estimator_psi(   E::TauEstimator, r::Real) = E.w * estimator_psi(E.est1, r) + estimator_psi(E.est2, r)
estimator_psider(E::TauEstimator, r::Real) = E.w * estimator_psider(E.est1, r) + 
                                                        estimator_psider(E.est2, r)
estimator_weight(E::TauEstimator, r::Real) = E.w * estimator_weight(E.est1, r) +
                                                        estimator_weight(E.est2, r)
function estimator_values(E::TauEstimator, r)
    vals1 = estimator_values(E.est1, r)
    vals2 = estimator_values(E.est1, r)
    E.w .* vals1 .+ vals2
end
estimator_chi(E::TauEstimator, r::Real) = estimator_chi(E.est1, r)
estimator_norm(E::TauEstimator, args...) = Inf
isbounded(E::TauEstimator) = true
isconvex( E::TauEstimator) = false


function scale_estimate(est::E, res::AbstractArray{T}; kwargs...) where {T<:AbstractFloat, E<:TauEstimator}
    scale_estimate(est.est1, res; kwargs...)
end

######
## TODO: Create types MEstimator, SEstimator and MMEstimator, like TauEstimator, that holds the useful parameters for the estimators
######


