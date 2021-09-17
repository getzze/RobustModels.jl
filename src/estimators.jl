
using QuadGK: quadgk
using Roots: find_zero, Order1, ConvergenceFailed, Newton

## Threshold to avoid numerical overflow of the weight function of L1Estimator and ArctanEstimator
DELTA = 1e-8    # chosen because it minimizes the error between (1-ATWDELTA)/DELTA and 1/3 for the linear approximation
L1WDELTA = 1/(DELTA)
ATWDELTA = atan(DELTA)/DELTA
#DELTA = 1e-8    # chosen because it minimizes the error between (1-ATWDELTA)/DELTA and 1/3 for the linear approximation
#L1WDELTA = 1/(2*sqrt(DELTA))
#ATWDELTA = atan(sqrt(DELTA))*2*L1WDELTA


#=
TODO: change the API to
rlm(form, data, MEstimator(Tukey))
rlm(X, y, TauEstimator(YohaiZamar); method=:cg)
rlm(X, y, MMEstimator(Tukey(2.1), Tukey(5.0)); σ0=:mad)

=#

"The loss function used for the estimation"
function loss end


"The loss function ρ for the M-estimator."
function rho end

"""
The influence function ψ is the derivative of the loss function for the M-estimator,
multiplied by the square of the tuning constant.
"""
function psi end

"The derivative of ψ, used for asymptotic estimates."
function psider end

"The weights for IRLS, the function ψ divided by r."
function weight end

"The integral of exp(-ρ) used for calculating the full-loglikelihood"
function estimator_norm end

"The limit at ∞ of the loss function. Used for scale estimation of bounded loss."
estimator_bound(::Type{<:LossFunction}) = Inf


"The tuning constant of the loss function, can be optimized to get efficient or robust estimates."
tuning_constant(loss::L) where {L<:LossFunction} = loss.c

"Boolean if the estimator or loss function is convex"
isconvex( f::LossFunction) = false
isconvex( f::ConvexLossFunction) = true

"Boolean if the estimator or loss function is bounded"
isbounded(f::LossFunction) = false
isbounded(f::BoundedLossFunction) = true

"The tuning constant associated to the loss that gives an efficient M-estimator."
estimator_high_breakdown_point_constant(::Type{L}) where {L<:LossFunction} = 1

"The tuning constant associated to the loss that gives a robust (high breakdown point) M-estimator."
estimator_high_efficiency_constant(::Type{L}) where {L<:LossFunction} = 1

# Static creators
"The loss initialized with an efficient tuning constant"
efficient_loss(::Type{L}) where {L<:LossFunction} = L(estimator_high_efficiency_constant(L))

"The loss initialized with a robust (high breakdown point) tuning constant"
robust_loss(   ::Type{L}) where {L<:LossFunction} = L(estimator_high_breakdown_point_constant(L))


rho(l::LossFunction, r) = _rho(l, r/tuning_constant(l))
psi(l::LossFunction, r) = tuning_constant(l)*_psi(l, r/tuning_constant(l))
psider(l::LossFunction, r) = _psider(l, r/tuning_constant(l))
weight(l::LossFunction, r) = _weight(l, r/tuning_constant(l))

"Faster version if you need ρ, ψ and w in the same call"
function values(loss::LossFunction, r::Real)
    c = tuning_constant(loss)
    rr = r/c
    return (_rho(loss, rr), c*_psi(loss, rr), _weight(loss, rr))
end


##
# For reminder and not to get lost with the tuning constant
#   ρ(r) ~ r²/(2c²)
#   ψ = c²ρ' ;  ψ ~ r  ;  _psi ~ r/c
#   w ~ 1
##


"""
The M-estimator norm is computed with:
     +∞                    +∞
Z = ∫  exp(-ρ(r))dr = c . ∫  exp(-ρ_1(r))dr    with ρ_1 the function for c=1
    -∞                    -∞
"""
function estimator_norm(loss::L) where {L<:LossFunction}
    2*quadgk(x->exp(-rho(loss, x)), 0, Inf)[1]
end


"""
The tuning constant c is computed so the efficiency for Normal distributed
residuals is 0.95. The efficiency of the mean estimate μ is defined by:

eff_μ = (E[ψ'])²/E[ψ²]
"""
function efficiency_tuning_constant(::Type{L}; eff::Real=0.95, c0::Real=1.0) where L<:LossFunction
    lpsi(x, c)  = RobustModels.psi(L(c), x)
    lpsip(x, c) = RobustModels.psider(L(c), x)

    I1(c) = quadgk(x->(lpsi(x, c))^2*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    I2(c) = quadgk(x->lpsip(x, c)*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    fun_eff(c) = (I2(c))^2/I1(c)
    copt = find_zero(c->fun_eff(c) - eff, c0, Order1())
end


"""
    mscale_loss(loss::L, x)

The rho-function that is used for M-scale estimation.

For monotone (convex) functions, χ(r) = r.ψ(r).

For bounded functions, χ(r) = ρ(r)/ρ(∞) so χ(∞) = 1.
"""
mscale_loss(loss::L, x) where L<:LossFunction = x*RobustModels.psi(loss, x)
mscale_loss(loss::L, x) where L<:BoundedLossFunction = RobustModels.rho(loss, x)/estimator_bound(L)


"""
The M-estimate of scale is computed by solving:

```math
\\dfrac{1}{n} \\sum_i \\chi\\left( \\dfrac{r_i}{\\hat{\\sigma}}\\right) = \\delta
```

For monotone (convex) functions, χ(r) = r.ψ(r) and δ is defined as E[χ(r)] = δ for the Normal distribution N(0,1)
For bounded functions, χ(r) = ρ(r)/ρ(∞) with χ(∞) = 1 and δ = E[χ]/χ(∞) with expectation w.r.t. Normal density.

The tuning constant c corresponding to a high breakdown point (0.5)
is such that δ = 1/2, from  1/n Σ χ(r/ŝ) = δ
"""
function breakdown_point_tuning_constant(::Type{L}; bp::Real=1/2, c0::Real=1.0) where L<:LossFunction
    (0 < bp <= 1/2) || error("breakdown-point should be between 0 and 1/2")

    I(c) = quadgk(x->mscale_loss(L(c), x)*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    copt = find_zero(c->I(c) - bp, c0, Order1())
end




########
###     Loss functions
########

"The (convex) L2 loss function is that of the standard least squares problem."
struct L2Loss <: ConvexLossFunction; end
L2Loss(c) = L2Loss()
rho(   ::L2Loss, r::Real) = r^2 / 2
psi(   ::L2Loss, r::Real) = r
psider(::L2Loss, r::Real) = oftype(r, 1)
weight(::L2Loss, r::Real) = oftype(r, 1)
values(::L2Loss, r::Real) = (r^2/2, r, oftype(r, 1))
estimator_norm(::L2Loss) = √(2*π)
tuning_constant(::L2Loss) = 1



"""
The standard L1 loss function takes the absolute value of the residual, and is
convex but non-smooth. It is not a real L1 loss but a Huber loss
with very small tuning constant.
"""
struct L1Loss <: ConvexLossFunction; end
L1Loss(c) = L1Loss()
rho(   ::L1Loss, r::Real) = abs(r)
psi(   ::L1Loss, r::Real) = sign(r)
psider(::L1Loss, r::Real) = if (abs(r)<DELTA); oftype(r, 1) else oftype(r, 0) end
weight(::L1Loss, r::Real) = if (abs(r)<DELTA); L1WDELTA else 1/abs(r) end
function values(est::L1Loss, r::Real)
    rr = abs(r)
    return (rr, sign(r), (if (rr<DELTA); L1WDELTA else 1/rr end) )
end
estimator_norm(::L1Loss) = 2
tuning_constant(::L1Loss) = 1


"""
The convex Huber loss function switches from between quadratic and linear cost/loss
function at a certain cutoff.
"""
struct HuberLoss <: ConvexLossFunction
    c::Float64

    HuberLoss(c::Real) = new(c)
    HuberLoss() = new(estimator_high_efficiency_constant(HuberLoss))
end

_rho(   l::HuberLoss, r::Real) = if (abs(r)<=1); r^2/2 else (abs(r) - 1/2) end
_psi(   l::HuberLoss, r::Real) = if (abs(r)<=1); r             else sign(r) end
_psider(l::HuberLoss, r::Real) = if (abs(r)<=1); oftype(r, 1)  else oftype(r, 0) end
_weight(l::HuberLoss, r::Real) = if (abs(r)<=1); oftype(r, 1)  else 1/abs(r) end
function values(l::HuberLoss, r::Real)
    rr = abs(r)
    if rr <= l.c
        return ((rr/l.c)^2/2 , r , oftype(r, 1) )
    else
        return (rr/l.c - 1/2 , l.c*sign(r) , l.c/rr )
    end
end
estimator_norm(l::HuberLoss) = l.c * 2.92431
estimator_high_efficiency_constant(::Type{HuberLoss}) = 1.345
estimator_high_breakdown_point_constant(::Type{HuberLoss}) = 0.6745



"""
The convex L1-L2 loss interpolates smoothly between L2 behaviour for small
residuals and L1 for outliers.
"""
struct L1L2Loss <: ConvexLossFunction
    c::Float64

    L1L2Loss(c::Real) = new(c)
    L1L2Loss() = new(estimator_high_efficiency_constant(L1L2Loss))
end
_rho(   l::L1L2Loss, r::Real) = (sqrt(1 + r^2) - 1)
_psi(   l::L1L2Loss, r::Real) = r / sqrt(1 + r^2)
_psider(l::L1L2Loss, r::Real) = 1 / (1 + r^2)^(3/2)
_weight(l::L1L2Loss, r::Real) = 1 / sqrt(1 + r^2)
function values(l::L1L2Loss, r::Real)
    sqr = sqrt(1 + (r/l.c)^2)
    return ((sqr - 1), r/sqr, 1/sqr)
end
estimator_norm(l::L1L2Loss) = l.c * 3.2723
estimator_high_efficiency_constant(::Type{L1L2Loss}) = 1.287
estimator_high_breakdown_point_constant(::Type{L1L2Loss}) = 0.8252


"""
The (convex) "fair" loss switches from between quadratic and linear
cost/loss function at a certain cutoff, and is C3 but non-analytic.
"""
struct FairLoss <: ConvexLossFunction
    c::Float64

    FairLoss(c::Real) = new(c)
    FairLoss() = new(estimator_high_efficiency_constant(FairLoss))
end
_rho(   l::FairLoss, r::Real) = abs(r) - log(1 + abs(r))
_psi(   l::FairLoss, r::Real) = r / (1 + abs(r))
_psider(l::FairLoss, r::Real) = 1 / (1 + abs(r))^2
_weight(l::FairLoss, r::Real) = 1 / (1 + abs(r))
function values(l::FairLoss, r::Real)
    ir = 1/(1 + abs(r/l.c))
    return (abs(r)/l.c + log(ir), r*ir, ir)
end
estimator_norm(l::FairLoss) = l.c * 4
estimator_high_efficiency_constant(::Type{FairLoss}) = 1.400
estimator_high_breakdown_point_constant(::Type{FairLoss}) = 1.4503

"""
The convex Log-Cosh loss function
log(cosh(r))
"""
struct LogcoshLoss <: ConvexLossFunction
    c::Float64

    LogcoshLoss(c::Real) = new(c)
    LogcoshLoss() = new(estimator_high_efficiency_constant(LogcoshLoss))
end
_rho(   l::LogcoshLoss, r::Real) = log(cosh(r))
_psi(   l::LogcoshLoss, r::Real) = tanh(r)
_psider(l::LogcoshLoss, r::Real) = 1 / (cosh(r))^2
_weight(l::LogcoshLoss, r::Real) = if (abs(r)<DELTA); (1 - (r)^2/3) else tanh(r) / r end
function values(l::LogcoshLoss, r::Real)
    tr = l.c * tanh(r/l.c)
    rr = abs(r/l.c)
    return ( log(cosh(rr)), tr, (if (rr<DELTA); (1 - rr^2/3) else tr/r end) )
end
estimator_norm(l::LogcoshLoss) = l.c * π
estimator_high_efficiency_constant(::Type{LogcoshLoss}) = 1.2047
estimator_high_breakdown_point_constant(::Type{LogcoshLoss}) = 0.7479

"""
The convex Arctan loss function
r * arctan(r) - 1/2*log(1 + r^2)
"""
struct ArctanLoss <: ConvexLossFunction
    c::Float64

    ArctanLoss(c::Real) = new(c)
    ArctanLoss() = new(estimator_high_efficiency_constant(ArctanLoss))
end
_rho(   l::ArctanLoss, r::Real) =  r * atan(r) - 1/2*log(1 + r^2)
_psi(   l::ArctanLoss, r::Real) = atan(r)
_psider(l::ArctanLoss, r::Real) = 1 / (1 + r^2)
_weight(l::ArctanLoss, r::Real) = if (abs(r)<DELTA); (1 - r^2/3) else atan(r) / r end
function values(l::ArctanLoss, r::Real)
    ar = atan(r/l.c)
    rr = abs(r/l.c)
    return ( r*ar/l.c - 1/2*log(1 + rr^2), l.c*ar, (if (rr<DELTA); (1 - rr^2/3) else l.c*ar/r end) )
end
estimator_norm(l::ArctanLoss) = l.c * 2.98151
estimator_high_efficiency_constant(::Type{ArctanLoss}) = 0.919
estimator_high_breakdown_point_constant(::Type{ArctanLoss}) = 0.612

"""
The non-convex Cauchy loss function switches from between quadratic behaviour to
logarithmic tails. This rejects outliers but may result in multiple minima.
For scale estimate, r.ψ(r) is used as a loss, which is the same as for Geman loss.
"""
struct CauchyLoss <: LossFunction
    c::Float64

    CauchyLoss(c::Real) = new(c)
    CauchyLoss() = new(estimator_high_efficiency_constant(CauchyLoss))
end
_rho(   l::CauchyLoss, r::Real) = log(1 + r^2) # * 1/2  # remove factor 1/2 so the loss has a norm
_psi(   l::CauchyLoss, r::Real) = r / (1 + r^2)
_psider(l::CauchyLoss, r::Real) = (1 - r^2) / (1 + r^2)^2
_weight(l::CauchyLoss, r::Real) = 1 / (1 + r^2)
function values(l::CauchyLoss, r::Real)
    ir = 1/(1 + (r/l.c)^2)
    return ( - log(ir), r*ir, ir )
end
estimator_norm(l::CauchyLoss) = l.c * π
isconvex( ::CauchyLoss) = false
isbounded(::CauchyLoss) = false

estimator_high_efficiency_constant(::Type{CauchyLoss}) = 2.385
estimator_high_breakdown_point_constant(::Type{CauchyLoss}) = 1.468

"""
The non-convex Geman-McClure for strong supression of outliers and does not guarantee a unique solution.
For the S-Estimator, it is equivalent to the Cauchy loss.
"""
struct GemanLoss <: BoundedLossFunction
    c::Float64

    GemanLoss(c::Real) = new(c)
    GemanLoss() = new(estimator_high_efficiency_constant(GemanLoss))
end
_rho(   l::GemanLoss, r::Real) = 1/2 * r^2 / (1 + r^2)
_psi(   l::GemanLoss, r::Real) = r / (1 + r^2)^2
_psider(l::GemanLoss, r::Real) = (1 - 3*r^2) / (1 + r^2)^3
_weight(l::GemanLoss, r::Real) = 1 / (1 + r^2)^2
function values(l::GemanLoss, r::Real)
    rr2 = (r/l.c)^2
    ir = 1/(1 + rr2)
    return ( 1/2 * rr2 * ir, r*ir^2, ir^2 )
end
estimator_norm(::GemanLoss) = Inf
isconvex( ::GemanLoss) = false
isbounded(::GemanLoss) = true

estimator_bound(::Type{GemanLoss}) = 1/2
estimator_high_efficiency_constant(::Type{GemanLoss}) = 3.787
estimator_high_breakdown_point_constant( ::Type{GemanLoss}) = 0.61200



"""
The non-convex Welsch for strong supression of ourliers and does not guarantee a unique solution
"""
struct WelschLoss <: BoundedLossFunction
    c::Float64

    WelschLoss(c::Real) = new(c)
    WelschLoss() = new(estimator_high_efficiency_constant(WelschLoss))
end
_rho(   l::WelschLoss, r::Real) = -1/2 * Base.expm1(-r^2)
_psi(   l::WelschLoss, r::Real) = r * exp(-r^2)
_psider(l::WelschLoss, r::Real) = (1 - 2*r^2)*exp(-r^2)
_weight(l::WelschLoss, r::Real) = exp(-r^2)
function values(l::WelschLoss, r::Real)
    rr2 = (r/l.c)^2
    er = exp(-rr2)
    return ( -1/2 * Base.expm1(-rr2), r*er, er )
end
estimator_norm(::WelschLoss) = Inf
isconvex( ::WelschLoss) = false
isbounded(::WelschLoss) = true

estimator_bound(::Type{WelschLoss}) = 1/2
estimator_high_efficiency_constant(::Type{WelschLoss}) = 2.985
estimator_high_breakdown_point_constant( ::Type{WelschLoss}) = 0.8165


"""
The non-convex Tukey biweight estimator which completely suppresses the outliers,
and does not guaranty a unique solution
"""
struct TukeyLoss <: BoundedLossFunction
    c::Float64

    TukeyLoss(c::Real) = new(c)
    TukeyLoss() = new(estimator_high_efficiency_constant(TukeyLoss))
end
_rho(   l::TukeyLoss, r::Real) = if (abs(r)<=1); 1/6 * (1 - ( 1 - r^2 )^3) else 1/6  end
_psi(   l::TukeyLoss, r::Real) = if (abs(r)<=1); r*(1 - r^2)^2             else oftype(r, 0) end
_psider(l::TukeyLoss, r::Real) = if (abs(r)<=1); 1 - 6*r^2 + 5*r^4         else oftype(r, 0) end
_weight(l::TukeyLoss, r::Real) = if (abs(r)<=1); (1 - r^2)^2               else oftype(r, 0) end
function values(l::TukeyLoss, r::Real)
    pr = (abs(r)<=l.c) * (1 - (r/l.c)^2)
    return ( 1/6*(1 - pr^3), r*pr^2, pr^2 )
end
estimator_norm(::TukeyLoss) = Inf
isconvex( ::TukeyLoss) = false
isbounded(::TukeyLoss) = true

estimator_bound(::Type{TukeyLoss}) = 1/6
estimator_high_efficiency_constant(::Type{TukeyLoss}) = 4.685
estimator_high_breakdown_point_constant( ::Type{TukeyLoss}) = 1.5476


"""
The non-convex (and bounded) optimal Yohai-Zamar loss function that
minimizes the estimator bias. It was originally introduced in 
Optimal locally robust M-estimates of regression (1997) by Yohai and Zamar
with a slightly different formula.
"""
struct YohaiZamarLoss <: BoundedLossFunction
    c::Float64

    YohaiZamarLoss(c::Real) = new(c)
    YohaiZamarLoss() = new(estimator_high_efficiency_constant(YohaiZamarLoss))
end
function rho(l::YohaiZamarLoss, r::Real)
    z = (r/l.c)^2
    if (z<=4/9)
        1.3846 * z
    elseif (z<=1)
        min(1.0, 0.5514 - 2.6917*z + 10.7668*z^2 - 11.6640*z^3 + 4.0375*z^4)
    else
        oftype(r, 1)
    end
end
function psi(l::YohaiZamarLoss, r::Real)
    z = (r/l.c)^2
    if (z<=4/9)
        2.7692 * r
    elseif (z<=1)
        r*max(0, -5.3834 + 43.0672*z - 69.984*z^2 + 32.3*z^3)
    else
        oftype(r, 0)
    end
end
function psider(l::YohaiZamarLoss, r::Real)
    z = (r/l.c)^2
    if (z<=4/9)
        2.7692
    elseif (z<=0.997284)  # from the root of ψ expression
        -5.3834 + 129.2016*z - 349.92*z^2 + 226.1*z^3
    else
        oftype(r, 0)
    end
end
function weight(l::YohaiZamarLoss, r::Real)
    z = (r/l.c)^2
    if (z<=4/9)
        2.7692
    elseif (z<=1)
        max(0, -5.3834 + 43.0672*z - 69.984*z^2 + 32.3*z^3)
    else
        oftype(r, 0)
    end
end
function values(l::YohaiZamarLoss, r::Real)
    z = (r/l.c)^2
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
estimator_norm(::YohaiZamarLoss) = Inf
isconvex( ::YohaiZamarLoss) = false
isbounded(::YohaiZamarLoss) = true

estimator_bound(::Type{YohaiZamarLoss}) = 1
estimator_high_efficiency_constant(::Type{YohaiZamarLoss}) = 3.1806
estimator_high_breakdown_point_constant( ::Type{YohaiZamarLoss}) = 1.2139

######
###   Convex sum of loss functions
######
struct CompositeLossFunction{L1<:LossFunction, L2<:LossFunction} <: LossFunction
    α1::Float64
    loss1::L1
    α2::Float64
    loss2::L2

end
function CompositeLossFunction(loss1::LossFunction, loss2::LossFunction, α1::Real=1.0, α2::Real=1.0)
    α1 >= 0 || throw(DomainError(α1, "coefficients of CompositeLossFunction should be non-negative."))
    α2 >= 0 || throw(DomainError(α2, "coefficients of CompositeLossFunction should be non-negative."))
    
    return CompositeLossFunction{typeof(loss1), typeof(loss2)}(float(α1), loss1, float(α2), loss2)
end

function show(io::IO, e::CompositeLossFunction)
    print(io, "CompositeLossFunction $(round(e.α1; digits=2)) . $(e.loss1) + $(round(e.α2; digits=2)) . $(e.loss2)")
end

Base.first(e::CompositeLossFunction) = e.loss1
Base.last(e::CompositeLossFunction) = e.loss2

isbounded(e::CompositeLossFunction) = isbounded(e.loss1) && isbounded(e.loss2)
isconvex(e::CompositeLossFunction)  = isconvex(e.loss1)  && isconvex(e.loss2)

estimator_norm(e::CompositeLossFunction) = e.α1 * estimator_norm(e.loss1) + e.α2 * estimator_norm(e.loss2)
rho(   e::CompositeLossFunction, r::Real) = e.α1 * rho(e.loss1) * (tuning_constant(E.loss1))^2 + e.α2 * rho(e.loss2) * (tuning_constant(E.loss2))^2
psi(   e::CompositeLossFunction, r::Real) = e.α1 * psi(e.loss1) + e.α2 * psi(e.loss2)
psider(e::CompositeLossFunction, r::Real) = e.α1 * psider(e.loss1) + e.α2 * psider(e.loss2)
weight(e::CompositeLossFunction, r::Real) = e.α1 * weight(e.loss1) + e.α2 * weight(e.loss2)
values(e::CompositeLossFunction, r::Real) = @.(e.α1 * values(e.loss1) + e.α2 * values(e.loss2))


######
###   Scale estimation
######

"""
    scale_estimate(loss, res; σ0=1.0, wts=[], verbose=false,
                             order=1, approx=false, nmax=30, 
                             rtol=1e-4, atol=0.1)

Compute the M-scale estimate from the loss function.
If the loss is bounded, ρ is used as the function χ in the sum,
otherwise r.ψ(r) is used if the loss is not bounded, to coincide with
the Maximum Likelihood Estimator.
Also, for bounded estimator, because f(s) = 1/(nδ) Σ ρ(ri/s) is decreasing
the iteration step is not using the weights but is multiplicative.
"""
function scale_estimate(l::L, res::AbstractArray{T};
            verbose::Bool=false, bound::Union{Nothing, T}=nothing, 
            σ0::T=1.0, wts::AbstractArray{T}=T[], 
            order::Int=1, approx::Bool=false, nmax::Int=30, 
            rtol::Real=1e-4, atol::Real=0.1) where {L<:LossFunction, T<:AbstractFloat}

    # Compute δ such that E[ρ] = δ for a Normal N(0, 1)
    if isnothing(bound)
        bound = quadgk(x->mscale_loss(l, x)*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    end
    
    Nz = sum(iszero, res)
    if Nz > length(res)*(1-bound)
        # The M-scale cannot be estimated because too many residuals are zeros.
        verbose && println("there are too many zero residuals for M-scale estimation: #(r=0) > n*(1-b), $(Nz) > $(length(res)*(1-bound))")
        throw(ConvergenceFailed("the M-scale cannot be estimated because too many residuals are zeros: $(res)"))
    end

    # Approximate the solution with `nmax` iterations
    σn = σ0
    converged = false
    verbose && println("Initial M-scale estimate: $(σn)")
    for n in 1:nmax
        ## For non-bounded loss, we suppose that χ = r.ψ,
        ## therefore the weight ( χ/r² ) is ψ/r which is w.
        ww = weight.(l, res./σn)
        if length(wts) == length(res)
            ww .*= wts
        end
        σnp1 = sqrt( mean( res ./ σn , weights(ww) ) / bound )

        verbose && println("M-scale update: $(σn) ->  $(σnp1)")

        ε = σnp1/σn
        σn = σnp1
        if abs(ε-1) < rtol
            verbose && println("M-scale converged after $(n) steps.")
            converged = true
            break
        end
    end

    if !approx
        converged || @warn "the M-scale did not converge, consider increasing the maximum" *
                           " number of iterations nmax=$(nmax) or starting with a better" *
                           " initial value σ0=$(σ0). Return the current estimate: $(σn)"
    end

    return σn
end

function scale_estimate(l::L, res::AbstractArray{T};
            verbose::Bool=false, bound::Union{Nothing, Real}=0.5,
            σ0::Real=1.0, wts::AbstractArray{T}=T[], 
            order::Int=1, approx::Bool=false, nmax::Int=30, 
            rtol::Real=1e-4, atol::Real=0.1) where {L<:BoundedLossFunction,T<:AbstractFloat}
    
    # Compute δ such that E[ρ] = δ for a Normal N(0, 1)
    if isnothing(bound)
        bound = quadgk(x->mscale_loss(l, x)*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    end

    Nz = sum(iszero, res)
    if Nz > length(res)*(1-bound)
        # The M-scale cannot be estimated because too many residuals are zeros.
        verbose && println("there are too many zero residuals for M-scale estimation: #(r=0) > n*(1-b), $(Nz) > $(length(res)*(1-bound))")
        throw(ConvergenceFailed("the M-scale cannot be estimated because too many residuals are zeros: $(res)"))
    end

    # Approximate the solution with `nmax` iterations
    if !approx && order>=2
        bb = tuning_constant(l)^2 * estimator_bound(L) * bound
    end
    σn = σ0
    converged = false
    verbose && println("Initial M-scale estimate: $(σn)")
    ε = 0
    for n in 1:nmax
        rr = res / σn
        ε = if length(wts) == length(res)
            mean( mscale_loss.(l, rr), weights(wts) ) / bound
        else
            mean(x->mscale_loss(l, x), rr) / bound
        end
        verbose && println("M-scale 1st order update: $(ε)")
        
        ## Implemented, but it gives worst results than 1st order...
        ## Uses Newton's method to find the root of
        ## f(σ) = log( 1/(nb) Σ ρ(r/σ) ) = 0
        if !approx && order>=2
            εp = if length(wts) == length(res)
                mean( rr .* psi.(l, rr), weights(wts) ) / bb
            else
                mean(x-> x * psi(l, x), rr) / bb
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
        converged || @warn """the M-scale did not converge ε=$(round(abs(ε-1); digits=5)), consider increasing the maximum
                              number of iterations nmax=$(nmax) or starting with a better
                              initial value σ0=$(σ0). Return the current estimate: $(σn)"""
    end
    return σn
end


######
###   M-Estimators
######
"""
    MEstimator{L<:LossFunction} <: AbstractEstimator
    
M-estimator for a given loss function.

The M-estimator is obtained by minimizing the loss function:

```math
\\hat{\\mathbf{\\beta}} = \\underset{\\mathbf{\\beta}}{\\textrm{argmin}} \\sum_{i=1}^n \\rho\\left(\\dfrac{r_i}{\\hat{\\sigma}}\\right)
```

with the residuals  ``\\mathbf{r} = \\mathbf{y} - \\mathbf{X} \\mathbf{\\beta}`` ,
and a robust scale estimate ``\\hat{\\sigma}``.


# Fields
- `loss`: the [`LossFunction`](@ref) used for the robust estimation.

"""
struct MEstimator{L<:LossFunction} <: AbstractEstimator
    loss::L
end
MEstimator{L}() where L<:LossFunction = MEstimator(efficient_loss(L))
MEstimator(::Type{L}) where L<:LossFunction = MEstimator(efficient_loss(L))

loss(e::MEstimator) = e.loss

function show(io::IO, obj::MEstimator)
    print(io, "M-Estimator($(obj.loss))")
end

# Forward all methods to the `loss` field
rho(   e::MEstimator, r::Real) = rho(   e.loss, r)
psi(   e::MEstimator, r::Real) = psi(   e.loss, r)
psider(e::MEstimator, r::Real) = psider(e.loss, r)
weight(e::MEstimator, r::Real) = weight(e.loss, r)
values(e::MEstimator, r::Real) = values(e.loss, r)
estimator_norm(e::MEstimator, args...) = estimator_norm(e.loss, args...)
estimator_bound(e::MEstimator) = estimator_bound(typeof(e.loss))

isbounded(e::MEstimator) = isbounded(e.loss)
isconvex( e::MEstimator) = isconvex( e.loss)

scale_estimate(est::E, res; kwargs...) where {E<:MEstimator} = scale_estimate(est.loss, res; kwargs...)

"`L1Estimator` is a shorthand name for `MEstimator{L1Loss}`. Using exact QuantileRegression should be prefered."
const L1Estimator = MEstimator{L1Loss}

"`L2Estimator` is a shorthand name for `MEstimator{L2Loss}`, the non-robust OLS."
const L2Estimator = MEstimator{L2Loss}



######
###   S-Estimators
######

"""
    SEstimator{L<:BoundedLossFunction} <: AbstractEstimator
    
S-estimator for a given bounded loss function.

The S-estimator is obtained by minimizing the scale estimate:

```math
\\hat{\\mathbf{\\beta}} = \\underset{\\mathbf{\\beta}}{\\textrm{argmin }} \\hat{\\sigma}^2
```

where the robust scale estimate ``\\hat{\\sigma}}`` is solution of:


```math
\\dfrac{1}{n} \\sum_{i=1}^n \\rho\\left(\\dfrac{r_i}{\\hat{\\sigma}}\\right) = \\delta
```

with the residuals  ``\\mathbf{r} = \\mathbf{y} - \\mathbf{X} \\mathbf{\\beta}`` ,
``\\rho`` is a bounded loss function with  ``\\underset{r \\to \\infty}{\\lim} \\rho(r) = 1`` and
``\\delta`` is the finite breakdown point, usually 0.5.


# Fields
- `loss`: the [`LossFunction`](@ref) used for the robust estimation.

"""
struct SEstimator{L<:BoundedLossFunction} <: AbstractEstimator
    loss::L
end
SEstimator{L}() where L<:BoundedLossFunction = SEstimator(robust_loss(L))
SEstimator(::Type{L}) where L<:BoundedLossFunction = SEstimator(robust_loss(L))

loss(e::SEstimator) = e.loss

function show(io::IO, obj::SEstimator)
    print(io, "S-Estimator($(obj.loss))")
end

# Forward all methods to the `loss` field
rho(   e::SEstimator, r::Real) = rho(   e.loss, r)
psi(   e::SEstimator, r::Real) = psi(   e.loss, r)
psider(e::SEstimator, r::Real) = psider(e.loss, r)
weight(e::SEstimator, r::Real) = weight(e.loss, r)
values(e::SEstimator, r::Real) = values(e.loss, r)
estimator_norm(e::SEstimator, args...) = Inf
estimator_bound(e::SEstimator) = estimator_bound(typeof(e.loss))
isbounded(e::SEstimator) = true
isconvex( e::SEstimator) = false

scale_estimate(est::E, res; kwargs...) where {E<:SEstimator} = scale_estimate(est.loss, res; kwargs...)




######
###   MM-Estimators
######

"""
    MMEstimator{L1<:BoundedLossFunction, L2<:LossFunction} <: AbstractEstimator
    
MM-estimator for the given loss functions.

The MM-estimator is obtained using a two-step process:

1. compute a robust scale estimate with a high breakdown point using a S-estimate and the loss function `L1`.
2. compute an efficient estimate using a M-estimate with the loss function `L2`.


# Fields
- `loss1`: the [`BoundedLossFunction`](@ref) used for the high breakdown point S-estimation.
- `loss2`: the [`LossFunction`](@ref) used for the efficient M-estimation.
- `scaleest`: boolean specifying the if the estimation is in the S-estimation step (`true`)
or the M-estimation step (`false`).

"""
mutable struct MMEstimator{L1<:BoundedLossFunction, L2<:LossFunction} <: AbstractEstimator
    "high breakdown point loss function"
    loss1::L1

    "high efficiency loss function"
    loss2::L2

    "S-Estimator phase indicator (or M-Estimator phase)"
    scaleest::Bool
    
    MMEstimator{L1, L2}(loss1::L1, loss2::L2, scaleest::Bool=true) where {L1<:BoundedLossFunction, L2<:LossFunction} = new(loss1, loss2, scaleest)
end
MMEstimator(loss1::L1, loss2::L2, scaleest::Bool) where {L1<:BoundedLossFunction, L2<:LossFunction} = MMEstimator{L1, L2}(loss1, loss2, scaleest)
MMEstimator(loss1::L1, loss2::L2) where {L1<:BoundedLossFunction, L2<:LossFunction} = MMEstimator{L1, L2}(loss1, loss2, true)
MMEstimator(::Type{L1}, ::Type{L2}) where {L1<:BoundedLossFunction, L2<:LossFunction} = MMEstimator(robust_loss(L1), efficient_loss(L2))
MMEstimator{L}() where L<:BoundedLossFunction = MMEstimator(robust_loss(L), efficient_loss(L))
MMEstimator(::Type{L}) where L<:BoundedLossFunction = MMEstimator{L}()

loss(e::MMEstimator) = if e.scaleest; e.loss1 else e.loss2 end

"MEstimator, set to S-Estimation phase"
set_SEstimator(e::MMEstimator) = (e.scaleest=true; e)

"MEstimator, set to M-Estimation phase"
set_MEstimator(e::MMEstimator) = (e.scaleest=false; e)

function show(io::IO, obj::MMEstimator)
    print(io, "MM-Estimator($(obj.loss1), $(obj.loss2))")
end

# Forward all methods to the selected loss
rho(   E::MMEstimator, r::Real) = rho(loss(E), r)
psi(   E::MMEstimator, r::Real) = psi(loss(E), r)
psider(E::MMEstimator, r::Real) = psider(loss(E), r)
weight(E::MMEstimator, r::Real) = weight(loss(E), r)
values(E::MMEstimator, r::Real) = values(loss(E), r)

# For these methods, only the SEstimator loss is useful,
# not the MEstimator, so E.loss1 is used instead of loss(E)
estimator_bound(E::MMEstimator) = estimator_bound(typeof(E.loss1))
# For these methods, only the MEstimator loss is useful,
# not the SEstimator, so E.loss2 is used instead of loss(E)
estimator_norm(E::MMEstimator, args...) = estimator_norm(E.loss2, args...)
isbounded(E::MMEstimator) = isbounded(E.loss2)
isconvex( E::MMEstimator) = isconvex(E.loss2)

scale_estimate(est::E, res; kwargs...) where {E<:MMEstimator} = scale_estimate(est.loss1, res; kwargs...)


######
###   τ-Estimators
######

"""
    TauEstimator{L1<:BoundedLossFunction, L2<:BoundedLossFunction} <: AbstractEstimator
    
τ-estimator for the given loss functions.

The τ-estimator corresponds to a M-estimation, where the loss function is a weighted
sum of a high breakdown point loss and an efficient loss. The weight is recomputed at
every step of the Iteratively Reweighted Least Square, so the estimate is both robust
(high breakdown point) and efficient.


# Fields
- `loss1`: the high breakdown point [`BoundedLossFunction`](@ref).
- `loss2`: the high efficiency [`BoundedLossFunction`](@ref).
- `w`: the weight in the sum of losses: `w . loss1 + loss2`.

"""
mutable struct TauEstimator{L1<:BoundedLossFunction, L2<:BoundedLossFunction} <: AbstractEstimator
    "high breakdown point loss function"
    loss1::L1

    "high efficiency loss function"
    loss2::L2

    "loss weight"
    w::Float64
    
    TauEstimator{L1, L2}(l1::L1, l2::L2, w::Real=0.0) where {L1<:BoundedLossFunction, L2<:BoundedLossFunction} = new(l1, l2, float(w))
end
TauEstimator(l1::L1, l2::L2, args...) where {L1<:BoundedLossFunction, L2<:BoundedLossFunction} = TauEstimator{L1, L2}(l2, l2, args...)
# Warning: The tuning constant of the the efficient loss is NOT optimized for different loss functions
TauEstimator(::Type{L1}, ::Type{L2}, args...) where {L1<:BoundedLossFunction, L2<:BoundedLossFunction} =
            TauEstimator(robust_loss(L1), efficient_loss(L2), args...)

# With the same loss function, the tuning constant of the the efficient loss is optimized
TauEstimator{L}() where L<:BoundedLossFunction =
            TauEstimator(robust_loss(L), L(estimator_tau_efficient_constant(L)))
TauEstimator(::Type{L}) where L<:BoundedLossFunction = TauEstimator{L}()

loss(e::TauEstimator) = CompositeLossFunction(e.loss1, e.loss2, e.w, 1)

function show(io::IO, obj::TauEstimator)
    print(io, "τ-Estimator($(obj.loss1), $(obj.loss2))")
end

"Compute the tuning constant that corresponds to a high breakdown point for the τ-estimator."
function tau_efficiency_tuning_constant(::Type{L1}, ::Type{L2}; eff::Real=0.95, c0::Real=1.0) where {L1<:BoundedLossFunction, L2<:BoundedLossFunction}
    loss1 = L1(estimator_high_breakdown_point_constant(L1))
    w1 = quadgk(x-> x*psi(loss1, x)*2*exp(-x^2/2)/√(2π), 0, Inf)[1]

    function τest(c)
        loss2 = L2(c)
        w2 = quadgk(x->(2*rho(loss2, x)*(tuning_constant(loss2))^2 - x * psi(loss2, x))*
                        2*exp(-x^2/2)/√(2π), 0, Inf)[1]
        TauEstimator{L1, L2}(loss1, loss2, w2 / w1)
    end

    lpsi(x, c)  = psi(τest(c), x)
    lpsip(x, c) = psider(τest(c), x)

    I1(c) = quadgk(x->(lpsi(x, c))^2*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    I2(c) = quadgk(x->lpsip(x, c)*2*exp(-x^2/2)/√(2π), 0, Inf)[1]
    fun_eff(c) = (I2(c))^2/I1(c)
    copt = find_zero(c->fun_eff(c) - eff, c0, Order1())
end
tau_efficiency_tuning_constant(::Type{L}; kwargs...) where {L<:BoundedLossFunction} = tau_efficiency_tuning_constant(L, L; kwargs...)

"The tuning constant associated to the loss that gives a robust τ-estimator."
estimator_tau_efficient_constant(::Type{GemanLoss})      = 5.632
estimator_tau_efficient_constant(::Type{WelschLoss})     = 4.043
estimator_tau_efficient_constant(::Type{TukeyLoss})      = 6.040
estimator_tau_efficient_constant(::Type{YohaiZamarLoss}) = 3.270


"""
    update_weight!(E::TauEstimator, res::AbstractArray{T}; wts::AbstractArray{T}=T[])

Update the weight between the two estimators of a τ-estimator using the scaled residual.
"""
function update_weight!(E::TauEstimator, res::AbstractArray{T};
            wts::AbstractArray{T}=T[]) where {T<:AbstractFloat}
    c² = (tuning_constant(E.loss2))^2
    E.w = if length(wts) == length(res)
        w2 = sum(@.( wts*(2*rho(E.loss2, res)*c² - res * psi(E.loss2, res)) ))
        w1 = sum(@.( wts * res * psi(E.loss1, res) ))
        w2 / w1
    else
        w2 = sum(r-> 2*rho(E.loss2, r)*c² - r*psi(E.loss2, r), res)
        w1 = sum(r-> r*psi(E.loss1, r), res)
        w2 / w1
    end
    E
end
update_weight!(E::TauEstimator, res::AbstractArray; wts::AbstractArray=[]) = update_weight!(E, float(res); wts=float(wts))
update_weight!(E::TauEstimator, w::Real) = (E.w = w; E)

# Forward all methods to the `loss` fields
rho(   E::TauEstimator, r::Real) = E.w * rho(E.loss1, r) * (tuning_constant(E.loss1))^2 +
                                         rho(E.loss2, r) * (tuning_constant(E.loss2))^2
psi(   E::TauEstimator, r::Real) = E.w * psi(E.loss1, r) + psi(E.loss2, r)
psider(E::TauEstimator, r::Real) = E.w * psider(E.loss1, r) + psider(E.loss2, r)
weight(E::TauEstimator, r::Real) = E.w * weight(E.loss1, r) + weight(E.loss2, r)
function values(E::TauEstimator, r::Real)
    vals1 = values(E.loss1, r)
    vals2 = values(E.loss2, r)
    c12, c22 = (tuning_constant(E.loss1))^2, (tuning_constant(E.loss2))^2
    (E.w * vals1[1] * c12 + vals2[1] * c22, E.w * vals1[2] + vals2[3], E.w * vals1[3] + vals2[3])
end
estimator_norm(E::TauEstimator, args...) = Inf
estimator_bound(E::TauEstimator) = estimator_bound(typeof(E.loss1))
isbounded(E::TauEstimator) = true
isconvex( E::TauEstimator) = false

scale_estimate(est::E, res; kwargs...) where {E<:TauEstimator} = scale_estimate(est.loss1, res; kwargs...)

"""
    tau_scale_estimate!(E::TauEstimator, res::AbstractArray{T}, σ::Real, sqr::Bool=false;
                        wts::AbstractArray{T}=T[], bound::AbstractFloat=0.5) where {T<:AbstractFloat}

The τ-scale estimate, where `σ` is the scale estimate from the robust M-scale.
If `sqr` is true, return the squared value.
"""
function tau_scale_estimate(est::TauEstimator, res::AbstractArray{T}, σ::Real, sqr::Bool=false;
                            wts::AbstractArray=[], bound::AbstractFloat=0.5) where {T<:AbstractFloat}
    t = if length(wts) == length(res)
        mean( mscale_loss.(Ref(est.loss2), res ./ σ), weights(r.wts) ) / bound
    else
        mean( mscale_loss.(Ref(est.loss2), res ./ σ)) / bound
    end
    if sqr; σ * √t else σ^2 * t end
end




######
###   MQuantile Estimators
######
"""
    quantile_weight(τ::Real, r::Real)

Wrapper function to compute quantile-like loss function.
"""
quantile_weight(τ::Real, r::Real) = oftype(r, 2*ifelse(r>0, τ, 1 - τ))

"""
    GeneralizedQuantileEstimator{L<:LossFunction} <: AbstractQuantileEstimator
    
Generalized Quantile Estimator is an M-Estimator with asymmetric loss function.

For [`L1Loss`](@ref), this corresponds to quantile regression (although it is better
to use [`quantreg`](@ref) for quantile regression because it gives the exact solution).

For [`L2Loss`](@ref), this corresponds to Expectile regression (see [`ExpectileEstimator`](@ref)).

# Fields
- `loss`: the [`LossFunction`](@ref).
- `τ`: the quantile value to estimate, between 0 and 1.

# Properties
- `tau`, `q`, `quantile` are aliases for `τ`.
"""
mutable struct GeneralizedQuantileEstimator{L<:LossFunction} <: AbstractQuantileEstimator
    loss::L
    τ::Float64
end
function GeneralizedQuantileEstimator(l::L, τ::Real=0.5) where L<:LossFunction
    (0 < τ < 1) || throw(DomainError(τ, "quantile should be a number between 0 and 1 excluded"))
    GeneralizedQuantileEstimator{L}(l, float(τ))
end
GeneralizedQuantileEstimator{L}(τ::Real=0.5) where L<:LossFunction = GeneralizedQuantileEstimator(L(), float(τ))

function ==(e1::GeneralizedQuantileEstimator{L1}, e2::GeneralizedQuantileEstimator{L2}) where {L1<:LossFunction, L2<:LossFunction}
    if (L1 !== L2) || (loss(e1) != loss(e2)) || (e1.τ != e2.τ)
        return false
    end
    return true
end
function show(io::IO, obj::GeneralizedQuantileEstimator)
    print(io, "MQuantile($(obj.τ), $(obj.loss))")
end
loss(e::GeneralizedQuantileEstimator) = e.loss

function Base.getproperty(r::GeneralizedQuantileEstimator, s::Symbol)
    if s ∈ (:tau, :q, :quantile)
        r.τ
    else
        getfield(r, s)
    end
end

function Base.setproperty!(r::GeneralizedQuantileEstimator, s::Symbol, v)
    if s ∈ (:tau, :q, :quantile)
        (0 < v < 1) || throw(DomainError(v, "quantile should be a number between 0 and 1 excluded"))
        r.τ = float(v)
    else
        setfield!(r, s, v)
    end
end

Base.propertynames(r::GeneralizedQuantileEstimator, private=false) = (:loss, :τ, :tau, :q, :quantile)


# Forward all methods to the `loss` field
rho(   e::GeneralizedQuantileEstimator, r::Real) = quantile_weight(e.τ, r) * rho(   e.loss, r)
psi(   e::GeneralizedQuantileEstimator, r::Real) = quantile_weight(e.τ, r) * psi(   e.loss, r)
psider(e::GeneralizedQuantileEstimator, r::Real) = quantile_weight(e.τ, r) * psider(e.loss, r)
weight(e::GeneralizedQuantileEstimator, r::Real) = quantile_weight(e.τ, r) * weight(e.loss, r)
function values(e::GeneralizedQuantileEstimator, r::Real)
    w = quantile_weight(e.τ, r)
    vals = values(e.loss, r)
    Tuple([x * w for x in vals])
end
estimator_norm(e::GeneralizedQuantileEstimator, args...) = estimator_norm(e.loss, args...)
estimator_bound(e::GeneralizedQuantileEstimator) = estimator_bound(loss(e))
isbounded(e::GeneralizedQuantileEstimator) = isbounded(e.loss)
isconvex( e::GeneralizedQuantileEstimator) = isconvex( e.loss)

function scale_estimate(est::E, res; kwargs...) where {E<:GeneralizedQuantileEstimator}
    error("the M-scale estimate of a generalized quantile estimator is not defined")
end

"""
The expectile estimator is a generalization of the L2 estimator, for other quantile τ ∈ [0,1].

[1] Schnabel, Eilers - Computational Statistics and Data Analysis 53 (2009) 4168–4177 - Optimal expectile smoothing
doi:10.1016/j.csda.2009.05.002
"""
const ExpectileEstimator = GeneralizedQuantileEstimator{L2Loss}

"Non-exact quantile estimator, `GeneralizedQuantileEstimator{L1Loss}`. Prefer using [`QuantileRegression`](@ref)"
const QuantileEstimator = GeneralizedQuantileEstimator{L1Loss}

const UnionL1 = Union{L1Estimator, GeneralizedQuantileEstimator{L1Loss}}

const UnionMEstimator = Union{MEstimator, GeneralizedQuantileEstimator}
