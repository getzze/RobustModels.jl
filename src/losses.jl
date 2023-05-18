## Threshold to avoid numerical overflow of the weight function of L1Estimator and ArctanEstimator
DELTA = 1e-8    # chosen because it minimizes the error between (1-ATWDELTA)/DELTA and 1/3 for the linear approximation
L1WDELTA = 1 / (DELTA)
ATWDELTA = atan(DELTA) / DELTA
#DELTA = 1e-8    # chosen because it minimizes the error between (1-ATWDELTA)/DELTA and 1/3 for the linear approximation
#L1WDELTA = 1/(2*sqrt(DELTA))
#ATWDELTA = atan(sqrt(DELTA))*2*L1WDELTA


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
estimator_bound(f::LossFunction) = Inf

"The tuning constant of the loss function, can be optimized to get efficient or robust estimates."
tuning_constant(loss::L) where {L<:LossFunction} = loss.c

"Boolean if the estimator or loss function is convex"
isconvex(f::LossFunction) = false
isconvex(f::ConvexLossFunction) = true

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
function robust_loss(::Type{L}) where {L<:LossFunction}
    return L(estimator_high_breakdown_point_constant(L))
end


rho(l::LossFunction, r) = _rho(l, r / tuning_constant(l))
psi(l::LossFunction, r) = tuning_constant(l) * _psi(l, r / tuning_constant(l))
psider(l::LossFunction, r) = _psider(l, r / tuning_constant(l))
weight(l::LossFunction, r) = _weight(l, r / tuning_constant(l))

"Faster version if you need ρ, ψ and w in the same call"
function estimator_values(l::LossFunction, r::Real)
    c = tuning_constant(l)
    rr = r / c
    return (_rho(l, rr), c * _psi(l, rr), _weight(l, rr))
end

"""
    threshold(l::LossFunction, x, λ) = x / λ - psi(l, x / λ)

Threshold function associated with the loss, with optional factor.
"""
threshold(l::LossFunction, x, λ::Real=1.0) = x / λ - psi(l, x / λ)


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
function estimator_norm(l::L) where {L<:LossFunction}
    return 2 * quadgk(x -> exp(-rho(l, x)), 0, Inf)[1]
end


"""
The tuning constant c is computed so the efficiency for Normal distributed
residuals is 0.95. The efficiency of the mean estimate μ is defined by:

eff_μ = (E[ψ'])²/E[ψ²]
"""
function efficiency_tuning_constant(
    ::Type{L}; eff::Real=0.95, c0::Real=1.0
) where {L<:LossFunction}
    lpsi(x, c) = RobustModels.psi(L(c), x)
    lpsip(x, c) = RobustModels.psider(L(c), x)

    I1(c) = quadgk(x -> (lpsi(x, c))^2 * 2 * exp(-x^2 / 2) / √(2π), 0, Inf)[1]
    I2(c) = quadgk(x -> lpsip(x, c) * 2 * exp(-x^2 / 2) / √(2π), 0, Inf)[1]
    fun_eff(c) = (I2(c))^2 / I1(c)
    return copt = find_zero(c -> fun_eff(c) - eff, c0, Order1())
end


"""
    mscale_loss(loss::L, x)

The rho-function that is used for M-scale estimation.

For monotone (convex) functions, χ(r) = r.ψ(r)/c^2.

For bounded functions, χ(r) = ρ(r)/ρ(∞) so χ(∞) = 1.
"""
mscale_loss(l::LossFunction, x) = x * RobustModels.psi(l, x) / (tuning_constant(l))^2
mscale_loss(l::BoundedLossFunction, x) = RobustModels.rho(l, x) / estimator_bound(l)


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
function breakdown_point_tuning_constant(
    ::Type{L}; bp::Real=1 / 2, c0::Real=1.0
) where {L<:LossFunction}
    (0 < bp <= 1 / 2) || error("breakdown-point should be between 0 and 1/2")

    function I(c)
        return quadgk(
            x -> RobustModels.mscale_loss(L(c), x) * 2 * exp(-x^2 / 2) / √(2π), 0, Inf
        )[1]
    end
    return copt = find_zero(c -> I(c) - bp, c0, Order1())
end




########
###     Loss functions
########

"""
The (convex) L2 loss function is that of the standard least squares problem.
ψ(r) = r
"""
struct L2Loss <: ConvexLossFunction end
L2Loss(c) = L2Loss()
rho(::L2Loss, r::Real) = r^2 / 2
psi(::L2Loss, r::Real) = r
psider(::L2Loss, r::Real) = oftype(r, 1)
weight(::L2Loss, r::Real) = oftype(r, 1)
estimator_values(::L2Loss, r::Real) = (r^2 / 2, r, oftype(r, 1))
estimator_norm(::L2Loss) = √(2 * π)
tuning_constant(::L2Loss) = 1



"""
The standard L1 loss function takes the absolute value of the residual, and is
convex but non-smooth. It is not a real L1 loss but a Huber loss
with very small tuning constant.
ψ(r) = sign(r)
Use ``QuantileRegression`` for a correct implementation of the L1 loss.
"""
struct L1Loss <: ConvexLossFunction end
L1Loss(c) = L1Loss()
rho(::L1Loss, r::Real) = abs(r)
psi(::L1Loss, r::Real) = sign(r)
psider(::L1Loss, r::Real) = (abs(r) < DELTA) ? oftype(r, 1) : oftype(r, 0)
weight(::L1Loss, r::Real) = (abs(r) < DELTA) ? L1WDELTA : 1 / abs(r)
function estimator_values(est::L1Loss, r::Real)
    rr = abs(r)
    return (rr, sign(r), ((rr < DELTA) ? L1WDELTA : 1 / rr))
end
estimator_norm(::L1Loss) = 2
tuning_constant(::L1Loss) = 1


"""
The convex Huber loss function switches from between quadratic and linear cost/loss
function at a certain cutoff.
ψ(r) = (abs(r) <= 1) ? r : sign(r)
"""
struct HuberLoss <: ConvexLossFunction
    c::Float64

    HuberLoss(c::Real) = new(c)
    HuberLoss() = new(estimator_high_efficiency_constant(HuberLoss))
end

_rho(l::HuberLoss, r::Real) = (abs(r) <= 1) ? r^2 / 2 : (abs(r) - 1 / 2)
_psi(l::HuberLoss, r::Real) = (abs(r) <= 1) ? r : sign(r)
_psider(l::HuberLoss, r::Real) = (abs(r) <= 1) ? oftype(r, 1) : oftype(r, 0)
_weight(l::HuberLoss, r::Real) = (abs(r) <= 1) ? oftype(r, 1) : 1 / abs(r)
function estimator_values(l::HuberLoss, r::Real)
    rr = abs(r)
    if rr <= l.c
        return ((rr / l.c)^2 / 2, r, oftype(r, 1))
    else
        return (rr / l.c - 1 / 2, l.c * sign(r), l.c / rr)
    end
end
estimator_norm(l::HuberLoss) = l.c * 2.92431
estimator_high_efficiency_constant(::Type{HuberLoss}) = 1.345
estimator_high_breakdown_point_constant(::Type{HuberLoss}) = 0.6745


"""
The convex L1-L2 loss interpolates smoothly between L2 behaviour for small
residuals and L1 for outliers.
ψ(r) = r / √(1 + r^2)
"""
struct L1L2Loss <: ConvexLossFunction
    c::Float64

    L1L2Loss(c::Real) = new(c)
    L1L2Loss() = new(estimator_high_efficiency_constant(L1L2Loss))
end
_rho(l::L1L2Loss, r::Real) = (sqrt(1 + r^2) - 1)
_psi(l::L1L2Loss, r::Real) = r / sqrt(1 + r^2)
_psider(l::L1L2Loss, r::Real) = 1 / (1 + r^2)^(3 / 2)
_weight(l::L1L2Loss, r::Real) = 1 / sqrt(1 + r^2)
function estimator_values(l::L1L2Loss, r::Real)
    sqr = sqrt(1 + (r / l.c)^2)
    return ((sqr - 1), r / sqr, 1 / sqr)
end
estimator_norm(l::L1L2Loss) = l.c * 3.2723
estimator_high_efficiency_constant(::Type{L1L2Loss}) = 1.287
estimator_high_breakdown_point_constant(::Type{L1L2Loss}) = 0.8252


"""
The (convex) "fair" loss switches from between quadratic and linear
cost/loss function at a certain cutoff, and is C3 but non-analytic.
ψ(r) = r / (1 + abs(r))
"""
struct FairLoss <: ConvexLossFunction
    c::Float64

    FairLoss(c::Real) = new(c)
    FairLoss() = new(estimator_high_efficiency_constant(FairLoss))
end
_rho(l::FairLoss, r::Real) = abs(r) - log(1 + abs(r))
_psi(l::FairLoss, r::Real) = r / (1 + abs(r))
_psider(l::FairLoss, r::Real) = 1 / (1 + abs(r))^2
_weight(l::FairLoss, r::Real) = 1 / (1 + abs(r))
function estimator_values(l::FairLoss, r::Real)
    ir = 1 / (1 + abs(r / l.c))
    return (abs(r) / l.c + log(ir), r * ir, ir)
end
estimator_norm(l::FairLoss) = l.c * 4
estimator_high_efficiency_constant(::Type{FairLoss}) = 1.400
estimator_high_breakdown_point_constant(::Type{FairLoss}) = 1.4503

"""
The convex Log-Cosh loss function
ψ(r) = tanh(r)
"""
struct LogcoshLoss <: ConvexLossFunction
    c::Float64

    LogcoshLoss(c::Real) = new(c)
    LogcoshLoss() = new(estimator_high_efficiency_constant(LogcoshLoss))
end
_rho(l::LogcoshLoss, r::Real) = log(cosh(r))
_psi(l::LogcoshLoss, r::Real) = tanh(r)
_psider(l::LogcoshLoss, r::Real) = 1 / (cosh(r))^2
_weight(l::LogcoshLoss, r::Real) = (abs(r) < DELTA) ? (1 - (r)^2 / 3) : tanh(r) / r
function estimator_values(l::LogcoshLoss, r::Real)
    tr = l.c * tanh(r / l.c)
    rr = abs(r / l.c)
    return (log(cosh(rr)), tr, ((rr < DELTA) ? (1 - rr^2 / 3) : tr / r))
end
estimator_norm(l::LogcoshLoss) = l.c * π
estimator_high_efficiency_constant(::Type{LogcoshLoss}) = 1.2047
estimator_high_breakdown_point_constant(::Type{LogcoshLoss}) = 0.7479

"""
The convex Arctan loss function
ψ(r) = atan(r)
"""
struct ArctanLoss <: ConvexLossFunction
    c::Float64

    ArctanLoss(c::Real) = new(c)
    ArctanLoss() = new(estimator_high_efficiency_constant(ArctanLoss))
end
_rho(l::ArctanLoss, r::Real) = r * atan(r) - 1 / 2 * log(1 + r^2)
_psi(l::ArctanLoss, r::Real) = atan(r)
_psider(l::ArctanLoss, r::Real) = 1 / (1 + r^2)
_weight(l::ArctanLoss, r::Real) = (abs(r) < DELTA) ? (1 - r^2 / 3) : atan(r) / r
function estimator_values(l::ArctanLoss, r::Real)
    ar = atan(r / l.c)
    rr = abs(r / l.c)
    return (
        r * ar / l.c - 1 / 2 * log(1 + rr^2),
        l.c * ar,
        ((rr < DELTA) ? (1 - rr^2 / 3) : l.c * ar / r),
    )
end
estimator_norm(l::ArctanLoss) = l.c * 2.98151
estimator_high_efficiency_constant(::Type{ArctanLoss}) = 0.919
estimator_high_breakdown_point_constant(::Type{ArctanLoss}) = 0.612

"""
The convex (wide) Catoni loss function.
See: "Catoni (2012) - Challenging the empirical mean and empirical variance: A deviation study"

ψ(r) = sign(r) * log(1 + abs(r) + r^2/2)
"""
struct CatoniWideLoss <: ConvexLossFunction
    c::Float64

    CatoniWideLoss(c::Real) = new(c)
    CatoniWideLoss() = new(estimator_high_efficiency_constant(CatoniWideLoss))
end

function _rho(l::CatoniWideLoss, r::Real)
    return (1 + abs(r)) * log(1 + abs(r) + r^2 / 2) - 2 * abs(r) + 2 * atan(1 + abs(r)) -
           π / 2
end
_psi(l::CatoniWideLoss, r::Real) = sign(r) * log(1 + abs(r) + r^2 / 2)
_psider(l::CatoniWideLoss, r::Real) = (1 + abs(r)) / (1 + abs(r) + r^2 / 2)
function _weight(l::CatoniWideLoss, r::Real)
    return (abs(r) < DELTA) ? oftype(r, 1) : log(1 + abs(r) + r^2 / 2) / abs(r)
end
function estimator_values(l::CatoniWideLoss, r::Real)
    rr = abs(r / l.c)
    lr = log(1 + rr + rr^2 / 2)
    return (
        (1 + rr) * lr - 2 * rr + 2 * atan(1 + rr) - π / 2,
        sign(r) * l.c * lr,
        (r < DELTA) ? oftype(r, 1) : lr / rr,
    )
end
estimator_norm(l::CatoniWideLoss) = l.c * 2.64542
estimator_high_efficiency_constant(::Type{CatoniWideLoss}) = 0.21305
estimator_high_breakdown_point_constant(::Type{CatoniWideLoss}) = 0.20587


"""
The convex (narrow) Catoni loss function.
See: "Catoni (2012) - Challenging the empirical mean and empirical variance: A deviation study"

ψ(r) = (abs(r) <= 1) ? -sign(r) * log(1 - abs(r) + r^2/2) : sign(r) * log(2)
"""
struct CatoniNarrowLoss <: ConvexLossFunction
    c::Float64

    CatoniNarrowLoss(c::Real) = new(c)
    CatoniNarrowLoss() = new(estimator_high_efficiency_constant(CatoniNarrowLoss))
end

function _rho(l::CatoniNarrowLoss, r::Real)
    if abs(r) <= 1
        return (1 - abs(r)) * log(1 - abs(r) + r^2 / 2) +
               2 * abs(r) +
               2 * atan(1 - abs(r)) - π / 2
    end
    return (abs(r) - 1) * log(2) + 2 - π / 2
end
function _psi(l::CatoniNarrowLoss, r::Real)
    if abs(r) <= 1
        return -sign(r) * log(1 - abs(r) + r^2 / 2)
    end
    return sign(r) * log(2)
end
function _psider(l::CatoniNarrowLoss, r::Real)
    if abs(r) <= 1
        return (1 - abs(r)) / (1 - abs(r) + r^2 / 2)
    end
    return 0
end
function _weight(l::CatoniNarrowLoss, r::Real)
    if r == 0
        return oftype(r, 1)
    elseif abs(r) <= 1
        return -log(1 - abs(r) + r^2 / 2) / abs(r)
    end
    return log(2) / abs(r)
end
function estimator_values(l::CatoniNarrowLoss, r::Real)
    rr = abs(r / l.c)
    lr = log(1 - rr + rr^2 / 2)
    if abs(r) <= 1
        return (
            (1 - rr) * lr + 2 * rr + 2 * atan(1 - rr) - π / 2, -sign(r) * l.c * lr, -lr / rr
        )
    end
    return ((rr - 1) * log(2) + 2 - π / 2, sign(r) * log(2), log(2) / rr)
end
estimator_norm(l::CatoniNarrowLoss) = l.c * 3.60857
estimator_high_efficiency_constant(::Type{CatoniNarrowLoss}) = 1.7946
estimator_high_breakdown_point_constant(::Type{CatoniNarrowLoss}) = 0.9949


"""
The non-convex Cauchy loss function switches from between quadratic behaviour to
logarithmic tails. This rejects outliers but may result in multiple minima.
For scale estimate, r.ψ(r) is used as a loss, which is the same as for Geman loss.
ψ(r) = r / (1 + r^2)
"""
struct CauchyLoss <: LossFunction
    c::Float64

    CauchyLoss(c::Real) = new(c)
    CauchyLoss() = new(estimator_high_efficiency_constant(CauchyLoss))
end
_rho(l::CauchyLoss, r::Real) = log(1 + r^2) # * 1/2  # remove factor 1/2 so the loss has a norm
_psi(l::CauchyLoss, r::Real) = r / (1 + r^2)
_psider(l::CauchyLoss, r::Real) = (1 - r^2) / (1 + r^2)^2
_weight(l::CauchyLoss, r::Real) = 1 / (1 + r^2)
function estimator_values(l::CauchyLoss, r::Real)
    ir = 1 / (1 + (r / l.c)^2)
    return (-log(ir), r * ir, ir)
end
estimator_norm(l::CauchyLoss) = l.c * π
estimator_bound(l::CauchyLoss) = 1.0
isconvex(::CauchyLoss) = false
isbounded(::CauchyLoss) = false

estimator_high_efficiency_constant(::Type{CauchyLoss}) = 2.385
estimator_high_breakdown_point_constant(::Type{CauchyLoss}) = 1.468

"""
The non-convex Geman-McClure for strong supression of outliers and does not guarantee a unique solution.
For the S-Estimator, it is equivalent to the Cauchy loss.
ψ(r) = r / (1 + r^2)^2
"""
struct GemanLoss <: BoundedLossFunction
    c::Float64

    GemanLoss(c::Real) = new(c)
    GemanLoss() = new(estimator_high_efficiency_constant(GemanLoss))
end
_rho(l::GemanLoss, r::Real) = 1 / 2 * r^2 / (1 + r^2)
_psi(l::GemanLoss, r::Real) = r / (1 + r^2)^2
_psider(l::GemanLoss, r::Real) = (1 - 3 * r^2) / (1 + r^2)^3
_weight(l::GemanLoss, r::Real) = 1 / (1 + r^2)^2
function estimator_values(l::GemanLoss, r::Real)
    rr2 = (r / l.c)^2
    ir = 1 / (1 + rr2)
    return (1 / 2 * rr2 * ir, r * ir^2, ir^2)
end
estimator_norm(::GemanLoss) = Inf
estimator_bound(::GemanLoss) = 1 / 2
isconvex(::GemanLoss) = false
isbounded(::GemanLoss) = true

estimator_high_efficiency_constant(::Type{GemanLoss}) = 3.787
estimator_high_breakdown_point_constant(::Type{GemanLoss}) = 0.61200



"""
The non-convex Welsch for strong supression of outliers and does not guarantee a unique solution
ψ(r) = r * exp(-r^2)
"""
struct WelschLoss <: BoundedLossFunction
    c::Float64

    WelschLoss(c::Real) = new(c)
    WelschLoss() = new(estimator_high_efficiency_constant(WelschLoss))
end
_rho(l::WelschLoss, r::Real) = -1 / 2 * Base.expm1(-r^2)
_psi(l::WelschLoss, r::Real) = r * exp(-r^2)
_psider(l::WelschLoss, r::Real) = (1 - 2 * r^2) * exp(-r^2)
_weight(l::WelschLoss, r::Real) = exp(-r^2)
function estimator_values(l::WelschLoss, r::Real)
    rr2 = (r / l.c)^2
    er = exp(-rr2)
    return (-1 / 2 * Base.expm1(-rr2), r * er, er)
end
estimator_norm(::WelschLoss) = Inf
estimator_bound(::WelschLoss) = 1 / 2
isconvex(::WelschLoss) = false
isbounded(::WelschLoss) = true

estimator_high_efficiency_constant(::Type{WelschLoss}) = 2.985
estimator_high_breakdown_point_constant(::Type{WelschLoss}) = 0.8165


"""
The non-convex Tukey biweight estimator which completely suppresses the outliers,
and does not guaranty a unique solution.
ψ(r) = (abs(r) <= 1) ? r * (1 - r^2)^2 : 0
"""
struct TukeyLoss <: BoundedLossFunction
    c::Float64

    TukeyLoss(c::Real) = new(c)
    TukeyLoss() = new(estimator_high_efficiency_constant(TukeyLoss))
end
_rho(l::TukeyLoss, r::Real) = (abs(r) <= 1) ? 1 / 6 * (1 - (1 - r^2)^3) : 1 / 6
_psi(l::TukeyLoss, r::Real) = (abs(r) <= 1) ? r * (1 - r^2)^2 : oftype(r, 0)
_psider(l::TukeyLoss, r::Real) = (abs(r) <= 1) ? 1 - 6 * r^2 + 5 * r^4 : oftype(r, 0)
_weight(l::TukeyLoss, r::Real) = (abs(r) <= 1) ? (1 - r^2)^2 : oftype(r, 0)
function estimator_values(l::TukeyLoss, r::Real)
    pr = (abs(r) <= l.c) * (1 - (r / l.c)^2)
    return (1 / 6 * (1 - pr^3), r * pr^2, pr^2)
end
estimator_norm(::TukeyLoss) = Inf
estimator_bound(::TukeyLoss) = 1 / 6
isconvex(::TukeyLoss) = false
isbounded(::TukeyLoss) = true

estimator_high_efficiency_constant(::Type{TukeyLoss}) = 4.685
estimator_high_breakdown_point_constant(::Type{TukeyLoss}) = 1.5476


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
    z = (r / l.c)^2
    if (z <= 4 / 9)
        1.3846 * z
    elseif (z <= 1)
        min(1.0, 0.5514 - 2.6917 * z + 10.7668 * z^2 - 11.6640 * z^3 + 4.0375 * z^4)
    else
        oftype(r, 1)
    end
end
function psi(l::YohaiZamarLoss, r::Real)
    z = (r / l.c)^2
    if (z <= 4 / 9)
        2.7692 * r
    elseif (z <= 1)
        r * max(0, -5.3834 + 43.0672 * z - 69.984 * z^2 + 32.3 * z^3)
    else
        oftype(r, 0)
    end
end
function psider(l::YohaiZamarLoss, r::Real)
    z = (r / l.c)^2
    if (z <= 4 / 9)
        2.7692
    elseif (z <= 0.997284)  # from the root of ψ expression
        -5.3834 + 129.2016 * z - 349.92 * z^2 + 226.1 * z^3
    else
        oftype(r, 0)
    end
end
function weight(l::YohaiZamarLoss, r::Real)
    z = (r / l.c)^2
    if (z <= 4 / 9)
        2.7692
    elseif (z <= 1)
        max(0, -5.3834 + 43.0672 * z - 69.984 * z^2 + 32.3 * z^3)
    else
        oftype(r, 0)
    end
end
function estimator_values(l::YohaiZamarLoss, r::Real)
    z = (r / l.c)^2
    if (z <= 4 / 9)
        return (1.3846 * z, 2.7692 * r, 2.7692)
    elseif (z <= 1)
        ρ = min(1, 0.5514 - 2.6917 * z + 10.7668 * z^2 - 11.6640 * z^3 + 4.0375 * z^4)
        w = max(0, -5.3834 + 43.0672 * z - 69.984 * z^2 + 32.3 * z^3)
        return (ρ, r * w, w)
    else
        return (oftype(r, 1), oftype(r, 0), oftype(r, 0))
    end
end
estimator_norm(::YohaiZamarLoss) = Inf
estimator_bound(::YohaiZamarLoss) = 1
isconvex(::YohaiZamarLoss) = false
isbounded(::YohaiZamarLoss) = true

estimator_high_efficiency_constant(::Type{YohaiZamarLoss}) = 3.1806
estimator_high_breakdown_point_constant(::Type{YohaiZamarLoss}) = 1.2139


"""
The non-convex hard-threshold loss function, or saturated L2 loss. Non-smooth.
ψ(r) = (abs(r) <= 1) ? r : 0
"""
struct HardThresholdLoss <: BoundedLossFunction
    c::Float64

    HardThresholdLoss(c::Real) = new(c)
    HardThresholdLoss() = new(estimator_high_efficiency_constant(HardThresholdLoss))
end
_rho(l::HardThresholdLoss, r::Real) = (abs(r) <= 1) ? r^2 / 2 : oftype(r, 1 / 2)
_psi(l::HardThresholdLoss, r::Real) = (abs(r) <= 1) ? r : oftype(r, 0)
function _psider(l::HardThresholdLoss, r::Real)
    if abs(r) <= 1
        return oftype(r, 1)
    elseif abs(r) < 1 + DELTA
        return -L1WDELTA
    else
        return oftype(r, 0)
    end
end
_weight(l::HardThresholdLoss, r::Real) = (abs(r) <= 1) ? oftype(r, 1) : oftype(r, 0)
function estimator_values(l::HardThresholdLoss, r::Real)
    ar = abs(r / l.c)
    if ar <= 1
        return (ar^2 / 2, r, oftype(r, 1))
    else
        return (oftype(r, 1), oftype(r, 0), oftype(r, 0))
    end
end
estimator_norm(::HardThresholdLoss) = Inf
estimator_bound(::HardThresholdLoss) = 1 / 2
isconvex(::HardThresholdLoss) = false
isbounded(::HardThresholdLoss) = true

## Computed with `find_zero(c -> I1(c) - 0.95, 2.8, Order1())` after simplification (fun_eff = I1)
estimator_high_efficiency_constant(::Type{HardThresholdLoss}) = 2.795
estimator_high_breakdown_point_constant(::Type{HardThresholdLoss}) = 1.041


#################################
### Two (or more) parameters loss functions
#################################

"""
The 3-parameter non-convex bounded Hampel's loss function.
ψ(r) = (abs(r) <= 1) ? r : (
       (abs(r) <= l.ν1) ? sign(r) : (
       (abs(r) <= l.ν2) ? (l.ν2 - abs(r)) / (l.ν2 - l.ν1) * sign(r) : 0))
"""
struct HampelLoss <: BoundedLossFunction
    c::Float64
    ν1::Float64
    ν2::Float64

    function HampelLoss(c::Real, ν1::Real, ν2::Real)
        c >= 0 || throw(ArgumentError("constant c must be non-negative: $c"))
        ν1 >= 1 || throw(ArgumentError("constant ν1 must be greater than 1: $ν1"))
        ν2 >= ν1 ||
            throw(ArgumentError("constant ν2 must be greater than ν1: $ν2 >= ν1=$ν1"))
        return new(c, ν1, ν2)
    end
    HampelLoss(c::Real) = new(c, 2.0, 4.0)
    HampelLoss() = new(estimator_high_efficiency_constant(HampelLoss), 2.0, 4.0)
end
function _rho(l::HampelLoss, r::Real)
    if (abs(r) <= 1)
        return r^2 / 2
    elseif (abs(r) <= l.ν1)
        return abs(r) - 1 / 2
    elseif (abs(r) < l.ν2)
        return -(l.ν2 - abs(r))^2 / (2 * (l.ν2 - l.ν1)) + (l.ν1 + l.ν2 - 1) / 2
    else
        return (l.ν1 + l.ν2 - 1) / 2
    end
end
function _psi(l::HampelLoss, r::Real)
    if (abs(r) <= 1)
        return r
    elseif (abs(r) <= l.ν1)
        return sign(r)
    elseif (abs(r) < l.ν2)
        return (l.ν2 - abs(r)) / (l.ν2 - l.ν1) * sign(r)
    else
        return oftype(r, 0)
    end
end
function _psider(l::HampelLoss, r::Real)
    if (abs(r) <= 1)
        return oftype(r, 1)
    elseif (abs(r) <= l.ν1)
        return oftype(r, 0)
    elseif (abs(r) < l.ν2)
        return -1 / (l.ν2 - l.ν1)
    else
        return oftype(r, 0)
    end
end
function _weight(l::HampelLoss, r::Real)
    if (abs(r) <= 1)
        return oftype(r, 1)
    elseif (abs(r) <= l.ν1)
        return 1 / abs(r)
    elseif (abs(r) < l.ν2)
        return (l.ν2 / abs(r) - 1) / (l.ν2 - l.ν1)
    else
        return oftype(r, 0)
    end
end
estimator_norm(::HampelLoss) = Inf
estimator_bound(l::HampelLoss) = (l.ν1 + l.ν2 - 1) / 2
isconvex(::HampelLoss) = false
isbounded(::HampelLoss) = true

# Values of `c` for (ν1, ν2) = (2, 4)
estimator_high_efficiency_constant(::Type{HampelLoss}) = 1.382
estimator_high_breakdown_point_constant(::Type{HampelLoss}) = 0.396


######
###   Convex sum of loss functions
######
struct CompositeLossFunction{L1<:LossFunction,L2<:LossFunction} <: LossFunction
    α1::Float64
    loss1::L1
    α2::Float64
    loss2::L2
end
function CompositeLossFunction(
    loss1::LossFunction, loss2::LossFunction, α1::Real=1.0, α2::Real=1.0
)
    α1 >= 0 ||
        throw(DomainError(α1, "coefficients of CompositeLossFunction are non-negative."))
    α2 >= 0 ||
        throw(DomainError(α2, "coefficients of CompositeLossFunction are non-negative."))

    return CompositeLossFunction{typeof(loss1),typeof(loss2)}(
        float(α1), loss1, float(α2), loss2
    )
end

function show(io::IO, e::CompositeLossFunction)
    mess =
        "CompositeLossFunction $(round(e.α1; digits=2)) ." *
        " $(e.loss1) + $(round(e.α2; digits=2)) . $(e.loss2)"
    return print(io, mess)
end

Base.first(e::CompositeLossFunction) = e.loss1
Base.last(e::CompositeLossFunction) = e.loss2

isbounded(e::CompositeLossFunction) = isbounded(e.loss1) && isbounded(e.loss2)
isconvex(e::CompositeLossFunction) = isconvex(e.loss1) && isconvex(e.loss2)

function estimator_norm(e::CompositeLossFunction)
    return e.α1 * estimator_norm(e.loss1) + e.α2 * estimator_norm(e.loss2)
end
function estimator_bound(e::CompositeLossFunction)
    return e.α1 * estimator_bound(e.loss1) * (tuning_constant(e.loss1))^2 +
           e.α2 * estimator_bound(e.loss2) * (tuning_constant(e.loss2))^2
end
function rho(e::CompositeLossFunction, r::Real)
    return e.α1 * rho(e.loss1, r) * (tuning_constant(e.loss1))^2 +
           e.α2 * rho(e.loss2, r) * (tuning_constant(e.loss2))^2
end
psi(e::CompositeLossFunction, r::Real) = e.α1 * psi(e.loss1, r) + e.α2 * psi(e.loss2, r)
function psider(e::CompositeLossFunction, r::Real)
    return e.α1 * psider(e.loss1, r) + e.α2 * psider(e.loss2, r)
end
function weight(e::CompositeLossFunction, r::Real)
    return e.α1 * weight(e.loss1, r) + e.α2 * weight(e.loss2, r)
end
function estimator_values(e::CompositeLossFunction, r::Real)
    @.(e.α1 * estimator_values(e.loss1, r) + e.α2 * estimator_values(e.loss2, r))
end

function mscale_loss(e::CompositeLossFunction, x)
    if !isa(e.loss1, BoundedLossFunction) || !isa(e.loss2, BoundedLossFunction)
        error(
            "mscale_loss for CompositeLossFunction is defined only if both losses are bounded",
        )
    end
    return rho(e, x) / estimator_bound(e)
end
