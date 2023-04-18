
using QuadGK: quadgk
using Roots: find_zero, Order1, ConvergenceFailed, Newton

#=
TODO: change the API to
rlm(form, data, MEstimator(Tukey))
rlm(X, y, TauEstimator(YohaiZamar); method=:cg)
rlm(X, y, MMEstimator(Tukey(2.1), Tukey(5.0)); σ0=:mad)

=#


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
function scale_estimate(
    l::L,
    res::AbstractArray{T};
    verbose::Bool=false,
    bound::Union{Nothing,T}=nothing,
    σ0::T=1.0,
    wts::AbstractArray{T}=T[],
    order::Int=1,
    approx::Bool=false,
    nmax::Int=30,
    rtol::Real=1e-4,
    atol::Real=0.1,
) where {L<:LossFunction,T<:AbstractFloat}

    # Compute δ such that E[ρ] = δ for a Normal N(0, 1)
    if isnothing(bound)
        bound = quadgk(x -> mscale_loss(l, x) * 2 * exp(-x^2 / 2) / √(2π), 0, Inf)[1]
    end

    Nz = sum(iszero, res)
    if Nz > length(res) * (1 - bound)
        # The M-scale cannot be estimated because too many residuals are zeros.
        verbose && println(
            "there are too many zero residuals for M-scale estimation: #(r=0) > n*(1-b),"*
            " $(Nz) > $(length(res)*(1-bound))",
        )
        m = "the M-scale cannot be estimated because too many residuals are zeros: $(res)"
        throw(ConvergenceFailed(m))
    end

    # Approximate the solution with `nmax` iterations
    σn = σ0
    converged = false
    verbose && println("Initial M-scale estimate: $(σn)")
    for n in 1:nmax
        ## For non-bounded loss, we suppose that χ = r.ψ,
        ## therefore the weight ( χ/r² ) is ψ/r which is w.
        ww = weight.(l, res ./ σn)
        if length(wts) == length(res)
            ww .*= wts
        end
        σnp1 = sqrt(mean(res ./ σn, weights(ww)) / bound)

        verbose && println("M-scale update: $(σn) ->  $(σnp1)")

        ε = σnp1 / σn
        σn = σnp1
        if abs(ε - 1) < rtol
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

function scale_estimate(
    l::L,
    res::AbstractArray{T};
    verbose::Bool=false,
    bound::Union{Nothing,Real}=0.5,
    σ0::Real=1.0,
    wts::AbstractArray{T}=T[],
    order::Int=1,
    approx::Bool=false,
    nmax::Int=30,
    rtol::Real=1e-4,
    atol::Real=0.1,
) where {L<:BoundedLossFunction,T<:AbstractFloat}

    # Compute δ such that E[ρ] = δ for a Normal N(0, 1)
    if isnothing(bound)
        bound = quadgk(x -> mscale_loss(l, x) * 2 * exp(-x^2 / 2) / √(2π), 0, Inf)[1]
    end

    Nz = sum(iszero, res)
    if Nz > length(res) * (1 - bound)
        # The M-scale cannot be estimated because too many residuals are zeros.
        verbose && println(
            "there are too many zero residuals for M-scale estimation:"*
            " #(r=0) > n*(1-b), $(Nz) > $(length(res)*(1-bound))",
        )
        m = "the M-scale cannot be estimated because too many residuals are zeros: $(res)"
        throw(ConvergenceFailed(m))
    end

    # Approximate the solution with `nmax` iterations
    if !approx && order >= 2
        bb = tuning_constant(l)^2 * estimator_bound(l) * bound
    end
    σn = σ0
    converged = false
    verbose && println("Initial M-scale estimate: $(σn)")
    ε = 0
    for n in 1:nmax
        rr = res / σn
        ε = if length(wts) == length(res)
            mean(mscale_loss.(l, rr), weights(wts)) / bound
        else
            mean(x -> mscale_loss(l, x), rr) / bound
        end
        verbose && println("M-scale 1st order update: $(ε)")

        ## Implemented, but it gives worst results than 1st order...
        ## Uses Newton's method to find the root of
        ## f(σ) = log( 1/(nb) Σ ρ(r/σ) ) = 0
        if !approx && order >= 2
            εp = if length(wts) == length(res)
                mean(rr .* psi.(l, rr), weights(wts)) / bb
            else
                mean(x -> x * psi(l, x), rr) / bb
            end
            # Use the gradient only if the scale is not too small
            if isnan(εp) || !isfinite(εp) || εp <= atol
                verbose && println("M-scale order is set to 1 for this iteration")
            else
                ε = exp(ε * log(ε) / εp)
            end
        end

        σnp1 = σn * ε
        verbose && println("M-scale update: $(ε) : $(σn) ->  $(σnp1)")

        σn = σnp1
        if abs(ε - 1) < rtol
            verbose && println("M-scale converged after $(n) steps.")
            converged = true
            break
        end
    end

    if !approx && !converged
        @warn("the M-scale did not converge ε=$(round(abs(ε-1); digits=5)),"*
              " consider increasing the maximum number of iterations nmax=$(nmax)"*
              " or starting with a better initial value σ0=$(σ0)."*
              " Return the current estimate: $(σn)"
        )
    end
    return σn
end


######
###   M-Estimators
######
"""
    MEstimator{L<:LossFunction} <: AbstractMEstimator

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
struct MEstimator{L<:LossFunction} <: AbstractMEstimator
    loss::L
end
MEstimator{L}() where {L<:LossFunction} = MEstimator(efficient_loss(L))
MEstimator(::Type{L}) where {L<:LossFunction} = MEstimator(efficient_loss(L))

loss(e::MEstimator) = e.loss

function show(io::IO, obj::MEstimator)
    print(io, "M-Estimator($(obj.loss))")
end

# Forward all methods to the `loss` field
rho(e::MEstimator, r::Real) = rho(e.loss, r)
psi(e::MEstimator, r::Real) = psi(e.loss, r)
psider(e::MEstimator, r::Real) = psider(e.loss, r)
weight(e::MEstimator, r::Real) = weight(e.loss, r)
estimator_values(e::MEstimator, r::Real) = estimator_values(e.loss, r)
estimator_norm(e::MEstimator, args...) = estimator_norm(e.loss, args...)
estimator_bound(e::MEstimator) = estimator_bound(e.loss)

isbounded(e::MEstimator) = isbounded(e.loss)
isconvex(e::MEstimator) = isconvex(e.loss)

scale_estimate(est::E, res; kwargs...) where {E<:MEstimator} =
    scale_estimate(est.loss, res; kwargs...)

"`L1Estimator` is a shorthand name for `MEstimator{L1Loss}`. Using exact QuantileRegression should be prefered."
const L1Estimator = MEstimator{L1Loss}

"`L2Estimator` is a shorthand name for `MEstimator{L2Loss}`, the non-robust OLS."
const L2Estimator = MEstimator{L2Loss}



######
###   S-Estimators
######

"""
    SEstimator{L<:BoundedLossFunction} <: AbstractMEstimator

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
struct SEstimator{L<:BoundedLossFunction} <: AbstractMEstimator
    loss::L
end
SEstimator{L}() where {L<:BoundedLossFunction} = SEstimator(robust_loss(L))
SEstimator(::Type{L}) where {L<:BoundedLossFunction} = SEstimator(robust_loss(L))

loss(e::SEstimator) = e.loss

function show(io::IO, obj::SEstimator)
    print(io, "S-Estimator($(obj.loss))")
end

# Forward all methods to the `loss` field
rho(e::SEstimator, r::Real) = rho(e.loss, r)
psi(e::SEstimator, r::Real) = psi(e.loss, r)
psider(e::SEstimator, r::Real) = psider(e.loss, r)
weight(e::SEstimator, r::Real) = weight(e.loss, r)
estimator_values(e::SEstimator, r::Real) = estimator_values(e.loss, r)
estimator_norm(e::SEstimator, args...) = Inf
estimator_bound(e::SEstimator) = estimator_bound(e.loss)
isbounded(e::SEstimator) = true
isconvex(e::SEstimator) = false

scale_estimate(est::E, res; kwargs...) where {E<:SEstimator} =
    scale_estimate(est.loss, res; kwargs...)




######
###   MM-Estimators
######

"""
    MMEstimator{L1<:BoundedLossFunction, L2<:LossFunction} <: AbstractMEstimator

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
mutable struct MMEstimator{L1<:BoundedLossFunction,L2<:LossFunction} <: AbstractMEstimator
    "high breakdown point loss function"
    loss1::L1

    "high efficiency loss function"
    loss2::L2

    "S-Estimator phase indicator (or M-Estimator phase)"
    scaleest::Bool

    MMEstimator{L1,L2}(
        loss1::L1,
        loss2::L2,
        scaleest::Bool=true,
    ) where {L1<:BoundedLossFunction,L2<:LossFunction} = new(loss1, loss2, scaleest)
end
MMEstimator(
    loss1::L1,
    loss2::L2,
    scaleest::Bool,
) where {L1<:BoundedLossFunction,L2<:LossFunction} =
    MMEstimator{L1,L2}(loss1, loss2, scaleest)
MMEstimator(loss1::L1, loss2::L2) where {L1<:BoundedLossFunction,L2<:LossFunction} =
    MMEstimator{L1,L2}(loss1, loss2, true)
MMEstimator(::Type{L1}, ::Type{L2}) where {L1<:BoundedLossFunction,L2<:LossFunction} =
    MMEstimator(robust_loss(L1), efficient_loss(L2))
MMEstimator{L}() where {L<:BoundedLossFunction} =
    MMEstimator(robust_loss(L), efficient_loss(L))
MMEstimator(::Type{L}) where {L<:BoundedLossFunction} = MMEstimator{L}()

loss(e::MMEstimator) = e.scaleest ? e.loss1 : e.loss2

"MEstimator, set to S-Estimation phase"
set_SEstimator(e::MMEstimator) = (e.scaleest = true; e)

"MEstimator, set to M-Estimation phase"
set_MEstimator(e::MMEstimator) = (e.scaleest = false; e)

function show(io::IO, obj::MMEstimator)
    print(io, "MM-Estimator($(obj.loss1), $(obj.loss2))")
end

# Forward all methods to the selected loss
rho(E::MMEstimator, r::Real) = rho(loss(E), r)
psi(E::MMEstimator, r::Real) = psi(loss(E), r)
psider(E::MMEstimator, r::Real) = psider(loss(E), r)
weight(E::MMEstimator, r::Real) = weight(loss(E), r)
estimator_values(E::MMEstimator, r::Real) = estimator_values(loss(E), r)

# For these methods, only the SEstimator loss is useful,
# not the MEstimator, so E.loss1 is used instead of loss(E)
estimator_bound(E::MMEstimator) = estimator_bound(E.loss1)
# For these methods, only the MEstimator loss is useful,
# not the SEstimator, so E.loss2 is used instead of loss(E)
estimator_norm(E::MMEstimator, args...) = estimator_norm(E.loss2, args...)
isbounded(E::MMEstimator) = isbounded(E.loss2)
isconvex(E::MMEstimator) = isconvex(E.loss2)

scale_estimate(est::E, res; kwargs...) where {E<:MMEstimator} =
    scale_estimate(est.loss1, res; kwargs...)


######
###   τ-Estimators
######

"""
    TauEstimator{L1<:BoundedLossFunction, L2<:BoundedLossFunction} <: AbstractMEstimator

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
mutable struct TauEstimator{L1<:BoundedLossFunction,L2<:BoundedLossFunction} <:
               AbstractMEstimator
    "high breakdown point loss function"
    loss1::L1

    "high efficiency loss function"
    loss2::L2

    "loss weight"
    w::Float64

    TauEstimator{L1,L2}(
        l1::L1,
        l2::L2,
        w::Real=0.0,
    ) where {L1<:BoundedLossFunction,L2<:BoundedLossFunction} = new(l1, l2, float(w))
end
TauEstimator(
    l1::L1,
    l2::L2,
    args...,
) where {L1<:BoundedLossFunction,L2<:BoundedLossFunction} =
    TauEstimator{L1,L2}(l2, l2, args...)
# Warning: The tuning constant of the the efficient loss is NOT optimized for different loss functions
TauEstimator(
    ::Type{L1},
    ::Type{L2},
    args...,
) where {L1<:BoundedLossFunction,L2<:BoundedLossFunction} =
    TauEstimator(robust_loss(L1), efficient_loss(L2), args...)

# With the same loss function, the tuning constant of the the efficient loss is optimized
TauEstimator{L}() where {L<:BoundedLossFunction} =
    TauEstimator(robust_loss(L), L(estimator_tau_efficient_constant(L)))
TauEstimator(::Type{L}) where {L<:BoundedLossFunction} = TauEstimator{L}()

loss(e::TauEstimator) = CompositeLossFunction(e.loss1, e.loss2, e.w, 1)

function show(io::IO, obj::TauEstimator)
    print(io, "τ-Estimator($(obj.loss1), $(obj.loss2))")
end

"""
    tau_efficiency_tuning_constant(::Type{L1}, ::Type{L2}; eff::Real=0.95, c0::Real=1.0)
        where {L1<:BoundedLossFunction,L2<:BoundedLossFunction}

Compute the tuning constant that corresponds to a high breakdown point for the τ-estimator.
"""
function tau_efficiency_tuning_constant(
    ::Type{L1},
    ::Type{L2};
    eff::Real=0.95,
    c0::Real=1.0,
) where {L1<:BoundedLossFunction,L2<:BoundedLossFunction}
    loss1 = L1(estimator_high_breakdown_point_constant(L1))
    w1 = quadgk(x -> x * psi(loss1, x) * 2 * exp(-x^2 / 2) / √(2π), 0, Inf)[1]

    function τest(c)
        loss2 = L2(c)
        t2 = (tuning_constant(loss2))^2
        w2 = quadgk(
            x -> (2 * rho(loss2, x) * t2 - x * psi(loss2, x)) * 2 * exp(-x^2 / 2) / √(2π),
            0,
            Inf,
        )[1]
        TauEstimator{L1,L2}(loss1, loss2, w2 / w1)
    end

    lpsi(x, c) = psi(τest(c), x)
    lpsip(x, c) = psider(τest(c), x)

    I1(c) = quadgk(x -> (lpsi(x, c))^2 * 2 * exp(-x^2 / 2) / √(2π), 0, Inf)[1]
    I2(c) = quadgk(x -> lpsip(x, c) * 2 * exp(-x^2 / 2) / √(2π), 0, Inf)[1]
    fun_eff(c) = (I2(c))^2 / I1(c)
    copt = find_zero(c -> fun_eff(c) - eff, c0, Order1())
end
tau_efficiency_tuning_constant(::Type{L}; kwargs...) where {L<:BoundedLossFunction} =
    tau_efficiency_tuning_constant(L, L; kwargs...)

"The tuning constant associated to the loss that gives a robust τ-estimator."
estimator_tau_efficient_constant(::Type{GemanLoss}) = 5.632
estimator_tau_efficient_constant(::Type{WelschLoss}) = 4.043
estimator_tau_efficient_constant(::Type{TukeyLoss}) = 6.040
estimator_tau_efficient_constant(::Type{YohaiZamarLoss}) = 3.270
estimator_tau_efficient_constant(::Type{HardThresholdLoss}) = 1.0328  # Found after analytical simplifications
estimator_tau_efficient_constant(::Type{HampelLoss}) = 1.631


"""
    update_weight!(E::TauEstimator, res::AbstractArray{T}; wts::AbstractArray{T}=T[])

Update the weight between the two estimators of a τ-estimator using the scaled residual.
"""
function update_weight!(
    E::TauEstimator,
    res::AbstractArray{T};
    wts::AbstractArray{T}=T[],
) where {T<:AbstractFloat}
    c² = (tuning_constant(E.loss2))^2
    E.w = if length(wts) == length(res)
        w2 = sum(@.(wts * (2 * rho(E.loss2, res) * c² - res * psi(E.loss2, res))))
        w1 = sum(@.(wts * res * psi(E.loss1, res)))
        w2 / w1
    else
        w2 = sum(r -> 2 * rho(E.loss2, r) * c² - r * psi(E.loss2, r), res)
        w1 = sum(r -> r * psi(E.loss1, r), res)
        w2 / w1
    end
    E
end
update_weight!(E::TauEstimator, res::AbstractArray; wts::AbstractArray=[]) =
    update_weight!(E, float(res); wts=float(wts))
update_weight!(E::TauEstimator, w::Real) = (E.w = w; E)

# Forward all methods to the `loss` fields
rho(E::TauEstimator, r::Real) =
    E.w * rho(E.loss1, r) * (tuning_constant(E.loss1))^2 +
    rho(E.loss2, r) * (tuning_constant(E.loss2))^2
psi(E::TauEstimator, r::Real) = E.w * psi(E.loss1, r) + psi(E.loss2, r)
psider(E::TauEstimator, r::Real) = E.w * psider(E.loss1, r) + psider(E.loss2, r)
weight(E::TauEstimator, r::Real) = E.w * weight(E.loss1, r) + weight(E.loss2, r)
function estimator_values(E::TauEstimator, r::Real)
    vals1 = estimator_values(E.loss1, r)
    vals2 = estimator_values(E.loss2, r)
    c12, c22 = (tuning_constant(E.loss1))^2, (tuning_constant(E.loss2))^2
    return (
        E.w * vals1[1] * c12 + vals2[1] * c22,
        E.w * vals1[2] + vals2[3],
        E.w * vals1[3] + vals2[3],
    )
end
estimator_norm(E::TauEstimator, args...) = Inf
estimator_bound(E::TauEstimator) = estimator_bound(E.loss1)
isbounded(E::TauEstimator) = true
isconvex(E::TauEstimator) = false

scale_estimate(est::E, res; kwargs...) where {E<:TauEstimator} =
    scale_estimate(est.loss1, res; kwargs...)

"""
    tau_scale_estimate!(E::TauEstimator, res::AbstractArray{T}, σ::Real, sqr::Bool=false;
                        wts::AbstractArray{T}=T[], bound::AbstractFloat=0.5) where {T<:AbstractFloat}

The τ-scale estimate, where `σ` is the scale estimate from the robust M-scale.
If `sqr` is true, return the squared value.
"""
function tau_scale_estimate(
    est::TauEstimator,
    res::AbstractArray{T},
    σ::Real,
    sqr::Bool=false;
    wts::AbstractArray=[],
    bound::AbstractFloat=0.5,
) where {T<:AbstractFloat}
    t = if length(wts) == length(res)
        mean(mscale_loss.(Ref(est.loss2), res ./ σ), weights(wts)) / bound
    else
        mean(mscale_loss.(Ref(est.loss2), res ./ σ)) / bound
    end

    return sqr ? σ * √t : σ^2 * t
end




######
###   MQuantile Estimators
######
"""
    quantile_weight(τ::Real, r::Real)

Wrapper function to compute quantile-like loss function.
"""
quantile_weight(τ::Real, r::Real) = oftype(r, 2 * ifelse(r > 0, τ, 1 - τ))

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
function GeneralizedQuantileEstimator(l::L, τ::Real=0.5) where {L<:LossFunction}
    (0 < τ < 1) ||
        throw(DomainError(τ, "quantile should be a number between 0 and 1 excluded"))
    GeneralizedQuantileEstimator{L}(l, float(τ))
end
GeneralizedQuantileEstimator{L}(τ::Real=0.5) where {L<:LossFunction} =
    GeneralizedQuantileEstimator(L(), float(τ))

function ==(
    e1::GeneralizedQuantileEstimator{L1},
    e2::GeneralizedQuantileEstimator{L2},
) where {L1<:LossFunction,L2<:LossFunction}
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
        (0 < v < 1) ||
            throw(DomainError(v, "quantile should be a number between 0 and 1 excluded"))
        r.τ = float(v)
    else
        setfield!(r, s, v)
    end
end

Base.propertynames(r::GeneralizedQuantileEstimator, private=false) =
    (:loss, :τ, :tau, :q, :quantile)


# Forward all methods to the `loss` field
rho(e::GeneralizedQuantileEstimator, r::Real) = quantile_weight(e.τ, r) * rho(e.loss, r)
psi(e::GeneralizedQuantileEstimator, r::Real) = quantile_weight(e.τ, r) * psi(e.loss, r)
psider(e::GeneralizedQuantileEstimator, r::Real) =
    quantile_weight(e.τ, r) * psider(e.loss, r)
weight(e::GeneralizedQuantileEstimator, r::Real) =
    quantile_weight(e.τ, r) * weight(e.loss, r)
function estimator_values(e::GeneralizedQuantileEstimator, r::Real)
    w = quantile_weight(e.τ, r)
    vals = estimator_values(e.loss, r)
    Tuple([x * w for x in vals])
end
estimator_norm(e::GeneralizedQuantileEstimator, args...) = estimator_norm(e.loss, args...)
estimator_bound(e::GeneralizedQuantileEstimator) = estimator_bound(e.loss)
isbounded(e::GeneralizedQuantileEstimator) = isbounded(e.loss)
isconvex(e::GeneralizedQuantileEstimator) = isconvex(e.loss)

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

const UnionL1 = Union{L1Estimator,GeneralizedQuantileEstimator{L1Loss}}

const UnionMEstimator = Union{MEstimator,GeneralizedQuantileEstimator}
