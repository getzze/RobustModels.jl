
using RobustModels:
    cost,
    proximal,
    proximal!,
    isconcrete,
    concrete,
    RangedPenalties,
    End,
    nothing  # stopper

λ = 10_000.0

seed = 1234
rng = MersenneTwister(seed)

t = 10 * randn(rng, 30)


@testset "Methods penalty functions: $(name)" for name in ("No", penalties...)
    tt = copy(t)

    typepenalty = getproperty(RobustModels, Symbol(name * "Penalty"))

    pen = typepenalty(λ)
    penpos = typepenalty(λ, true)

    @test_nowarn println(pen)

    # Cost
    P0 = cost(pen, zeros(Float64, 4))
    P1 = cost(pen, ones(Float64, 4))
    @test P1 >= P0

    # Proximal operator
    copy!(tt, t)
    @test_nowarn proximal!(pen, tt, t, 1.0)
    @test tt ≈ proximal(pen, t, 1.0)

    # Shrinkage
    @test all(@.(abs(tt) <= abs(t)))

    # Single index update
    copy!(tt, t)
    proximal!(pen, tt, 1, t, 1.0)
    @test abs(tt[1]) <= abs(t[1])
    @test all(tt[2:end] .== t[2:end])

    # Nonnegative
    if name != "No"
        tt = proximal(penpos, t, 1.0)
        # Get only non-negative values
        @test all(>=(0), tt)

        # Negative values are set to 0
        nonpos_indices = findall(<=(0), t)
        @test all(iszero, tt[nonpos_indices])
    end

    # Concrete
    new_pen = concrete(pen, length(t))
    @test new_pen == pen

    @testset "RangedPenalties" begin
        n = length(t)
        seppen = RangedPenalties([2:End()], [pen])
        concpen = RangedPenalties([2:n], [pen])

        # Not concrete
        @test !isconcrete(seppen)

        copy!(tt, t)
        @test_throws Exception cost(seppen, t)
        @test_throws Exception proximal(seppen, t, 1.0)
        @test_throws Exception proximal!(seppen, tt, t, 1.0)
        @test_throws Exception proximal!(seppen, tt, 2, t, 1.0)

        # Concrete
        new_pen = concrete(seppen, n)
        @test new_pen == concpen
        @test isconcrete(new_pen)

        # Proximal operator
        copy!(tt, t)
        @test_nowarn proximal!(new_pen, tt, t, 1.0)
        @test tt[1] == t[1]
        @test all(abs.(tt[2:end]) .<= abs.(t[2:end]))

        @test_nowarn proximal!(new_pen, tt, 2, t, 1.0)
        @test_throws Exception proximal!(new_pen, tt, 0, t, 1.0)
        @test_throws Exception proximal!(new_pen, tt, length(t) + 1, t, 1.0)

        # Split ranges
        @test pen == concrete(pen, n, nothing)
        @test concpen == concrete(pen, n, 1)
        for j in (n, 2)
            new_pen = concrete(pen, n, j)
            @test length(new_pen.ranges) == 1
            @test length(new_pen.penalties) == 1
            @test Set(only(new_pen.ranges)) == Set(filter(!=(j), 1:n))
            @test only(new_pen.penalties) == pen
        end
    end
end


@testset "M-estimator with penalty $(name)" for name in penalties
    typepenalty = getproperty(RobustModels, Symbol(name * "Penalty"))

    pen = typepenalty(λ)
    penpos = typepenalty(λ, true)
    kwargs = (; maxiter=1000)

    m1 = fit(RobustLinearModel, form, data, pen; verbose=false, kwargs...)
    @test all(isfinite.(coef(m1)))

    # make sure that it is not a TableRegressionModel
    @test !isa(m1, TableRegressionModel)

    @testset "method: $(method)" for method in pen_methods
        if method === :admm; continue end
        m2 = fit(RobustLinearModel, form, data, pen; method=method, kwargs...)

        @test isapprox(coef(m1), coef(m2); rtol=1e-2)
    end

    m3 = fit(RobustLinearModel, form, data, penpos; verbose=false, kwargs...)
    j = hasintercept(m1) ? 2 : 1
    @test all(>=(0), coef(m3)[j:end])

    # refit
    β1 = copy(coef(m1))
    refit!(m1)
    @test all(coef(m1) .== β1)
end
