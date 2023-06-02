

seed = 1234
rng = MersenneTwister(seed)

function gendata(
    rng::AbstractRNG,
    ::Type{T},
    N::Integer,
    p::Integer,
    nonzero::Integer,
    rho::Real,
    snr::Real = 3,
    alternate::Bool=true,
) where {T<:AbstractFloat}
    if nonzero > 0
        beta = nonzero:-1:1
        if alternate
            beta = [(nonzero - i + 1) * (i % 2 == 0 ? 1 : -1) for i in 1:nonzero]
        else
            beta = nonzero:-1:1
        end
        beta = vcat(beta, zeros(T, p-nonzero))
    else
        beta = zeros(T, p)
    end
    
    X = randn(rng, N, p)
    if rho > 0
        z = randn(N)
        X .+= sqrt(rho/(1-rho)) * z
        X *= sqrt(1-rho)
    end
    
    ssd = sqrt((1-rho)*sum(abs2, beta) + rho*sum(abs, beta))
    nsd = ssd / snr
    
    y = X * beta + nsd * randn(N)
    return X, y, beta
end


X3, y3, β03 = gendata(rng, Float64, 100, 1000, 30, 0.3)

loss1 = L2Loss()
est1 = MEstimator{L2Loss}()
λ = 1.0
pen1 = L1Penalty(λ)
pen2 = SquaredL2Penalty(λ)

@testset "Rank deficient X: penalty ($(pen))" for (pen, methods) in zip(
    (nothing, pen1, pen2), (nopen_methods, pen_methods, pen_methods)
)
    def_rtol = isnothing(pen) ? 1e-6 : 1e-4

    @testset "solver method $(method)" for method in methods
        rtol = def_rtol
        if method === :fista
            rtol = 1e-1
        end

        ### RobustLinearModel
        name = "rlm(X3, y3, L2Loss, $(pen); method=$(method))"
        VERBOSE && println("\n\t\u25CF $(name)")

        if isnothing(pen)
            if method in (:chol, :cholesky, :auto)
                @test_throws Exception rlm(X3, y3, est1; method=method, initial_scale=1, dropcollinear=false)
            end
            m1 = rlm(X3, y3, est1; method=method, initial_scale=1, dropcollinear=true)
        else
            m1 = rlm(X3, y3, pen; method=method, initial_scale=1, maxiter=1000, dropcollinear=true)
        end

        # Test printing the model
        @test_nowarn show(devnull, m1)

        # interface
        @testset "method: $(f)" for f in interface_methods
            @test_nowarn f(m1)
        end

        ### IPODRegression
        name = "ipod(X3, y3, L2Loss, $(pen); method=$(method))"
        VERBOSE && println("\n\t\u25CF $(name)")

        if isnothing(pen) && method in (:chol, :cholesky, :auto)
            @test_throws Exception ipod(X3, y3, loss1, pen; method=method, initial_scale=1, dropcollinear=false)
        end
        m2 = ipod(X3, y3, loss1, pen; method=method, initial_scale=1, maxiter=1000, dropcollinear=true)

        # Test printing the model
        @test_nowarn show(devnull, m2)

        # interface
        @testset "method: $(f)" for f in interface_methods
            if f == workingweights
                # Not defined for IPODRegression
                continue
            end
            @test_nowarn f(m2)
        end
    end
end
