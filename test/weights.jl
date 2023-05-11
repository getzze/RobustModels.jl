
seed = 123987

rng = MersenneTwister(seed)

n = 100
r = 30
σ = 10
xlin = collect(1.0:n)
Xlin = hcat(ones(Float64, n), xlin)
β1lin = [0, -5]
β2lin = [20, -1]

ylin = vcat(Xlin[1:r, :] * β1lin, Xlin[r+1:end, :] * β2lin)
ylin += σ * randn(rng, n)

wlin = float((1:n) .<= r)
wwlin = 1 .- wlin


estimator_list = (
    MEstimator{L2Loss}(),
    MEstimator{HuberLoss}(),
    MEstimator{TukeyLoss}(),
    MMEstimator{TukeyLoss}(),
    TauEstimator{TukeyLoss}(),
)

@testset "weights: $(est)" for est in estimator_list
    mtot = rlm(Xlin, ylin, est)

    m1 = rlm(Xlin[1:r, :], ylin[1:r], est)
    m1w = rlm(Xlin, ylin, est; wts=wlin)

    m2 = rlm(Xlin[r+1:end, :], ylin[r+1:end], est)
    m2w = rlm(Xlin, ylin, est; wts=wwlin)

    # Check identical values
    @testset "$(func)" for func in interface_methods
        f1 = func(m1)
        f1w = func(m1w)
        f2 = func(m2)
        f2w = func(m2w)
        if func == weights
            @test isempty(f1)
            @test isempty(f2)
            @test !isempty(f1w)
            @test !isempty(f2w)
        elseif f1 isa AbstractArray && size(f1) != size(f1w)
            if ndims(f1) == 1
                @test f1 ≈ f1w[1:r]
                @test f2 ≈ f2w[r+1:end]
            elseif ndims(f1) == 2
                if size(f1, 2) == size(f1w, 2)
                    @test f1 ≈ f1w[1:r, :]
                    @test f2 ≈ f2w[r+1:end, :]
                else
                    @test f1 ≈ f1w[1:r, 1:r]
                    @test f2 ≈ f2w[r+1:end, r+1:end]
                end
            end
        else
            @test f1 ≈ f1w
            @test f2 ≈ f2w
        end
    end
end

@testset "weights: QuantileRegression)" begin
    mtot = quantreg(Xlin, ylin)

    m1 = quantreg(Xlin[1:r, :], ylin[1:r])
    m1w = quantreg(Xlin, ylin; wts=wlin)

    m2 = quantreg(Xlin[r+1:end, :], ylin[r+1:end])
    m2w = quantreg(Xlin, ylin; wts=wwlin)

    # Check identical values
    @testset "$(func)" for func in interface_methods
        f1 = func(m1)
        f1w = func(m1w)
        f2 = func(m2)
        f2w = func(m2w)
        if func == weights
            @test isempty(f1)
            @test isempty(f2)
            @test !isempty(f1w)
            @test !isempty(f2w)
        elseif f1 isa AbstractArray && size(f1) != size(f1w)
            if ndims(f1) == 1
                @test f1 ≈ f1w[1:r]
                @test f2 ≈ f2w[r+1:end]
            elseif ndims(f1) == 2
                if size(f1, 2) == size(f1w, 2)
                    @test f1 ≈ f1w[1:r, :]
                    @test f2 ≈ f2w[r+1:end, :]
                else
                    @test f1 ≈ f1w[1:r, 1:r]
                    @test f2 ≈ f2w[r+1:end, r+1:end]
                end
            end
        else
            @test f1 ≈ f1w
            @test f2 ≈ f2w
        end
    end
end
