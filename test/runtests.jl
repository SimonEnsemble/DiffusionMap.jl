module Test_DiffusionMap
using DiffusionMap, LinearAlgebra, Test

@testset "normalize_to_stochastic_matrix!" begin
    # generate random symmetric matrix
    P = rand(20, 20)
    P = P + P'
    # normalize
    normalize_to_stochastic_matrix!(P)
    # test rows sum to 1
    @test all(sum.(eachrow(P)) .≈ 1)
    # make an assymmetric matrix
    P = P + rand(20, 20)
    # test asymmetry is detected
    @test_throws ErrorException normalize_to_stochastic_matrix!(P)
end

@testset "diffusion_map" verbose=true begin
    # function signature 1: diffusion_map(P, d; t=1)
    @testset "matrix P" begin
        # test input validation: non-square matrix P
        @test_throws ErrorException diffusion_map(rand(20, 10), 2)
        # test input validation: rows of P don't sum to 1
        @test_throws ErrorException diffusion_map(rand(20, 20), 2)
        # test input validation: negative values in P
        @test_throws ErrorException diffusion_map(rand(20, 20) - rand(20, 20), 2)
        # test a trivial case
        D = zeros(10, 2)
        D[2, 1] = D[3, 2] = 1
        @test diffusion_map(diagm(ones(10)), 2) == D
    end

    # function signature 2: diffusion_map(X, kernel, d, t=1)
    @testset "matrix X and kernel" begin
        kernel(xᵢ, xⱼ) = gaussian_kernel(xᵢ, xⱼ, 0.5)
        @test isapprox(sum(diffusion_map(ones(20, 20), kernel, 2)), 0; atol=1e-9)
    end
end

@testset "pca" begin
    # test a trivial case
    @test pca(ones(10, 10), 2) == zeros(10, 2)
end

end
