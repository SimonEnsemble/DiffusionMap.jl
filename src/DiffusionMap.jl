module DiffusionMap
using CUDA, LinearAlgebra, StatsBase

export normalize_to_stochastic_matrix!, diffusion_map, gaussian_kernel, pca

const BANNER = String(read(joinpath(dirname(pathof(DiffusionMap)), "banner.txt")))
banner() = println(BANNER)

function gaussian_kernel(xⱼ, xᵢ, ℓ::Real)
    r² = sum((xᵢ - xⱼ) .^ 2)
    return exp(-r² / ℓ ^ 2)
end

"""
    normalize_to_stochastic_matrix!(P, check_symmetry=true)

normalize a kernel matrix `P` so that rows sum to one.

checks for symmetry.
"""
function normalize_to_stochastic_matrix!(P::Matrix{<: Real}; check_symmetry::Bool=true)
    check_symmetry && @assert issymmetric(P) "Kernel matrix must be symmetric."
    @assert all(P .>= 0.0) "Kernel matrix elements must be non-negative."
    # make sure rows sum to one
    # this is equivalent to P = D⁻¹ * P
    #    with > d = vec(sum(P, dims=1))
    #         > D⁻¹ = diagm(1 ./ d)
    for i = 1:size(P)[1]
        P[i, :] ./= sum(P[i, :])
    end
end

"""
    diffusion_map(P, d; t=1)
    diffusion_map(X, kernel, d; t=1)

compute diffusion map. 

two call signatures:
* the data matrix `X` is passed in. examples are in the columns.
* the right-stochastic matrix `P` is passed in. (eg. for a precomputed kernel matrix)

# arguments
* `d`: dim
* `t`: # of steps

# example
```julia
# define kernel
kernel(xᵢ, xⱼ) = gaussian_kernel(xᵢ, xⱼ, 0.5)

# data matrix (100 data pts, 2D vectors)
X = rand(2, 100)

# diffusion map to 1D
X̂ = diff_map(X, kernel, 1)
```
"""
function diffusion_map(P::Matrix{<: Real}, d::Int; t::Int=1, cuda::Bool=false)::Matrix{Float64}
    if size(P)[1] ≠ size(P)[2]
        error("P is not square.")
    end
    if ! all(sum.(eachrow(P)) .≈ 1.0)
        error("P is not right-stochastic. call `normalize_to_stochastic_matrix!` first.")
    end
    if ! all(P .>= 0.0)
        error("P contains negative values.")
    end

    # if available, use CUDA (compute capability ≥ 3.5)
    if cuda && capability(device()) ≥ v"3.5.0"
        P = cu(P)
    end

    # eigen-decomposition of the stochastic matrix
    sv_decomp = svd(P)
	eigenvalues = sv_decomp.S

    @assert (maximum(abs.(eigenvalues)) - 1.0) < 0.01 "largest eigenvalue should be 1.0"

    # eigenvalues should all be real numbers, but numerical imprecision can promote
    # the results to "complex" numbers with imaginary components of 0
    for (i, ev) in enumerate(eigenvalues)
        if isa(ev, Complex)
            @assert isapprox(imag(ev), 0; atol=1e-6)
            eigenvalues[i] = real(ev)
        end
    end

    # sort eigenvalues, highest to lowest
    # skip the first eigenvalue
    idx = sortperm(Float64.(eigenvalues), rev=true)[2:end]

    # get first d eigenvalues and vectors. scale eigenvectors.
    λs = eigenvalues[idx][1:d]
    Vs = sv_decomp.V[:, idx][:, 1:d] * diagm(λs .^ t)
    return Vs
end

function diffusion_map(X::Matrix{<: Real}, kernel::Function, 
                       d::Int; t::Int=1, verbose::Bool=true)
    if verbose
        println("# features: ", size(X)[1])
        println("# examples: ", size(X)[2])
    end

    # compute Laplacian matrix
    P = pairwise(kernel, eachcol(X), symmetric=true)
    normalize_to_stochastic_matrix!(P)

    return diffusion_map(P, d; t=t)
end

function pca(X::Matrix{<: Real}, d::Int; verbose::Bool=true, cuda::Bool=false)
    if verbose
        println("# features: ", size(X)[1])
        println("# examples: ", size(X)[2])
    end

    # center
    X̂ = deepcopy(X)
    for f = 1:size(X)[1]
        X̂[f, :] = X[f, :] .- mean(X[f, :])
    end

    # if available, use CUDA (compute capability ≥ 3.5)
    if cuda && capability(device()) ≥ v"3.5.0"
        X̂ = cu(X̂)
    end

    the_svd = svd(X̂)
    return the_svd.V[:, 1:d] * diagm(the_svd.S[1:d])
end

end
