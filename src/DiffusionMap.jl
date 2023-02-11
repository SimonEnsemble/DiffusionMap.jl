module DiffusionMap
using LinearAlgebra

export normalize_to_stochastic_matrix!, diffusion_map

"""
    normalize_to_stochastic_matrix!(P)

normalize a kernel matrix `p` so that rows sum to one.
"""
function normalize_to_stochastic_matrix!(P::Matrix{Float64})
    # make sure rows sum to one
    # this is equivalent to P = D⁻¹ * P
    #    with > d = vec(sum(P, dims=1))
    #         > D⁻¹ = diagm(1 ./ d)
    for i = 1:size(P)[1]
        P[i, :] = P[i, :] / sum(P[i, :])
    end
end

"""
    diffusion_map(P, d; t=1)
    diffusion_map(X, kernel, d, t=1)

compute diffusion map. 

two call signatures.
* the data matrix `X` is passed in. examples are in the columns.
* the right-stochastic matrix `P` is passed in. (eg. for precomputed kernel matrix)

# arguments
* `d`: dim
* `t`: # of steps
"""
function diffusion_map(P::Matrix{Float64}, d::Int; t::Int=1)
    if ! all(sum.(eachrow(P)) .≈ 1.0)
        error("not a right-stochastic matrix. use normalize_to_stochastic_matrix.")
    end
    if ! all(P .> 0.0)
        error("should be positive entries...")
    end

    # eigen-decomposition of the stochastic matrix
    eigen_decomp = eigen(P)

    if ! (abs(maximum(eigen_decomp.values) - 1.0) < 0.0001)
        error("largest eigenvalue should be 1.0")
    end

    # sort eigenvalues, highest to lowest
    # skip the first eigenvalue
    idx = sortperm(eigen_decomp.values, rev=true)[2:end]

    # get first d eigenvalues and vectors. scale eigenvectors.
    λs = eigen_decomp.values[idx][1:d]
    Vs = eigen_decomp.vectors[:, idx][:, 1:d] * diagm(λs .^ t)
    return Vs
end

function diffusion_map(X::Matrix{Float64}, kernel::Function, d::Int; t::Int=1)
    # compute Laplacian matrix
    P = pairwise(kernel, eachcol(X), symmetric=true)
    normalize_to_stochastic_matrix!(P)

    return diffusion_map(P, d; t=t)
end

end
