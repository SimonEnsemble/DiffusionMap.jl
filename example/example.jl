### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ cedad242-4983-11eb-2f32-3d405f151b77
begin
    using DiffusionMap
    using CairoMakie, ColorSchemes, ManifoldLearning, PlutoUI, StatsBase
end

# ╔═╡ 1996b541-e783-4b3a-9f7a-aca380d1047f
TableOfContents()

# ╔═╡ f306c9dc-cfd5-4fe0-b4cb-9a9d2da7281c
import AlgebraOfGraphics as aog

# ╔═╡ bad179ee-cef7-444b-b7cb-15053ac62959
begin
    aog.set_aog_theme!(; fonts=[aog.firasans("Light"), aog.firasans("Light")])
    update_theme!(;
        fontsize=20,
        linewidth=4,
        markersize=14,
        titlefont=aog.firasans("Light"),
        resolution=(500, 380)
    )
end

# ╔═╡ 61f9fccf-65af-45de-b232-f32506fdb75f
md"# viz Gaussian kernel"

# ╔═╡ 9b487b40-63c1-4a9b-b1f1-ef7703dd4dcc
kernel(xᵢ, xⱼ) = gaussian_kernel(xᵢ, xⱼ, 0.5)

# ╔═╡ 3c5dd85c-d21c-43f9-a0f6-fe3873a5268e
function viz_gaussian_kernel()
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="||xᵢ-xⱼ|| / ℓ²", ylabel="k(xᵢ, xⱼ)")
    vlines!(0.0; color="lightgray", linewidth=1)
    hlines!(0.0; color="lightgray", linewidth=1)
    rs = range(0.0, 4.0; length=100)
    lines!(rs, exp.(-rs .^ 2))
    return fig
end

# ╔═╡ fc635aaf-54cb-43e9-b338-8c0f9510b251
md"## helpers"

# ╔═╡ c64b57a4-ac19-461b-ba0f-4a3bfe583db2
function viz_data(X::Matrix{Float64}, name::String)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x₁", ylabel="x₂", aspect=DataAspect())
    scatter!(X[1, :], X[2, :]; color="black")
    save("raw_data_$name.pdf", fig)
    return fig
end

# ╔═╡ 659956e9-aaa6-437f-85c3-c25e01457a28
function viz_graph(X::Matrix{Float64}, kernel::Function, name::String)
    K = pairwise(kernel, eachcol(X); symmetric=true)

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x₁", ylabel="x₂", aspect=DataAspect())
    for i in 1:size(X)[2]
        for j in (i + 1):size(X)[2]
            w = K[i, j]
            lines!(
                [X[1, i], X[1, j]],
                [X[2, i], X[2, j]];
                linewidth=0.5,
                color=(get(reverse(ColorSchemes.grays), w), w)
            )
        end
    end
    scatter!(X[1, :], X[2, :]; color="black")
    save("graph_rep_$name.pdf", fig)
    return fig
end

# ╔═╡ 310ba8ae-7443-44a5-b77d-168336af39bd
function color_points(X::Matrix{Float64}, x̂::Vector{Float64}, name::String)
    cmap = ColorSchemes.terrain

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x₁", ylabel="x₂", title=name, aspect=DataAspect())
    sp = scatter!(X[1, :], X[2, :]; color=x̂, colormap=cmap, strokewidth=1)
    Colorbar(fig[1, 2], sp; label="latent dim")
    save("dim_reduction_$name.pdf", fig)
    return fig
end

# ╔═╡ 8e01ddd4-27c9-4f5c-8d7f-abd4f2be4f82
md"# S-curve

### generate data.
"

# ╔═╡ c793e6b0-4983-11eb-29ae-3366c7d31e84
begin
    nb_data = 125
    _X, _ = ManifoldLearning.scurve(nb_data, 0.1)

    x₁ = _X[3, :]
    x₂ = _X[1, :]

    X = collect(hcat(x₁, x₂)')
end

# ╔═╡ 3172a389-10c7-45e0-a556-a8c8f1465418
viz_data(X, "S_curve")

# ╔═╡ ecd00826-6e79-47b1-ab60-46ac3e1bfef9
md"### translate data to graph"

# ╔═╡ 48aa38c3-9876-4f7c-8e4f-e22dec5104fb
viz_gaussian_kernel()

# ╔═╡ a064c35e-0e4f-418e-a329-e747daae264e
viz_graph(X, kernel, "S_curve")

# ╔═╡ 17a20c5b-7da4-4982-9b7a-21ba3c3b8060
md"### diff map (success)"

# ╔═╡ 7fbe0950-27de-4969-9962-16080e6908ff
x̂ = diffusion_map(X, kernel, 1; cuda=true)[:]

# ╔═╡ 4311f687-824c-42d4-b0a0-b08945460e76
color_points(X, x̂, "diff map")

# ╔═╡ 1349c4d0-d6af-4154-b139-defdaec008a1
md"### PCA (fails)"

# ╔═╡ 1ec637b9-2279-4b54-b5a5-08445f229ab4
x̂_pca = pca(collect(X), 1; cuda=true)[:]

# ╔═╡ 2aaea099-6f33-4260-bf40-f82644e24eae
color_points(X, x̂_pca, "PCA")

# ╔═╡ bc368089-d8f8-43e6-8933-344f71ab66d7
md"# swiss roll"

# ╔═╡ 5f986a24-784b-4259-a5d7-4f6a86b5e6fd
_X_roll, _ = ManifoldLearning.swiss_roll(125, 0.3)

# ╔═╡ d8863ed7-8430-4135-a3a2-3efb8f54d1c0
X_roll = _X_roll[[1, 3], :] # make 2D

# ╔═╡ b682a6af-8b0b-41fc-914c-55f871dbdf4f
roll_kernel(xᵢ, xⱼ) = gaussian_kernel(xᵢ, xⱼ, 2.0)

# ╔═╡ 13ff129b-c936-451d-9908-80e840103450
x̂_roll = diffusion_map(X_roll, roll_kernel, 1; cuda=true)[:]

# ╔═╡ 62bf2efc-9614-4fbf-bb42-29e33f7d34a1
viz_graph(X_roll, roll_kernel, "roll")

# ╔═╡ 8fb74c20-eebb-40d5-a3aa-da192398bcf3
color_points(X_roll, x̂_roll, "diff map roll")

# ╔═╡ Cell order:
# ╠═cedad242-4983-11eb-2f32-3d405f151b77
# ╠═1996b541-e783-4b3a-9f7a-aca380d1047f
# ╠═f306c9dc-cfd5-4fe0-b4cb-9a9d2da7281c
# ╠═bad179ee-cef7-444b-b7cb-15053ac62959
# ╟─61f9fccf-65af-45de-b232-f32506fdb75f
# ╠═9b487b40-63c1-4a9b-b1f1-ef7703dd4dcc
# ╠═3c5dd85c-d21c-43f9-a0f6-fe3873a5268e
# ╟─fc635aaf-54cb-43e9-b338-8c0f9510b251
# ╠═c64b57a4-ac19-461b-ba0f-4a3bfe583db2
# ╠═659956e9-aaa6-437f-85c3-c25e01457a28
# ╠═310ba8ae-7443-44a5-b77d-168336af39bd
# ╟─8e01ddd4-27c9-4f5c-8d7f-abd4f2be4f82
# ╠═c793e6b0-4983-11eb-29ae-3366c7d31e84
# ╠═3172a389-10c7-45e0-a556-a8c8f1465418
# ╟─ecd00826-6e79-47b1-ab60-46ac3e1bfef9
# ╠═48aa38c3-9876-4f7c-8e4f-e22dec5104fb
# ╠═a064c35e-0e4f-418e-a329-e747daae264e
# ╟─17a20c5b-7da4-4982-9b7a-21ba3c3b8060
# ╠═7fbe0950-27de-4969-9962-16080e6908ff
# ╠═4311f687-824c-42d4-b0a0-b08945460e76
# ╟─1349c4d0-d6af-4154-b139-defdaec008a1
# ╠═1ec637b9-2279-4b54-b5a5-08445f229ab4
# ╠═2aaea099-6f33-4260-bf40-f82644e24eae
# ╟─bc368089-d8f8-43e6-8933-344f71ab66d7
# ╠═5f986a24-784b-4259-a5d7-4f6a86b5e6fd
# ╠═d8863ed7-8430-4135-a3a2-3efb8f54d1c0
# ╠═b682a6af-8b0b-41fc-914c-55f871dbdf4f
# ╠═13ff129b-c936-451d-9908-80e840103450
# ╠═62bf2efc-9614-4fbf-bb42-29e33f7d34a1
# ╠═8fb74c20-eebb-40d5-a3aa-da192398bcf3
