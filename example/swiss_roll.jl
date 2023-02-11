### A Pluto.jl notebook ###
# v0.19.20

using Markdown
using InteractiveUtils

# ╔═╡ cedad242-4983-11eb-2f32-3d405f151b77
begin
	import Pkg; Pkg.activate()
	push!(LOAD_PATH, "../src")
	using Revise
	using DiffusionMap
	using ManifoldLearning, CairoMakie, Statistics, JLD2, Random, StatsBase, LinearAlgebra, ColorSchemes, PlutoUI, CSV
end

# ╔═╡ 1996b541-e783-4b3a-9f7a-aca380d1047f
TableOfContents()

# ╔═╡ 1923b9da-6f21-4675-a7bf-6703f3165118
import ScikitLearn

# ╔═╡ 8049574b-7fa8-49ec-a022-7b879e9ea38b
ScikitLearn.@sk_import decomposition: PCA

# ╔═╡ f306c9dc-cfd5-4fe0-b4cb-9a9d2da7281c
import AlgebraOfGraphics as aog

# ╔═╡ bad179ee-cef7-444b-b7cb-15053ac62959
begin
	aog.set_aog_theme!(fonts=[aog.firasans("Light"), aog.firasans("Light")])
	update_theme!(
		fontsize=20, 
		linewidth=4,
		markersize=14,
		titlefont=aog.firasans("Light"),
		resolution=(500, 380)
	)
end

# ╔═╡ 8e01ddd4-27c9-4f5c-8d7f-abd4f2be4f82
md"# generate data"

# ╔═╡ c793e6b0-4983-11eb-29ae-3366c7d31e84
begin
	nb_data = 125
	_X, _ = ManifoldLearning.scurve(nb_data, 0.1)
	
	x₁ = _X[3, :]
	x₂ = _X[1, :]

	X = collect(hcat(x₁, x₂)')
end

# ╔═╡ c64b57a4-ac19-461b-ba0f-4a3bfe583db2
begin
	fig = Figure()
	ax  = Axis(fig[1, 1], xlabel="x₁", ylabel="x₂")
	scatter!(x₁, x₂, color="black")
	# save("raw_data.pdf", fig)
	fig
end

# ╔═╡ ecd00826-6e79-47b1-ab60-46ac3e1bfef9
md"# translate data to graph"

# ╔═╡ 9b487b40-63c1-4a9b-b1f1-ef7703dd4dcc
function kernel(xⱼ, xᵢ; ℓ=0.5)
	r² = sum((xᵢ - xⱼ) .^ 2)
	return exp(-r² / ℓ ^ 2)
end

# ╔═╡ 3c5dd85c-d21c-43f9-a0f6-fe3873a5268e
function viz_kernel()
	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="||xᵢ-xⱼ|| / ℓ²", ylabel="k(xᵢ, xⱼ)")
	vlines!(0.0, color="lightgray", linewidth=1)
	hlines!(0.0, color="lightgray", linewidth=1)
	rs = range(0.0, 4.0, length=100)
	lines!(rs, exp.(-rs.^2))
	fig
end

# ╔═╡ 48aa38c3-9876-4f7c-8e4f-e22dec5104fb
viz_kernel()

# ╔═╡ 659956e9-aaa6-437f-85c3-c25e01457a28
function viz_graph(X::Matrix{Float64}, kernel::Function)
	K = pairwise(kernel, eachcol(X), symmetric=true)
	
	fig = Figure()
	ax  = Axis(fig[1, 1], xlabel="x₁", ylabel="x₂", aspect=DataAspect())
	for i = 1:nb_data
		for j = (i+1):nb_data
			w = K[i, j]
			lines!([X[1, i], X[1, j]], [X[2, i], X[2, j]], 
				linewidth=0.5, 
				color=(get(reverse(ColorSchemes.grays), w), w)
			)
		end
	end
	scatter!(X[1, :], X[2, :], color="black")
	save("graph_rep.pdf", fig)
	fig
end

# ╔═╡ a064c35e-0e4f-418e-a329-e747daae264e
viz_graph(X, kernel)

# ╔═╡ 17a20c5b-7da4-4982-9b7a-21ba3c3b8060
md"## diff map"

# ╔═╡ 310ba8ae-7443-44a5-b77d-168336af39bd
function color_points(X̂::Vector{Float64}, name::String)
	cmap = ColorSchemes.terrain
	
	fig = Figure()
	ax = Axis(fig[1, 1], 
		xlabel="x₁", 
		ylabel="x₂", 
		title=name, 
		aspect=DataAspect()
	)
	sp = scatter!(x₁, x₂, color=X̂, colormap=cmap, strokewidth=1)
	Colorbar(fig[1, 2], sp, label="latent dim")
	save("$name.pdf", fig)
	fig
end

# ╔═╡ 7fbe0950-27de-4969-9962-16080e6908ff
X̂_cory = diffusion_map(X, kernel, 1)[:]

# ╔═╡ 4311f687-824c-42d4-b0a0-b08945460e76
color_points(X̂_cory, "diff map")

# ╔═╡ 1349c4d0-d6af-4154-b139-defdaec008a1
md"## compare to PCA"

# ╔═╡ ce982923-e6df-48c8-8011-4417b5e1a347
X̂_pca = PCA(n_components=1).fit_transform(X')[:]

# ╔═╡ 2aaea099-6f33-4260-bf40-f82644e24eae
color_points(X̂_pca, "PCA")

# ╔═╡ Cell order:
# ╠═cedad242-4983-11eb-2f32-3d405f151b77
# ╠═1996b541-e783-4b3a-9f7a-aca380d1047f
# ╠═1923b9da-6f21-4675-a7bf-6703f3165118
# ╠═8049574b-7fa8-49ec-a022-7b879e9ea38b
# ╠═f306c9dc-cfd5-4fe0-b4cb-9a9d2da7281c
# ╠═bad179ee-cef7-444b-b7cb-15053ac62959
# ╟─8e01ddd4-27c9-4f5c-8d7f-abd4f2be4f82
# ╠═c793e6b0-4983-11eb-29ae-3366c7d31e84
# ╠═c64b57a4-ac19-461b-ba0f-4a3bfe583db2
# ╟─ecd00826-6e79-47b1-ab60-46ac3e1bfef9
# ╠═9b487b40-63c1-4a9b-b1f1-ef7703dd4dcc
# ╠═3c5dd85c-d21c-43f9-a0f6-fe3873a5268e
# ╠═48aa38c3-9876-4f7c-8e4f-e22dec5104fb
# ╠═659956e9-aaa6-437f-85c3-c25e01457a28
# ╠═a064c35e-0e4f-418e-a329-e747daae264e
# ╟─17a20c5b-7da4-4982-9b7a-21ba3c3b8060
# ╠═310ba8ae-7443-44a5-b77d-168336af39bd
# ╠═7fbe0950-27de-4969-9962-16080e6908ff
# ╠═4311f687-824c-42d4-b0a0-b08945460e76
# ╟─1349c4d0-d6af-4154-b139-defdaec008a1
# ╠═ce982923-e6df-48c8-8011-4417b5e1a347
# ╠═2aaea099-6f33-4260-bf40-f82644e24eae
