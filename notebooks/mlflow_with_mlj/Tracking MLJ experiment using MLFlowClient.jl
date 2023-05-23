### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ c564ca48-f6b4-11ed-3b83-23f4fd2e4e3a
begin
	using Pkg
	Pkg.activate("./")

	using Random
	Random.seed!(9)
end

# ╔═╡ d08b6406-9c44-4f79-aa2d-1c5b92acfe1d
using MLJ, DataFrames

# ╔═╡ dde085f0-ed2b-4ab3-8af4-459b28e98dd5
using MLFlowClient

# ╔═╡ d6cdaf4a-c3ea-49d5-9402-00c52e33cb15
md"""
# Tracking MLJ experiment using MLFlowClient.jl
In this case, the `max_depth` hyperparameter is being tuned. MLFlow will track the entire parameter as an array, and receive the accuracy for each model.
"""

# ╔═╡ 5f9ce9bd-b7f9-4d31-a0ac-25161673e288
md"### Data ingestion"

# ╔═╡ 3c6ad955-a354-4598-a4c1-9441e1eba5a6
iris = load_iris() |> DataFrames.DataFrame

# ╔═╡ 6151a07d-3b11-4306-a61a-649d1f53b6df
schema(iris)

# ╔═╡ d2eee43a-d51e-46c2-aadb-9aaf6c500057
train, test = partition(iris, 0.8, shuffle=true)

# ╔═╡ c2079d3c-485d-4075-a73d-45350f96860c
train_y, train_X = unpack(train, ==(:target))

# ╔═╡ a6a0b427-7c52-4229-92d4-2a1f6ca19f05
test_y, test_X = unpack(test, ==(:target))

# ╔═╡ b8f54e7e-4e18-4233-9020-a48a7920c267
md"""
### MLFlowClient setup
"""

# ╔═╡ 0ae60171-4e67-4afc-bbb3-84900a25c857
mlf = MLFlow("http://localhost:5000")

# ╔═╡ 27cf83c4-d9c3-42db-8466-c36e6bddc9e1
if ismissing(getexperiment(mlf, "iris_classification"))
	experiment_id = createexperiment(mlf; name="iris_classification", artifact_location="./iris-artifacts")
else
	experiment_id = getexperiment(mlf, "iris_classification")
end

# ╔═╡ 8f77e826-1901-4f8a-94db-250fbbed7614
md"""
### Modeling
"""

# ╔═╡ e8d2bfd5-39e5-4fe6-b734-aa4064af326f
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree

# ╔═╡ fc5101e4-0305-417a-b426-8cbf37fa9918
dtc = DecisionTreeClassifier()

# ╔═╡ 96b53a9f-5947-4d47-8738-590c5dc8b732
max_depth_range = range(dtc, :max_depth, lower=2, upper=10, scale=:linear);

# ╔═╡ 26e3db6a-69d4-445e-b4d4-fdb0f250db12
model = TunedModel(
	model=dtc,
	resampling=CV(),
	tuning=Grid(),
	range=max_depth_range,
	measure=[accuracy, log_loss, misclassification_rate, brier_score]
)

# ╔═╡ 50e0a6b3-f66f-4d6e-aa80-c0555ec67505
mach = machine(model, train_X, train_y)

# ╔═╡ 82e30012-c755-4011-8466-a6f91e794759
fit!(mach)

# ╔═╡ e065f4d2-0e59-48c3-9eb2-49c970d23f5b
md"### Evaluating"

# ╔═╡ f81b703a-6db2-4164-aeda-18fa2ddfc901
model_values = report(mach).history .|> (x -> (x.measure, x.measurement, x.model.max_depth))

# ╔═╡ e9cc5704-eb50-4f10-9860-b786a47b6e05
for (measure, measurements, max_depth) in model_values
	exprun = createrun(mlf, experiment_id)
	logparam(mlf, exprun, "max_depth", max_depth)
	
	measures_names = [x.name for x in measure .|> info]
	for (name, val) in zip(measures_names, measurements)
		logmetric(mlf, exprun, "$(name)", val)
	end
end

# ╔═╡ Cell order:
# ╟─d6cdaf4a-c3ea-49d5-9402-00c52e33cb15
# ╠═c564ca48-f6b4-11ed-3b83-23f4fd2e4e3a
# ╠═d08b6406-9c44-4f79-aa2d-1c5b92acfe1d
# ╟─5f9ce9bd-b7f9-4d31-a0ac-25161673e288
# ╠═3c6ad955-a354-4598-a4c1-9441e1eba5a6
# ╠═6151a07d-3b11-4306-a61a-649d1f53b6df
# ╠═d2eee43a-d51e-46c2-aadb-9aaf6c500057
# ╠═c2079d3c-485d-4075-a73d-45350f96860c
# ╠═a6a0b427-7c52-4229-92d4-2a1f6ca19f05
# ╟─b8f54e7e-4e18-4233-9020-a48a7920c267
# ╠═dde085f0-ed2b-4ab3-8af4-459b28e98dd5
# ╠═0ae60171-4e67-4afc-bbb3-84900a25c857
# ╠═27cf83c4-d9c3-42db-8466-c36e6bddc9e1
# ╟─8f77e826-1901-4f8a-94db-250fbbed7614
# ╠═e8d2bfd5-39e5-4fe6-b734-aa4064af326f
# ╠═fc5101e4-0305-417a-b426-8cbf37fa9918
# ╠═96b53a9f-5947-4d47-8738-590c5dc8b732
# ╠═26e3db6a-69d4-445e-b4d4-fdb0f250db12
# ╠═50e0a6b3-f66f-4d6e-aa80-c0555ec67505
# ╠═82e30012-c755-4011-8466-a6f91e794759
# ╟─e065f4d2-0e59-48c3-9eb2-49c970d23f5b
# ╠═f81b703a-6db2-4164-aeda-18fa2ddfc901
# ╠═e9cc5704-eb50-4f10-9860-b786a47b6e05
