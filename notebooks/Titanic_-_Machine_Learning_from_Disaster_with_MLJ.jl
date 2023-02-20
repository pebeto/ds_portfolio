### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 2cf49011-bf27-452c-a982-29116fff2022
begin
	using Pkg

	Pkg.activate("..")
end

# ╔═╡ c4b15a5c-afb3-11ed-20d4-25f28fdd1c4e
begin
	using MLJ
	using Plots
	using Random
	using DataFrames
	using MLDatasets
	using Statistics
	using StatsPlots
end

# ╔═╡ e9c2a3b8-0488-4cc3-b7dc-cac42c7c88f1
md"""
# [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
###### (Not in Kaggle because of lack of official support 😞)

### Premise
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc). 
"""

# ╔═╡ f664d66a-78e7-4bdc-8bde-f1018697765a
Random.seed!(9)

# ╔═╡ 0d549d56-dfa0-4872-8917-bdc8b0169513
md"### Loading Titanic Dataset"

# ╔═╡ 5b9ef285-6922-4a6d-8ba9-c5e8321fbb95
begin
	df = MLDatasets.Titanic().dataframe
	MLJ.schema(df)
end

# ╔═╡ 750193f2-4207-43dd-b26a-7f81e006170a
md"""
### Data pre-processing
#### Checking features unique values
"""

# ╔═╡ 6242212e-80bb-4b80-a0ca-214e99554842
for f in names(df)
	println(f)
	println(df[!, f])
end

# ╔═╡ b6e5758e-983e-4eab-ab8b-ed48ef481c2b
md"#### Checking for missing values"

# ╔═╡ c16bcb1d-6f55-4171-87cc-e86dfe95719c
describe(df)[!, [:variable, :nmissing]]

# ╔═╡ 0ecd855d-154f-4d8c-8514-9814bc91f2f8
md"##### Dropping the Embarked observations"

# ╔═╡ 0a26b983-32e1-442f-ba9c-92a93f46f446
begin
	dropmissing!(df, :Embarked)
	describe(df)[!, [:variable, :nmissing]]
end

# ╔═╡ ec99a01c-6099-489b-a4a2-91bd171b8a0d
md"#### Adding SibSp and Parch as FamilySize"

# ╔═╡ 73933819-f5ef-444e-869d-3e4da6829eb6
DataFrames.transform!(df, [:SibSp, :Parch] => (+) => :FamilySize)

# ╔═╡ e2bcf0af-6174-427a-ad35-7e22b1785654
md"#### Selecting features"

# ╔═╡ 481535e6-74c0-494a-977e-77a61a495511
selected_features = [:Pclass, :Sex, :Age, :FamilySize, :Fare, :Embarked, :Survived]

# ╔═╡ 88bf1ca6-2338-4269-bff3-60a51884ba8c
sf_df = copy(df[!, selected_features])

# ╔═╡ 082369f1-3043-4c72-8054-5e2e9990e73f
schema(sf_df)

# ╔═╡ ffb10a6c-004c-41f8-ae64-5b13cb53678a
coerce!(sf_df, :Sex => Multiclass, :Embarked => Multiclass, :Pclass => Multiclass,  :Survived => OrderedFactor)

# ╔═╡ be534aed-4288-4c0a-aac5-dc0554965ba1
schema(sf_df)

# ╔═╡ ae74d5d7-7c26-4c95-bf2e-62b85e330064
md"#### Impute on features containing missing"

# ╔═╡ b3b27023-df7f-44bb-9ecc-f23dd91b1d5b
FillImputer = @load FillImputer pkg=MLJModels verbosity=0

# ╔═╡ 7c20ee66-913c-41a9-8cf7-a08878eedff0
imputer_machine = MLJ.machine(FillImputer(), sf_df) |> fit!

# ╔═╡ b68abb0b-3102-4078-a144-e81d05f34e1a
imp_df = MLJ.transform(imputer_machine, sf_df)

# ╔═╡ 16e017de-79ec-480d-a5ec-59dba4cd05ee
describe(imp_df)

# ╔═╡ 0505706d-11ed-4471-8e5f-59cc482a3612
md"#### Feature one-hot encoder"

# ╔═╡ 016a9777-d71e-48f6-9366-91c0f9e56970
OneHotEncoder = @load OneHotEncoder pkg=MLJModels verbosity=0

# ╔═╡ f4397fca-aa46-4649-984d-e4853f46c68b
ohe_machine = MLJ.machine(OneHotEncoder(features=[:Survived], ignore=true), imp_df) |> fit!

# ╔═╡ 49c82fb7-7071-40fb-8ac8-c57f45379243
ohe_df = MLJ.transform(ohe_machine, imp_df)

# ╔═╡ fb894bce-6fc4-41aa-890a-cdfa006a4aca
describe(ohe_df)

# ╔═╡ 69b4ef3f-5763-415f-9475-eb435529f3ee
md"#### Feature standardization"

# ╔═╡ 417f74a5-d9af-47a1-9e74-4e959131a9ee
Standardizer = @load Standardizer pkg=MLJModels verbosity=0

# ╔═╡ 2bb306b5-93c2-4def-bc1e-14ddcf35c659
std_machine = MLJ.machine(Standardizer(), ohe_df) |> fit!

# ╔═╡ 02a938e0-3b45-4324-9e3f-b9aabdcc79fe
std_df = MLJ.transform(std_machine, ohe_df)

# ╔═╡ 0654f6e5-632a-4088-b6c1-9df5a60cd880
describe(std_df)

# ╔═╡ 8874736b-4ec3-40b6-8968-182620b87082
df_train, df_test = partition(std_df, 0.8, rng=9)

# ╔═╡ 52f7ada2-6428-4be3-8edb-58e641d6fddf
y_train, X_train = unpack(df_train, ==(:Survived))

# ╔═╡ 05a41bcb-e708-45ab-89a9-25f7bda8d772
y_test, X_test = unpack(df_test, ==(:Survived))

# ╔═╡ bebb64c1-96d6-4338-8b8e-e28900716dd7
md"""
### Data analysis
#### Age histogram
It is possible to see that $20<Age<40$ is a great part of the distribution.
"""

# ╔═╡ bb037397-558f-4654-b6b6-abd1256b6574
histogram(imp_df[!, :Age], legend=false)

# ╔═╡ 2eeb077d-f5e3-4e43-ae6b-c7e984d97568
md"""
#### Ages based on sex histogram
Obviously, as history said, most of the Titanic passengers were man. There's something curious about $Age<20$ showing more girls in that range instead of boys. 

The two distributions shows a peak on $20<Age<40$, as saw in the general age plot.
"""

# ╔═╡ 9653e83d-bfec-42bb-bb1a-13c8219f201e
begin
	ages_female_group, ages_male_group = groupby(imp_df[:, [:Age, :Sex]], :Sex)
	ages_female_hist = histogram(ages_female_group[:, :Age], label="Female", color=:pink)
	ages_male_hist = histogram(ages_male_group[:, :Age], label="Male", color=:dodgerblue)
	plot(ages_female_hist, ages_male_hist)
end

# ╔═╡ 952f1fb2-c8bd-481e-9a70-70363036537d
md"""
#### Survived people based on Sex
- *Male* frequency is twice as *Female*
- From the total, there's a notorious difference between *Male* and *Female* in terms of survivability. In this graph is possible to confirm what history says about woman survival rate at disaster
"""

# ╔═╡ 1588ded8-d2f7-43c0-a0e8-e7cdacf51fd7
begin
	sex_survived_groups = groupby(imp_df[:, [:Survived, :Sex]], :Sex) |> collect
	sex_survived_totals = sex_survived_groups .|> size .|> first
	freq_sex_survived = map(g -> int(g[:, :Survived], type=Int) .- 1 |> sum, sex_survived_groups)
	freq_sex_not_survived = sex_survived_totals - freq_sex_survived
	groupedbar(["Female", "Male"], [sex_survived_totals freq_sex_not_survived freq_sex_survived], labels=["Total" "Not survived" "Survived"], c=[:dodgerblue :darkred :lightgreen])
end

# ╔═╡ c23d4cf9-2830-450e-8c60-4d9b93c68a20
md"""
#### Survived people based on Age
- People $Age < 15$ had a high survival rate
- People $15 < Age < 40$ comprehends a large group of not survivors
- People $Age > 40$ show a decrease on not-survivability
"""

# ╔═╡ f96583ec-b436-4704-81db-5ac11cc8297c
begin
	age_not_survived_group, age_survived_group = groupby(imp_df[:, [:Age, :Survived]], :Survived)
	not_survived_age_hist = histogram(age_not_survived_group[:, :Age], label="Not survived", color=:darkred)
	survived_age_hist = histogram(age_survived_group[:, :Age], label="Survived", color=:lightgreen)
	plot(not_survived_age_hist, survived_age_hist)
end

# ╔═╡ 9b07057f-bcd6-4791-8b1a-ee3962ee3c70
md"""
#### Survived people based on ticket class (Pclass)
Ticket class (Pclass) is a proxy for socio-economic status (SES)
- 1st = Upper
- 2nd = Middle
- 3rd = Lower

---

- Lower socio-economic class had most of passengers, but most didn't survive.
- High socio-economic class had the largest survival rate.
"""

# ╔═╡ c63e5921-dd99-4210-a684-7f834cfcc5e4
begin
	p_groups = groupby(imp_df[:, [:Pclass, :Age, :Survived]], :Pclass) |> collect
	p_totals = p_groups .|> size .|> first
	p_survived = map(p -> int(p[:, :Survived], type=Int) .- 1 |> sum, p_groups)
	p_not_survived = p_totals - p_survived
	groupedbar(["Upper", "Middle", "Lower"], [p_totals p_not_survived p_survived], labels=["Total" "Not survived" "Survived"], c=[:dodgerblue :darkred :lightgreen])
end

# ╔═╡ 812aaeb6-417d-499e-bd74-8c7042dab389
md"""
#### Ages histogram from survived people based on ticket class
"""

# ╔═╡ f906d508-a0c8-403b-b17e-25fde6192775
function generatephists(data::Vector, n::Integer)
	not_survived_hist = histogram(data[n][1], label="Not survived", color=:darkred)
	survived_hist = histogram(data[n][2], label="Survived", color=:lightgreen)
	return plot(not_survived_hist, survived_hist, layout=(1, 2), plot_title="Pclass $n")
end

# ╔═╡ 8df9d01f-39df-4faa-8deb-47d4379c9a5e
p_age_survived_ages = p_groups .|> (p -> groupby(p[:, [:Age, :Survived]], :Survived)) .|> collect .|> (p -> p .|> (q -> q[:, :Age]))

# ╔═╡ f41cb2ba-ca8f-47c8-8fdb-0e12cc33b796
p1_age_survived_hists = generatephists(p_age_survived_ages, 1)

# ╔═╡ 934bf39f-3aa4-4a90-974e-0d1f039ad241
p2_age_survived_hists = generatephists(p_age_survived_ages, 2)

# ╔═╡ c41ea385-5e8d-4d60-b0c3-bdca1bf78b4a
p3_age_survived_hists = generatephists(p_age_survived_ages, 3)

# ╔═╡ 95bf8ab4-9584-40df-b7f9-a04d6505d710
md"""
#### Correlation matrix
- *Sex_female* feature is high correlated to our target, meaning a high survival rate. Otherwise, *Male_sex* feature have an inverse behaviour with our target, meaning a low survival rate.
- People embarked on Southampton are low correlated to our target, meaning a low survival rate. Combined with the low correlation from *Fare*, and the high correlation with our target, we conclude that people from Southampton comprehend the majority from non-survived people.
"""

# ╔═╡ fd0a64f4-68eb-4c3e-95df-5f5572ee6a5f
heatmap(names(ohe_df), names(ohe_df), cor(Matrix(ohe_df)), xrotation=25, color=:acton)

# ╔═╡ f91d5c26-3670-418c-9e2e-840ac75a46e1
md"""
### Modeling phase
#### MLJ model suggestions
"""

# ╔═╡ 3dd1b6e2-ee4c-49f2-a416-4763c989ca23
models(matching(X_train, y_train))

# ╔═╡ 3af35df4-20aa-4d0a-a568-b88c0395dfa5
md"#### Define utility variables"

# ╔═╡ 5d3e58b2-474a-4f5c-b15c-4a9e3e840eb0
md"#### Decision tree classifier"

# ╔═╡ 3c323a0c-d2ed-4ae2-8dcb-b48d9f4e0e96
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=BetaML verbosity=0

# ╔═╡ 673b9518-4c2e-430f-8b35-f59bd63c5420
dtc_mach = machine(DecisionTreeClassifier(), X_train, y_train)

# ╔═╡ 17e2d70b-ce0d-4e3f-b442-a3d3382c42a5
evaluate!(dtc_mach, resampling=CV(), measure=[LogLoss(), Accuracy(), Precision(), Recall(), FScore()])

# ╔═╡ 07baf7a4-ff67-4cdc-a6dc-3be593bfa8e3
md"#### Linear perceptron"

# ╔═╡ 19548d15-6ed8-49cd-b4c0-7cbd7cb21826
LinearPerceptron = @load LinearPerceptron pkg=BetaML verbosity=0

# ╔═╡ 254ccda3-49f7-4cce-8bb6-c5e96f8f51c2
lp_mach = machine(LinearPerceptron(), X_train, y_train)

# ╔═╡ 3196cd78-1895-4635-b6ed-eff468627c3b
evaluate!(lp_mach, resampling=CV(), measure=[LogLoss(), Accuracy(), Precision(), Recall(), FScore()])

# ╔═╡ 73ed0867-94d7-46a5-a71f-2de3542b3e4d
md"#### Random forest classifier"

# ╔═╡ 68873c9b-5910-4ff4-b342-74213e4e4597
RandomForestClassifier = @load RandomForestClassifier pkg=BetaML verbosity=0

# ╔═╡ bffbb43a-dc75-4496-a012-4ab7f6c8fe6c
rfc_mach = machine(RandomForestClassifier(n_trees=100), X_train, y_train)

# ╔═╡ 3c024676-7b16-4e90-8d93-993e2abf5be8
evaluate!(rfc_mach, resampling=CV(), measure=[LogLoss(), Accuracy(), Precision(), Recall(), FScore()])

# ╔═╡ 79c867d5-7aba-4903-b3d6-81c1efd6ee34
md"#### Pegasos (SVM)"

# ╔═╡ cd153f6f-bd4f-40e9-9f86-d0a197f6e47e
Pegasos = @load Pegasos pkg=BetaML verbosity=0

# ╔═╡ 03dd2635-df86-43cb-b5bb-ecc32d66a474
pegasos_mach = machine(Pegasos(), X_train, y_train)

# ╔═╡ 319b7739-349f-4fb6-8a33-f5ae9b7588a9
evaluate!(pegasos_mach, resampling=CV(), measure=[LogLoss(), Accuracy(), Precision(), Recall(), FScore()])

# ╔═╡ ef390625-0a17-470c-bebe-18d05ff4f80d
md"#### Best model tuning"

# ╔═╡ 7b772bcf-9b59-4467-b3a8-62910de63300
rfc = RandomForestClassifier()

# ╔═╡ e418fabd-01cd-4d4c-a558-707933bd56ed
n_trees_range = range(rfc, :n_trees, lower=1, upper=20)

# ╔═╡ d4903a30-98d9-4177-b4b0-d4cfa1aefd6e
tuned_rfc = TunedModel(
	rfc,
	tuning=Grid(),
	range=[
		n_trees_range
	],
	measure=accuracy
)

# ╔═╡ 589809f9-912c-4fed-a367-abf6077dc2e5
tuned_rfc_mach = machine(tuned_rfc, X_train, y_train)

# ╔═╡ 98dc83c8-0d4a-4964-8852-aa351edd3d74
evaluate!(tuned_rfc_mach, resampling=CV(), measure=[LogLoss(), Accuracy(), Precision(), Recall(), FScore()])

# ╔═╡ 77a6902d-3026-461e-a402-bbd88cc760eb
fitted_params(tuned_rfc_mach).best_model

# ╔═╡ c1ec0f24-4184-44d5-b3e9-d158664e1ac6
md"### Observe results with test set"

# ╔═╡ 1ec1ffa7-3cf9-4873-aaee-fa2e77fd8215
yhat = predict(tuned_rfc_mach, X_test) .|> mode

# ╔═╡ 4485575c-5fde-4446-b9f1-9523864a9c85
rfc_accuracy = accuracy(yhat, y_test)

# ╔═╡ cadd5846-95cc-4e96-81ac-5e3bad500974
confusion_matrix(yhat, y_test)

# ╔═╡ b097191f-5aba-4657-a319-48cc86f992d8
rfc_precision = precision(yhat, y_test)

# ╔═╡ 6f74e94f-c1a3-40c2-aad3-572b67156980
rfc_recall = recall(yhat, y_test)

# ╔═╡ 98061d1c-8c73-4ddc-8dfb-33efbd71bbca
rfc_fscore = f1score(yhat, y_test)

# ╔═╡ Cell order:
# ╟─e9c2a3b8-0488-4cc3-b7dc-cac42c7c88f1
# ╟─2cf49011-bf27-452c-a982-29116fff2022
# ╠═c4b15a5c-afb3-11ed-20d4-25f28fdd1c4e
# ╠═f664d66a-78e7-4bdc-8bde-f1018697765a
# ╟─0d549d56-dfa0-4872-8917-bdc8b0169513
# ╠═5b9ef285-6922-4a6d-8ba9-c5e8321fbb95
# ╟─750193f2-4207-43dd-b26a-7f81e006170a
# ╠═6242212e-80bb-4b80-a0ca-214e99554842
# ╟─b6e5758e-983e-4eab-ab8b-ed48ef481c2b
# ╠═c16bcb1d-6f55-4171-87cc-e86dfe95719c
# ╟─0ecd855d-154f-4d8c-8514-9814bc91f2f8
# ╠═0a26b983-32e1-442f-ba9c-92a93f46f446
# ╟─ec99a01c-6099-489b-a4a2-91bd171b8a0d
# ╠═73933819-f5ef-444e-869d-3e4da6829eb6
# ╟─e2bcf0af-6174-427a-ad35-7e22b1785654
# ╠═481535e6-74c0-494a-977e-77a61a495511
# ╠═88bf1ca6-2338-4269-bff3-60a51884ba8c
# ╠═082369f1-3043-4c72-8054-5e2e9990e73f
# ╠═ffb10a6c-004c-41f8-ae64-5b13cb53678a
# ╠═be534aed-4288-4c0a-aac5-dc0554965ba1
# ╟─ae74d5d7-7c26-4c95-bf2e-62b85e330064
# ╠═b3b27023-df7f-44bb-9ecc-f23dd91b1d5b
# ╠═7c20ee66-913c-41a9-8cf7-a08878eedff0
# ╠═b68abb0b-3102-4078-a144-e81d05f34e1a
# ╠═16e017de-79ec-480d-a5ec-59dba4cd05ee
# ╟─0505706d-11ed-4471-8e5f-59cc482a3612
# ╠═016a9777-d71e-48f6-9366-91c0f9e56970
# ╠═f4397fca-aa46-4649-984d-e4853f46c68b
# ╠═49c82fb7-7071-40fb-8ac8-c57f45379243
# ╠═fb894bce-6fc4-41aa-890a-cdfa006a4aca
# ╟─69b4ef3f-5763-415f-9475-eb435529f3ee
# ╠═417f74a5-d9af-47a1-9e74-4e959131a9ee
# ╠═2bb306b5-93c2-4def-bc1e-14ddcf35c659
# ╠═02a938e0-3b45-4324-9e3f-b9aabdcc79fe
# ╠═0654f6e5-632a-4088-b6c1-9df5a60cd880
# ╠═8874736b-4ec3-40b6-8968-182620b87082
# ╠═52f7ada2-6428-4be3-8edb-58e641d6fddf
# ╠═05a41bcb-e708-45ab-89a9-25f7bda8d772
# ╟─bebb64c1-96d6-4338-8b8e-e28900716dd7
# ╟─bb037397-558f-4654-b6b6-abd1256b6574
# ╟─2eeb077d-f5e3-4e43-ae6b-c7e984d97568
# ╟─9653e83d-bfec-42bb-bb1a-13c8219f201e
# ╟─952f1fb2-c8bd-481e-9a70-70363036537d
# ╟─1588ded8-d2f7-43c0-a0e8-e7cdacf51fd7
# ╟─c23d4cf9-2830-450e-8c60-4d9b93c68a20
# ╟─f96583ec-b436-4704-81db-5ac11cc8297c
# ╟─9b07057f-bcd6-4791-8b1a-ee3962ee3c70
# ╟─c63e5921-dd99-4210-a684-7f834cfcc5e4
# ╟─812aaeb6-417d-499e-bd74-8c7042dab389
# ╟─f906d508-a0c8-403b-b17e-25fde6192775
# ╠═8df9d01f-39df-4faa-8deb-47d4379c9a5e
# ╟─f41cb2ba-ca8f-47c8-8fdb-0e12cc33b796
# ╟─934bf39f-3aa4-4a90-974e-0d1f039ad241
# ╟─c41ea385-5e8d-4d60-b0c3-bdca1bf78b4a
# ╟─95bf8ab4-9584-40df-b7f9-a04d6505d710
# ╠═fd0a64f4-68eb-4c3e-95df-5f5572ee6a5f
# ╟─f91d5c26-3670-418c-9e2e-840ac75a46e1
# ╠═3dd1b6e2-ee4c-49f2-a416-4763c989ca23
# ╟─3af35df4-20aa-4d0a-a568-b88c0395dfa5
# ╟─5d3e58b2-474a-4f5c-b15c-4a9e3e840eb0
# ╠═3c323a0c-d2ed-4ae2-8dcb-b48d9f4e0e96
# ╠═673b9518-4c2e-430f-8b35-f59bd63c5420
# ╠═17e2d70b-ce0d-4e3f-b442-a3d3382c42a5
# ╟─07baf7a4-ff67-4cdc-a6dc-3be593bfa8e3
# ╠═19548d15-6ed8-49cd-b4c0-7cbd7cb21826
# ╠═254ccda3-49f7-4cce-8bb6-c5e96f8f51c2
# ╠═3196cd78-1895-4635-b6ed-eff468627c3b
# ╟─73ed0867-94d7-46a5-a71f-2de3542b3e4d
# ╠═68873c9b-5910-4ff4-b342-74213e4e4597
# ╠═bffbb43a-dc75-4496-a012-4ab7f6c8fe6c
# ╠═3c024676-7b16-4e90-8d93-993e2abf5be8
# ╟─79c867d5-7aba-4903-b3d6-81c1efd6ee34
# ╠═cd153f6f-bd4f-40e9-9f86-d0a197f6e47e
# ╠═03dd2635-df86-43cb-b5bb-ecc32d66a474
# ╠═319b7739-349f-4fb6-8a33-f5ae9b7588a9
# ╟─ef390625-0a17-470c-bebe-18d05ff4f80d
# ╠═7b772bcf-9b59-4467-b3a8-62910de63300
# ╠═e418fabd-01cd-4d4c-a558-707933bd56ed
# ╠═d4903a30-98d9-4177-b4b0-d4cfa1aefd6e
# ╠═589809f9-912c-4fed-a367-abf6077dc2e5
# ╠═98dc83c8-0d4a-4964-8852-aa351edd3d74
# ╠═77a6902d-3026-461e-a402-bbd88cc760eb
# ╟─c1ec0f24-4184-44d5-b3e9-d158664e1ac6
# ╠═1ec1ffa7-3cf9-4873-aaee-fa2e77fd8215
# ╠═4485575c-5fde-4446-b9f1-9523864a9c85
# ╠═cadd5846-95cc-4e96-81ac-5e3bad500974
# ╠═b097191f-5aba-4657-a319-48cc86f992d8
# ╠═6f74e94f-c1a3-40c2-aad3-572b67156980
# ╠═98061d1c-8c73-4ddc-8dfb-33efbd71bbca
