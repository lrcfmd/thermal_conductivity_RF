from locallib import automatminer_featurizer_sklearn_pipeline
import pandas
from sklearn.preprocessing import *
from matminer.featurizers.composition import *
from automatminer.preprocessing import *
from sklearn.ensemble import RandomForestRegressor
import pickle as pk
import matplotlib.pyplot as plt
import automatminer
import os
#Read in data, create a unified kappa column but label computational samples

automatminer.utils.log.initalize_logger("none",open(os.devnull,"w"))
data = pandas.read_csv("data.csv")
data["computational_label"] = [int(i) for i in data["kappa_exp"].isnull()]
data["kappa"] = data["kappa_exp"].fillna(data["kappa_latt_comp"])
data = data.loc[data["kappa"]<50,:]
data = data.drop(columns=["kappa_latt_comp","kappa_exp"])
data = data.dropna()
data.reset_index(drop=True,inplace=True)

data_automatminer = data.drop(columns=["ICSD"])

Featurizer = automatminer_featurizer_sklearn_pipeline("express")
print(data_automatminer)
data_train,data_labels = Featurizer.fit_transform(data_automatminer,data_automatminer["kappa"])

model = RandomForestRegressor()
model.fit(data_train,data_labels)

data["pred_kappa"] = model.predict(data_train)
data.to_csv("post_train_sanity_check.csv")
print(data_train)
data_train.to_csv("Featurized_training_data.csv")

plt.figure()
plt.scatter(data["kappa"],data["pred_kappa"])
plt.savefig("parity_plot.png")

pk.dump(Featurizer,open("pickle_jar/featurizer.pk",mode="wb"))
pk.dump(model,open("pickle_jar/model.pk",mode="wb"))


