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
import numpy as np

#Read in data, create a unified kappa column but label computational samples
eps = 10**-6
data = pandas.read_csv("data.csv")
data["computational_label"] = [int(i) for i in data["kappa_exp"].isnull()]
data["kappa"] = data["kappa_exp"].fillna(data["kappa_latt_comp"])
data["kappa"] = np.log(data["kappa"] + eps)
data = data.loc[data["kappa"]<50,:]
data = data.drop(columns=["kappa_latt_comp","kappa_exp"])
data = data.dropna()
data.reset_index(drop=True,inplace=True)

data_automatminer = data.drop(columns=["ICSD"])

Featurizer = automatminer_featurizer_sklearn_pipeline("express")
print(data_automatminer)
data_train,data_labels = Featurizer.fit_transform(data_automatminer)

model = RandomForestRegressor(max_features="sqrt",min_samples_leaf=3,min_samples_split=6,n_estimators=329)
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

