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
import sklearn

data = pandas.read_csv("data.csv")
data = data.sample(frac=1).reset_index(drop=True)
data["computational_label"] = [int(i) for i in data["kappa_exp"].isnull()]
data["kappa"] = data["kappa_exp"].fillna(data["kappa_latt_comp"])
data = data.loc[data["kappa"]<50,:]
data = data.drop(columns=["kappa_latt_comp","kappa_exp"])
data = data.dropna()
data.reset_index(drop=True,inplace=True)
data_automatminer = data.drop(columns=["ICSD"])

indicies = sklearn.model_selection.KFold(n_splits=5)
test_frames = []
for train_i,test_i in indicies.split(data_automatminer):
    Featurizer = automatminer_featurizer_sklearn_pipeline("express")
    training_sample = pandas.DataFrame(data_automatminer.iloc[train_i,:])
    data_train,train_labels = Featurizer.fit_transform(training_sample)
    
    model = RandomForestRegressor()
    model.fit(data_train,train_labels)
    
    test_sample = pandas.DataFrame(data_automatminer.iloc[test_i,:])
    data_test,_ = Featurizer.transform(test_sample)
    kappa_pred = model.predict(data_test)
    CV_frame = pandas.DataFrame(zip(kappa_pred,_),columns=["pred","true"])
    test_frames.append(CV_frame)

CV_results = pandas.concat(test_frames,axis=0)
CV_results.to_csv("cross_val.csv")

plt.figure()
plt.scatter(CV_results["true"],CV_results["pred"])
plt.savefig("CV_results.png")
print(sklearn.metrics.r2_score(CV_results["true"],CV_results["pred"]))

