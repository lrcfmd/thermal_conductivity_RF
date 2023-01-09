from locallib import automatminer_featurizer_sklearn_pipeline
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle as pk
import pandas
import numpy as np


#Input is a dataframe that has compositions as strings in a column called "composition"
def thermal_conductivitiy_featurize_and_predict(data):
    eps = 10**-6
    preprocessor = pk.load(open("pickle_jar/featurizer.pk","rb"))
    model = pk.load(open("pickle_jar/model.pk","rb"))
    composition = pandas.DataFrame(data["composition"],columns=["composition"])
    composition["computational_label"] = 0
    composition["kappa"] = 0
    print(composition)
    data_input,_ = preprocessor.transform(composition)
    #The model actually predicts log values
    data["kappa_pred"] = model.predict(data_input)
    data["kappa_pred"] = np.exp(data["kappa_pred"])-eps
    return data


if __name__ == "__main__":
    data = pandas.read_csv("data.csv")
    data = thermal_conductivitiy_featurize_and_predict(data)
    plt.hist(data["kappa_pred"],bins="fd")
    plt.savefig("predictions_hist")
