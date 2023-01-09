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
import numpy as np
import matplotlib
import sklearn.metrics as metrics

def plot(true,pred,log = False):
    matplotlib.use('Agg')
    
    ### Set up data
    
    ### Set up viewer
    
    # Initialize figurel
    fig = plt.figure(figsize = [12.8,12.8])
    ax = fig.add_subplot(1, 1, 1, aspect='auto')
    
    ### Set up layers
    
    ## Layer 1: cluster-cross-validation-cut
    
    layer_data = data
    
    # Get main data values
    x = true
    y = pred
    
    ratio = np.sort(y/x)
    ratio = ratio[ratio > 0]
    
    #print(ratio)
    if log:
        r2 = metrics.r2_score(np.log(x+eps),np.log(y+eps))
        E2 = (np.log(x+eps)-np.log(y+eps))**2
    else:
        r2 = metrics.r2_score(x,y)
        E2 = (x-y)**2
    
    MSE = E2.mean()
    
    
    matplotlib.rcParams['axes.linewidth'] = 2 #set the value globally
    ax.plot(x, y, 'o', color='#595959', markersize=5, alpha=0.4, zorder=10, mec='none')
    ax.plot([0,100000],[0,100000],linewidth = 4,color='#8888ffee')
    ax.plot([0,100000],[0,100000*np.percentile(ratio,5)],linewidth = 6,color='#8888ee77')
    ax.plot([0,100000],[0,100000*np.percentile(ratio,1)],linewidth = 8,color='#8888bb44')
    ### Finalize viewer
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    if log:
        plt.figtext(.15,.825,"${MSE_{Log}}$ = %.1f" % MSE ,fontsize = 28)
    else:
        plt.figtext(.15,.825,"MSE = %.1f" % MSE ,fontsize = 28)
    
    if log:
        plt.figtext(.130,.775,"  ${R_{Log}^2}$  = %.2f" % r2 , fontsize= 28)
    else:
        plt.figtext(.146,.775,"  ${R^2}$  = %.2f" % r2 , fontsize= 28)
    
    #plt.figtext(.55,.82,"Prediction = True",fontsize = 28)
    plt.figtext(.75,.12+np.percentile(ratio,5)*0.7,"<95%", fontsize= 28)
    plt.figtext(.75,.12+np.percentile(ratio,1)*0.7,"<99%" ,fontsize = 28)
    # Set limits
    plt.xticks(ticks=[0,10,20,30,40])
    plt.yticks(ticks=[10,20,30,40])
    ax.set_xlim(0.0, 40.0)
    ax.set_ylim(0.0, 40.0)
    
    # Set scale (log or linear)
    
    # Set axis label properties
    ax.set_xlabel('true $\\kappa$ (W m$^{-1}$ K$^{-1}$)', weight='normal', size=32)
    ax.set_ylabel('predicted $\\kappa$ (W m$^{-1}$ K$^{-1}$)', weight='normal', size=32)
    
    # Set tick label properties
    plt.tick_params(length=20,width=2,direction = 'in')
    ax.tick_params('x', labelsize=32)
    ax.tick_params('y', labelsize=32)

    if log:
        prefix = "log_"
    else:
        prefix = ""
    
    # Save figure
    fig.savefig(prefix + 'cross_val_40.tif',format='tif')
    fig.savefig(prefix +'cross_val_40.png',format='png')
    
    # Set limits
    plt.xticks(ticks=[0,5,10,15])
    plt.yticks(ticks=[5,10,15])
    ax.set_xlim(0.0, 15.0)
    ax.set_ylim(0.0, 15.0)

    
    fig.savefig(prefix +'cross_val_15.tif',format='tif')
    fig.savefig(prefix +'cross_val_15.png',format='png')
    plt.close(fig)

    # Set limits
    ax.plot([0,0],[0,100000],linewidth = 4,color='#8888ffee')
    plt.xticks(ticks=[0,5,10,15])
    plt.yticks(ticks=[5,10,15])
    ax.set_xlim(-1.0, 7.0)
    ax.set_ylim(-1.0, 7.0)

    fig.savefig(prefix +'cross_val_7.tif',format='tif')
    fig.savefig(prefix +'cross_val_7.png',format='png')
    plt.close(fig)


#Load in the data cache dictionary
data_cache = pk.load(open("Cache/Data_Cache.pk","rb"))

#LOGARITHMIC MODEL  Parametric
eps = 10**-6
data = pandas.read_csv("data.csv")
data = data.sample(frac=1).reset_index(drop=True)
data["computational_label"] = [int(i) for i in data["kappa_exp"].isnull()]
#data["computational_label"] = 0
data["kappa"] = data["kappa_exp"].fillna(data["kappa_latt_comp"])
#Predicting the log is more robust at small values
data["kappa"] = data["kappa_exp"].fillna(data["kappa_latt_comp"])
data["kappa"] = np.log(data["kappa"] + eps)
data = data.loc[data["kappa"]<50,:]
data = data.drop(columns=["kappa_latt_comp","kappa_exp"])
data = data.dropna()
data.reset_index(drop=True,inplace=True)
data_automatminer = data.drop(columns=["ICSD"])

indicies = sklearn.model_selection.KFold(n_splits=5)
test_frames = []
for train_i,test_i in indicies.split(data_automatminer):
    Featurizer = automatminer_featurizer_sklearn_pipeline("heavy")
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
true = np.exp(CV_results["true"])-eps
pred = np.exp(CV_results["pred"])-eps
plot(true,pred,log=True)
plt.savefig("CV_results.png")
print(sklearn.metrics.r2_score(CV_results["true"],CV_results["pred"]))


#Linear model Parametric
eps = 10**-6
data = pandas.read_csv("data.csv")
data = data.sample(frac=1).reset_index(drop=True)
data["computational_label"] = [int(i) for i in data["kappa_exp"].isnull()]
#data["computational_label"] = 0
data["kappa"] = data["kappa_exp"].fillna(data["kappa_latt_comp"])
#Predicting the log is more robust at small values
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
    
    model = RandomForestRegressor(max_features="sqrt",min_samples_leaf=3,min_samples_split=6,n_estimators=329)
    model.fit(data_train,train_labels)
    
    test_sample = pandas.DataFrame(data_automatminer.iloc[test_i,:])
    data_test,_ = Featurizer.transform(test_sample)
    kappa_pred = model.predict(data_test)
    CV_frame = pandas.DataFrame(zip(kappa_pred,_),columns=["pred","true"])
    test_frames.append(CV_frame)

CV_results = pandas.concat(test_frames,axis=0)
CV_results.to_csv("cross_val.csv")



plt.figure()
plot(CV_results["true"],CV_results["pred"],log=False)
plt.savefig("CV_results.png")
print(sklearn.metrics.r2_score(CV_results["true"],CV_results["pred"]))

