import automatminer.featurization as auto_feat
from automatminer.preprocessing import *
from sklearn.preprocessing import *
from matminer.featurizers.composition import *
import pandas

class automatminer_featurizer_sklearn_pipeline():
    def __init__(self,preset="heavy"):
        self.pipe = auto_feat.AutoFeaturizer(preset=preset)
        self.DataCleaner = DataCleaner(max_na_frac=0)
        self.RobustScalar = RobustScaler()
    def fit_transform(self,composition_frame,kappa_series,**kargs):
        copy = composition_frame.copy()

        #Kappa is required for auto featurizer
        copy["kappa"] = kappa_series
        copy = self.pipe.fit_transform(copy,"kappa")
        print(copy)
        copy = self.DataCleaner.fit_transform(copy,"kappa",)
        labels = copy["kappa"]
        copy = copy.drop("kappa",axis=1)

        copy_columns = copy.columns
        copy = self.RobustScalar.fit_transform(copy)
        copy = pandas.DataFrame(copy,columns=copy_columns)
        return copy,labels
    def transform(self,composition_dataframe,y="kappa",**kargs):
        labels = composition_dataframe["kappa"]
        result = self.pipe.transform(composition_dataframe,y)
        print(result)
        result = self.DataCleaner.transform(result,y)
        result = result.drop("kappa",axis=1)
        result = self.RobustScalar.transform(result)
        return result,labels