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
    def fit_transform(self,composition_frame,y="kappa",**kargs):
        composition_frame = pandas.DataFrame(composition_frame)
        composition_frame = self.pipe.fit_transform(composition_frame,y)
        composition_frame = self.DataCleaner.fit_transform(composition_frame,y,)
        labels = composition_frame[y]
        composition_frame = composition_frame.drop(y,axis=1)
        copy_columns = composition_frame.columns
        composition_frame = self.RobustScalar.fit_transform(composition_frame)
        composition_frame = pandas.DataFrame(composition_frame,columns=copy_columns)
        return composition_frame,labels
    def transform(self,composition_dataframe,y="kappa",**kargs):
        data = pandas.DataFrame(composition_dataframe)
        data = self.pipe.transform(data,y)
        data = self.DataCleaner.transform(data,y)
        labels = data[y]
        data = data.drop(y,axis=1)
        data_columns = data.columns
        data = self.RobustScalar.transform(data)
        data = pandas.DataFrame(data,columns=data_columns)
        return data,labels