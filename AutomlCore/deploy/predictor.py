from utils import definitions, utils
from preprocess import feature_add, feature_missing, feature_outlier, feature_scaler
import joblib
import json

class Predictor:
  def __init__(self):
    self.f_add = feature_add.FeatureAdd().getFeatureAddMethodList()
    self.f_missing = feature_missing.MissingData().getMissingDataMethodList()
    self.f_outlier = feature_outlier.FeatureOutlier().getRemovedOutlierMethodList()
    self.f_scaler = feature_scaler.FeatureScaler().getFeatureScalerMethodList()
  
  def getPredictResults(self, _x, _project_name, _model_name, _data_ratio):
    x = self.__initDataProcess(_x)
    path_model = definitions.getBestModelFilePath(_project_name, _model_name, _data_ratio)
    path_params = definitions.getBestModelParamsFilePath(_project_name, _model_name, _data_ratio)
    
    model = joblib.load(path_model)
    
    result_file = open(path_params, 'r')
    best = json.load(result_file)
    result_file.close()

    x = __getPreprocessedDf(x, best)
    return model.predict(x)

  def __initDataProcess(self):
    df = utils.splitDateColumns(self.df)
    self.df = utils.convertObjectType(df)   

def __getPreprocessedDf(self, _df, _params):
    df = _df.copy()
    df = (list(self.f_missing.values())[_params['feature_missing']])(df)
    df = (list(self.f_outlier.values())[_params['feature_outlier']])(df)
    df = (list(self.f_add.values())[_params['feature_add']])(df)    
    x = (list(self.f_scaler.values())[_params['feature_scaler']])(df)
    return x

