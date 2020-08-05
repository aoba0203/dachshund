import pickle
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from preprocess import feature_add, feature_missing, feature_outlier, feature_scaler
import pickle
import os
from sklearn.linear_model import Lars
import numpy as np
import json
import joblib
from utils import definitions
import datetime

# outlier -> missing -> add -> scaler

class TrainModel:
  # def __init__(self, _project_name, _data_ratio, _df_train, _df_test, _target_column_name, _model):
  def __init__(self, _job):
    self.job = _job
    self.f_add = feature_add.FeatureAdd().getFeatureAddMethodList()
    self.f_missing = feature_missing.MissingData().getMissingDataMethodList()
    self.f_outlier = feature_outlier.FeatureOutlier().getRemovedOutlierMethodList()
    self.f_scaler = feature_scaler.FeatureScaler().getFeatureScalerMethodList()
    self.best_score = np.iinfo(np.int32).max
    return
  
  def __getHyperParamsSpace(self):
    return {
      'feature_add': hp.choice('feature_add', np.arange(len(self.f_add))),
      'feature_missing': hp.choice('feature_missing', np.arange(len(self.f_missing))),
      'feature_outlier': hp.choice('feature_outlier', np.arange(len(self.f_outlier))),
      'feature_scaler': hp.choice('feature_scaler', np.arange(len(self.f_scaler))),
      'model': self.job.model.getHyperParameterSpace(),
    }

  def __getPreprocessedDf(self, _df, _params):
    df = _df.copy()
    df = (list(self.f_missing.values())[_params['feature_missing']])(df)
    df = (list(self.f_outlier.values())[_params['feature_outlier']])(df)
    df = (list(self.f_add.values())[_params['feature_add']])(df)    
    x, y = self.__splitXy(df)
    x = (list(self.f_scaler.values())[_params['feature_scaler']])(x)
    return x, y

  def __splitXy(self, _df):
    y = _df[[self.job.target_column]]
    x = _df.drop(self.job.target_column, axis=1)
    return x, y
  
  def __minizeScore(self, _params):
    train_x, train_y = self.__getPreprocessedDf(self.job.df_train, _params)
    test_x, test_y = self.__getPreprocessedDf(self.job.df_test, _params)    
    score, trained_model = self.job.model.getScore(train_x, train_y, test_x, test_y, _params['model'])
    # if self.best_score > score:
    #   self.job.setTrainedModel(trained_model)
    return {'loss': score, 'status': STATUS_OK}
  
  def __writeBestParams(self, _best):
    params = self.job.model.params_list
    for key, value in zip(params.keys(), params.values()):
      v_idx = _best[key]
      _best[key]= params[key][v_idx]
    self.__saveBestParams(_best)
  
  # def __writeBestModel(self):
  #   filepath = definitions.getBestModelFilePath(self.job.project_name, self.job.model.model_name, self.job.data_ratio)
  #   joblib.dump(self.job.trained_model, filepath)  
  
  def optimizeModel(self):
    hyper_space = self.__getHyperParamsSpace()
    max_iter = self.job.model.getMaxIterCount()
    best = fmin(self.__minizeScore, hyper_space, algo=tpe.suggest, max_evals=max_iter)
    print(best)
    print('-optimize Model Success-')
    self.__writeBestParams(best)
  
  def getTrainedScore(self):
    params = self.getBestParams()
    print(params)
    train_x, train_y = self.__getPreprocessedDf(self.job.df_train, params)
    test_x, test_y = self.__getPreprocessedDf(self.job.df_test, params)
    score, model = self.job.model.getScore(train_x, train_y, test_x, test_y, params, _for_optimize=False)
    return score
  
  def __saveBestParams(self, _best):
    filepath = definitions.getBestModelParamsFilePath(self.job.project_name, self.job.model.model_name, self.job.data_ratio)
    with open(filepath, 'w') as result_file:
      json.dump(_best, result_file, default=self.myconverter)
  
  def getBestParams(self):
    filepath = definitions.getBestModelParamsFilePath(self.job.project_name, self.job.model.model_name, self.job.data_ratio)
    if os.path.exists(filepath):
      result_file = open(filepath, 'r')
      best = json.load(result_file)
      result_file.close()
      return best
    return {}

  def myconverter(self, obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()