import pickle
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from preprocess import feature_add, feature_missing, feature_outlier, feature_scaler
import pickle
import os
from sklearn.linear_model import Lars
import numpy as np
import json
from utils import definitions

# outlier -> missing -> add -> scaler

class TrainModel:
  # def __init__(self, _project_name, _data_ratio, _df_train, _df_test, _target_column_name, _model):
  def __init__(self, _job):
    # self.project_name = _job.project_name
    # self.data_ratio = _job.data_ratio
    # self.df_train = _job.train_df
    # self.df_test = _job.test_df
    # self.target_column_name = _job.target_column
    # self.model = _job.model
    self.job = _job
    self.f_add = feature_add.FeatureAdd().getFeatureAddMethodList()
    self.f_missing = feature_missing.MissingData().getMissingDataMethodList()
    self.f_outlier = feature_outlier.FeatureOutlier().getRemovedOutlierMethodList()
    self.f_scaler = feature_scaler.FeatureScaler().getFeatureScalerMethodList()
    return
  
  def __getHyperParamsSpace(self):
    return {      
      # 'feature_add': hp.choice('feature_add', list(self.f_add.getFeatureAddMethodList().values())),
      # 'feature_missing': hp.choice('feature_missing', list(self.f_missing.getMissingDataMethodList().values())),
      # 'feature_outlier': hp.choice('feature_outlier', list(self.f_outlier.getRemovedOutlierMethodList().values())),
      # 'feature_scaler': hp.choice('feature_scaler', list(self.f_scaler.getFeatureScalerMethodList().values())),
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
  
  def __calcScore(self, _params):
    train_x, train_y = self.__getPreprocessedDf(self.job.df_train, _params)
    test_x, test_y = self.__getPreprocessedDf(self.job.df_test, _params)
    # train_x, train_y = self.__splitXy(train)
    # test_x, test_y = self.__splitXy(test)
        
    # train_x = (list(self.f_scaler.values())[_params['feature_scaler']])(train_x)
    # test_x = (list(self.f_scaler.values())[_params['feature_scaler']])(test_x)    
    score = self.job.model.getScore(train_x, train_y, test_x, test_y, _params['model'])
    return {'loss': score, 'status': STATUS_OK}
  
  def __writeBestParams(self, _best):
    params = self.job.model.params_list
    for key, value in zip(params.keys(), params.values()):
      v_idx = _best[key]
      _best[key]= params[key][v_idx]
    self.__saveBestParams(_best)
    
  def optimizeModel(self):
    print('optimizeModel: ' + str(self.job))
    hyper_space = self.__getHyperParamsSpace()
    max_iter = self.job.model.getMaxIterCount()
    best = fmin(self.__calcScore, hyper_space, algo=tpe.suggest, max_evals=max_iter)
    print('-optimize Model Success-')
    self.__writeBestParams(best)
  
  def getTrainedScore(self):
    params = self.getBestParams()
    train_x, train_y = self.__getPreprocessedDf(self.job.df_train, params)
    test_x, test_y = self.__getPreprocessedDf(self.job.df_test, params)
    return self.job.model.getScore(train_x, train_y, test_x, test_y, params['model'], _for_optimize=False)
  
  def __saveBestParams(self, _best):
    for key in _best.keys():
      print(key, type(key))
      if (type(_best[key]) == np.int32) & (type(_best[key]) == np.int64):
        _best[key] = float(_best[key])
    filepath = definitions.getBestModelParamsFilePath(self.job.project_name, self.job.model.model_name, self.job.project_name)
    with open(filepath, 'w') as result_file:
      json.dump(_best, result_file)
      # result_file.close()
  
  def getBestParams(self):
    filepath = definitions.getBestModelParamsFilePath(self.job.project_name, self.job.model.model_name, self.job.project_name)
    if os.path.exists(filepath):
      result_file = open(filepath, 'r')
      best = json.load(result_file)
      result_file.close()
      return best
    return {}
