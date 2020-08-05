import numpy as np
from lightgbm import LGBMRegressor
from hyperopt import hp
from utils import definitions
from .. import model, model_regression

class LightGbmRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'LightGbmRegressor'
    self.params_list = {
      'boosting_type': ['gbdt', 'rf', 'dart'],
      'bagging_freq': np.arange(1, 12),
    }

  def getHyperParameterSpace(self):
    return {
      'max_depth': hp.quniform('max_depth', -1, 24, 2),
      'boosting_type': hp.choice('boosting_type', self.params_list['boosting_type']),
      'learning_rate': hp.uniform('learning_rate', 0, 0.1),
      'n_estimators': hp.quniform('n_estimators', 80, 200, 2),
      'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
      'bagging_freq': hp.choice('bagging_freq', self.params_list['bagging_freq']),
      'feature_fraction': hp.uniform('feature_fraction', 0, 1),
      'max_bin': hp.quniform('max_bin', 128, 512, 8),
      'min_data_in_leaf': hp.quniform('min_data_in_leaf', 12, 32, 2)
    }

  def getModel(self, _params):
    return LGBMRegressor(
      max_depth= int(_params['max_depth']),
      num_leaves= int((2 ** np.maximum(np.minimum(_params['max_depth'], 12), 2)) * 0.6),
      boosting_type= _params['boosting_type'],
      learning_rate= _params['learning_rate'],
      n_estimators= int(_params['n_estimators']),
      bagging_fraction= _params['bagging_fraction'],
      bagging_freq= _params['bagging_freq'],
      feature_fraction= _params['feature_fraction'],
      max_bin = int(_params['max_bin']),
      min_data_in_leaf = int(_params['min_data_in_leaf']),
      n_jobs= definitions.getNumberOfCore(),
    )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)
