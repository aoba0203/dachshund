import numpy as np
from lightgbm import LGBMClassifier
from hyperopt import hp
from utils import definitions
from .. import model, model_classification

class LightGbmClassifier(model.Model, model_classification.ModelClassification):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'LightGBMClassifier'
    self.params_list = {
      'boosting_type': ['gbdt', 'goss', 'rf', 'dart'],
    }

  def getHyperParameterSpace(self):
    return {
      'boosting_type': hp.choice('boosting_type', self.params_list['boosting_type']),
      'max_depth': hp.quniform('max_depth', -1, 24, 2),      
      'learning_rate': hp.uniform('learning_rate', 0, 0.1),
      'n_estimators': hp.quniform('n_estimators', 80, 200, 2),
      'subsample_for_bin': hp.quniform('subsample_for_bin', 150000, 250000, 20),
      'min_split_gain': hp.uniform('min_split_gain', 0, 1),
      'min_child_weight': hp.uniform('min_child_weight', 0, 1),
      'min_child_samples': hp.quniform('min_child_samples', 15, 25, 1),
      'reg_alpha': hp.uniform('reg_alpha', 0, 1),
      'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    }

  def getModel(self, _params):
    return LGBMClassifier(
      boosting_type= _params['boosting_type'],
      num_leaves= int((2 ** np.maximum(np.minimum(_params['max_depth'], 12), 2)) * 0.6),
      max_depth= int(_params['max_depth']),
      learning_rate= _params['learning_rate'],
      n_estimators= int(_params['n_estimators']),
      subsample_for_bin= int(_params['subsample_for_bin']),
      min_split_gain= _params['min_split_gain'],
      min_child_weight= _params['min_child_weight'],
      min_child_samples= int(_params['min_child_samples']),
      reg_alpha= _params['reg_alpha'],
      reg_lambda= _params['reg_lambda'],      
      n_jobs= definitions.getNumberOfCore(),
    )
    
  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)

  def getMaxIterCount(self):    
    return 2 ** 6