import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GradientBoosting
from hyperopt import hp
from utils import definitions
from .. import model, model_classification

class GradientBoostingClassifier(model.Model, model_classification.ModelClassification):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'GradientBoostingClassifier'
    self.params_list = {
      # 'loss': ['deviance', 'exponential'],
      'criterion': ['friedman_mse', 'mse', 'mae'],
      'max_features': [None, 'auto', 'sqrt','log2'],
    }

  def getHyperParameterSpace(self):
    return {
      # 'loss': hp.choice('loss', self.params_list['loss']),      
      'learning_rate': hp.uniform('learning_rate', 0, 1),
      'n_estimators': hp.quniform('n_estimators', 50, 200, 5),
      # 'subsample': hp.uniform('subsample', 0.5, 1),
      'criterion': hp.choice('criterion', self.params_list['criterion']),
      # 'min_samples_split': hp.uniform('min_samples_split', 0, 1),
      # 'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
      # 'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0, 0.5),
      'max_depth': hp.quniform('max_depth', 2, 24, 2),
      # 'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 1),
      'max_features': hp.choice('max_features', self.params_list['max_features']),
      # 'ccp_alpha': hp.uniform('ccp_alpha', 0, 2),
    }

  def getModel(self, _params):
    return GradientBoosting(
      # loss= _params['loss'],
      learning_rate= _params['learning_rate'],
      n_estimators= int(_params['n_estimators']),
      # subsample= _params['subsample'],
      criterion= _params['criterion'],
      # min_samples_split = _params['min_samples_split'],
      # min_samples_leaf = _params['min_samples_leaf'],
      # min_weight_fraction_leaf= _params['min_weight_fraction_leaf'],
      max_depth= _params['max_depth'],
      # min_impurity_decrease= _params['min_impurity_decrease'],
      max_features= _params['max_features'],
      # ccp_alpha = _params['ccp_alpha'],
      warm_start=True,
    )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)

  def getPredictProbaResult(self, x):
    return self.model.predict_proba(x)
 
    