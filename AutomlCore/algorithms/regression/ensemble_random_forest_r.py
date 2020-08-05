import numpy as np
from sklearn.ensemble import RandomForestRegressor as RandomForest
from hyperopt import hp
from sklearn.utils import class_weight
from utils import definitions
from .. import model, model_regression

class RandomForestRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'RandomForestRegressor'
    self.params_list = {
      # 'criterion': ['mse', 'mae'],
      'max_depth': np.hstack([[None], np.arange(2, 48,2)]),
      'max_features': [None, 'auto', 'sqrt','log2'],
    }

  def getHyperParameterSpace(self):
    return {
      'n_estimators': hp.quniform('n_estimators', 50, 200, 5),
      # 'criterion': hp.choice('criterion', self.params_list['criterion']),
      'max_depth': hp.choice('max_depth', self.params_list['max_depth']),
      # 'min_samples_split': hp.uniform('min_samples_split', 0, 1),
      # 'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
      'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0, 0.5),
      'max_features': hp.choice('max_features', self.params_list['max_features']),
      # 'max_leaf_nodes': hp.quniform('max_leaf_nodes', 2, 48, 2),
      # 'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 1),
      # 'bootstrap': hp.choice('bootstrap', [False, True]),
      'oob_score': hp.choice('oob_score', [False, True]),
      # 'ccp_alpha': hp.uniform('ccp_alpha', 0, 2),
    }

  def getModel(self, _params):
    return RandomForest(
      n_estimators= int(_params['n_estimators']),
      # criterion= _params['criterion'],
      max_depth= _params['max_depth'],
      # min_samples_split= _params['min_samples_split'],
      # min_samples_leaf= _params['min_samples_leaf'],
      min_weight_fraction_leaf= _params['min_weight_fraction_leaf'],
      max_features= _params['max_features'],
      # max_leaf_nodes= int(_params['max_leaf_nodes']),
      # min_impurity_decrease= _params['min_impurity_decrease'],
      # bootstrap= _params['bootstrap'],
      oob_score= _params['oob_score'],
      # ccp_alpha= _params['ccp_alpha'],
      n_jobs= definitions.getNumberOfCore(),
    )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)

  def getPredictProbaResult(self, x):
    return self.model.predict_proba(x)
    