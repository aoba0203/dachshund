import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier as HistGradientBoosting
from sklearn.metrics import accuracy_score
from hyperopt import hp
from utils import definitions
from .. import model, model_classification

class HistGradientBoostingClassifier(model.Model, model_classification.ModelClassification):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'HistGradientBoostingClassifier'
    self.params_list = {
      'max_leaf_nodes': np.hstack([[None], np.arange(24, 64)]),
      'max_depth': np.hstack([[None], np.arange(1, 24)]),
    }

  def getHyperParameterSpace(self):
    return {
      'learning_rate': hp.uniform('learning_rate', 0, 1),
      'max_iter': hp.quniform('max_iter', 80, 200, 5),
      'max_leaf_nodes': hp.choice('max_leaf_nodes', self.params_list['max_leaf_nodes']),
      'max_depth': hp.choice('max_depth', self.params_list['max_depth']),
      'min_samples_leaf': hp.quniform('min_samples_leaf', 15, 30, 1),
      'l2_regularization': hp.uniform('l2_regularization', 0, 1),
      'max_bins': hp.quniform('max_bins', 100, 255, 5),
    }

  def getModel(self, _params):
    return HistGradientBoosting(
      learning_rate= _params['learning_rate'],
      max_iter= int(_params['max_iter']),
      max_leaf_nodes= _params['max_leaf_nodes'],
      max_depth= _params['max_depth'],
      min_samples_leaf = _params['min_samples_leaf'],
      l2_regularization= _params['l2_regularization'],
      max_bins= _params['max_bins'],
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
    