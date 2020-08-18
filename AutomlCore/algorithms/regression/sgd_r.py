from sklearn.linear_model import SGDRegressor
from hyperopt import hp
from utils import definitions
from .. import model, model_regression

class SgdRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'SGDRegressor'
    self.params_list = {
      'penalty': ['l2', 'l1', 'elasticnet'],
    }

  def getHyperParameterSpace(self):
    return {
      'penalty': hp.choice('penalty', self.params_list['penalty']),
      'alpha': hp.uniform('alpha', 0, 1),
      'l1_ratio': hp.uniform('l1_ratio', 0, 1),
      'fit_intercept': hp.choice('fit_intercept', [False, True]),
      'max_iter': hp.quniform('max_iter', 800, 1200, 20),
      'shuffle': hp.choice('shuffle', [False, True]),
      'power_t': hp.uniform('power_t', 0, 1),
      'early_stopping': hp.choice('early_stopping', [False, True]),
    }

  def getModel(self, _params):
    return SGDRegressor(
      penalty= _params['penalty'], 
      alpha= _params['alpha'],
      l1_ratio= _params['l1_ratio'],
      fit_intercept= bool(_params['fit_intercept']),
      max_iter= int(_params['max_iter']),
      shuffle= bool(_params['shuffle']),
      power_t= _params['power_t'],
      early_stopping= bool(_params['early_stopping']),
    )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)

  