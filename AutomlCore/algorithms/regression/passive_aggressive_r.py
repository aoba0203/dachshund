from sklearn.linear_model import PassiveAggressiveRegressor as passiveAggressive
from hyperopt import hp
from utils import definitions
from .. import model, model_regression

class PassiveAggresiveRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'PassiveAggressiveRegressor'
    self.params_list = {}

  def getHyperParameterSpace(self):
    return {
      'fit_intercept': hp.choice('fit_intercept', [False, True]),
      'max_iter': hp.quniform('max_iter', 800, 1200, 20),
      'early_stopping': hp.choice('early_stopping', [False, True]),
      'validation_fraction': hp.uniform('validation_fraction', 0, 1),
      'shuffle': hp.choice('shuffle', [False, True]),
    }

  def getModel(self, _params):
    return passiveAggressive(
      fit_intercept= _params['fit_intercept'],
      max_iter= int(_params['max_iter']),
      early_stopping= _params['early_stopping'],
      validation_fraction= _params['validation_fraction'],
      shuffle= _params['shuffle'],
    )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)
