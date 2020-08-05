from sklearn.linear_model import LinearRegression
from hyperopt import hp
from utils import definitions
from .. import model, model_regression

class LinearRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'LinearRegressor'
    self.params_list = {}

  def getHyperParameterSpace(self):
    params = {
      'fit_intercept': hp.choice('fit_intercept', [False, True]),
      'normalize': hp.choice('normalize', [False, True]),
      'copy_X': hp.choice('copy_X', [False, True]),
    }
    return params
  
  def getModel(self, _params):
    return LinearRegression(
      fit_intercept= _params['fit_intercept'],
      normalize= _params['normalize'],
      copy_X= _params['copy_X'],
      # n_jobs= definitions.getNumberOfCore(),
    )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)
