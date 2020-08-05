from sklearn.linear_model import TweedieRegressor as twRegressor
from hyperopt import hp
from utils import definitions
from .. import model, model_regression

class TweedieRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'TweedieRegressor'
    self.params_list = {}

  def getHyperParameterSpace(self):
    return {
      'alpha': hp.uniform('alpha', 0, 1),
      'fit_intercept': hp.choice('fit_intercept', [False, True]),
      'max_iter': hp.quniform('max_iter', 50, 150, 10), 
    }

  def getModel(self, _params):
    return twRegressor(
      alpha= _params['alpha'],
      fit_intercept= _params['fit_intercept'],
      max_iter= int(_params['max_iter']),
    )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)
