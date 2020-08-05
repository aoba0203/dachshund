from sklearn.linear_model import LassoLars
from hyperopt import hp
from utils import definitions
from .. import model, model_regression

class LARSLassoRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'LARSLassoRegressor'
    self.params_list = {}

  def getHyperParameterSpace(self):
    return {
      'alpha': hp.uniform('alpha', 0, 1),
      'fit_intercept': hp.choice('fit_intercept', [False, True]),
      'normalize': hp.choice('normalize', [False, True]),
      'eps': hp.uniform('eps', 0, 1),
      'copy_X': hp.choice('copy_X', [False, True]),
      'positive': hp.choice('positive', [False, True]),
    }

  def getModel(self, _params):
    return LassoLars(
      alpha= _params['alpha'],
      fit_intercept= _params['fit_intercept'],
      normalize= _params['normalize'],
      eps= _params['eps'],
      copy_X= _params['copy_X'],
      positive= _params['positive'],
    )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)
