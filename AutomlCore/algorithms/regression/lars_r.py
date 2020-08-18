from sklearn.linear_model import Lars
from hyperopt import hp
from utils import definitions
from .. import model, model_regression

class LarsRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'LearAngleRegressor'
    self.params_list = {}

  def getHyperParameterSpace(self):
    return {
      'fit_intercept': hp.choice('fit_intercept', [False, True]),
      'normalize': hp.choice('normalize', [False, True]),
      'eps': hp.uniform('eps', 0, 1.0),
      'copy_X': hp.choice('copy_X', [False, True]),
    }

  def getModel(self, _params):
    return Lars(
      fit_intercept= _params['fit_intercept'],
      normalize= _params['normalize'],
      eps= _params['eps'],
      copy_X= _params['copy_X'],
    )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)

  def getMaxIterCount(self):    
    return 2 ** 3