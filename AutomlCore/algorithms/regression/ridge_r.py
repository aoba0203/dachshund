from sklearn.linear_model import Ridge
from hyperopt import hp
from utils import definitions
from .. import model, model_regression

class RidgeRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'RidgeRegressor'
    self.params_list = {}

  def getHyperParameterSpace(self):
    return {
      'alpha': hp.uniform('alpha', 0, 1),
      'fit_intercept': hp.choice('fit_intercept', [False, True]),
      'normalize': hp.choice('normalize', [False, True]),
      'copy_X': hp.choice('copy_X', [False, True]),
    }

  def getModel(self, _params):
    return Ridge(
      alpha= _params['alpha'],
      fit_intercept= bool(_params['fit_intercept']),
      normalize= bool(_params['normalize']),
      copy_X= bool(_params['copy_X']),      
    )
  
  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)

