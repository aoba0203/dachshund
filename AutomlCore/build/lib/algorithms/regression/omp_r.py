from sklearn.linear_model import OrthogonalMatchingPursuit
from hyperopt import hp
from utils import definitions
from .. import model, model_regression

class OmpRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'OrthogonalMatchingPursuit'
    self.params_list = {}

  def getHyperParameterSpace(self):
    return {
      'fit_intercept': hp.choice('fit_intercept', [False, True]),
      'normalize': hp.choice('normalize', [False, True]),
    }

  def getModel(self, _params):
    return OrthogonalMatchingPursuit(
      fit_intercept= _params['fit_intercept'],
      normalize= _params['normalize'],
    )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)
