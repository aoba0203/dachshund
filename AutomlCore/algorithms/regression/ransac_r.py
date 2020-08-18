from sklearn.linear_model import RANSACRegressor
from hyperopt import hp
from utils import definitions
from .. import model, model_regression

class RansacRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'RANSACRegressor'
    self.params_list = {}

  def getHyperParameterSpace(self):
    return {
      'min_samples': hp.uniform('min_samples', 0, 1),
      'residual_threshold': hp.uniform('residual_threshold', 0, 1),      
    }

  def getModel(self, _params):
    return RANSACRegressor(
      min_samples= _params['min_samples'],
      residual_threshold= _params['residual_threshold'],
    )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)

  def getMaxIterCount(self):    
    return 2 ** 2