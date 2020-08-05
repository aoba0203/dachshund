from sklearn.linear_model import ARDRegression
from hyperopt import hp
from utils import definitions
from .. import model, model_regression

class ARDRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'ARDRegression'
    self.params_list = {}

  def getHyperParameterSpace(self):
    return {
      'n_iter': hp.quniform('n_iter', 200, 400, 12), 
      'alpha_1': hp.uniform('alpha_1', 0, 1),
      'alpha_2': hp.uniform('alpha_2', 0, 1),
      'lambda_1': hp.uniform('lambda_1', 0, 1),
      'lambda_2': hp.uniform('lambda_2', 0, 1),
      'compute_score': hp.choice('compute_score', [False, True]),
      'fit_intercept': hp.choice('fit_intercept', [False, True]),
      'normalize': hp.choice('normalize', [False, True]),
      'copy_X': hp.choice('copy_X', [False, True]),
    }

  def getModel(self, _params):
    return ARDRegression(
      n_iter= int(_params['n_iter']),
      alpha_1 = _params['alpha_1'],
      alpha_2 = _params['alpha_2'],
      lambda_1 = _params['lambda_1'],
      lambda_2 = _params['lambda_2'],      
      compute_score= _params['compute_score'],
      fit_intercept= _params['fit_intercept'],
      normalize= _params['normalize'],
      copy_X= _params['copy_X'],      
      )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)

