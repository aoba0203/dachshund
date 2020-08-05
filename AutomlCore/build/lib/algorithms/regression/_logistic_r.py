from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from hyperopt import hp
from utils import definitions
from .. import model, model_regression

class LogisticRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'LogisticREgressor'
    self.params_list = {
      'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
    }

  def getHyperParameterSpace(self):
    return {
      'penalty': 'l2',
      'dual': hp.choice('dual', [False, True]),
      'C': hp.uniform('C', 0, 1),
      'fit_intercept': hp.choice('fit_intercept', [False, True]),
      'intercept_scaling': hp.uniform('intercept_scaling', 0, 1),
      'solver': hp.choice('solver', self.params_list['solver']),
      'max_iter': hp.quniform('max_iter', 50, 150, 10), 
      'l1_ratio': hp.uniform('l1_ratio', 0, 1),
    }

  def getModel(self, _params):
    return LogisticRegression(
      penalty= _params['penalty'], 
      dual= _params['dual'],
      C= _params['C'],
      fit_intercept= _params['fit_intercept'],
      intercept_scaling = _params['intercept_scaling'],
      solver= _params['solver'],
      max_iter= int(_params['max_iter']),
      l1_ratio= _params['l1_ratio'],
      n_jobs= definitions.getNumberOfCore(),
    )
  
  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)
