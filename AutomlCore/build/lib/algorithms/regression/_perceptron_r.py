from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from hyperopt import hp
from utils import definitions
from .. import model, model_regression

class PerceptronRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'Perceptron'
    self.params_list = {
      'penalty': ['l2', 'l1', 'elasticnet'],
    }

  def getHyperParameterSpace(self):
    return {
      'penalty': hp.choice('penalty', self.params_list['penalty']),
      'alpha': hp.uniform('alpha', 0, 1),
      'fit_intercept': hp.choice('fit_intercept', [False, True]),
      'max_iter': hp.quniform('max_iter', 800, 1200, 20), 
      'shuffle': hp.choice('shuffle', [False, True]),
      'early_stopping': hp.choice('early_stopping', [False, True]),
      'validation_fraction': hp.uniform('validation_fraction', 0, 1)
    }

  def getModel(self, _params):
    return Perceptron(
      penalty= _params['penalty'], 
      alpha= _params['alpha'],
      fit_intercept= _params['fit_intercept'],
      max_iter= int(_params['max_iter']),
      shuffle= _params['shuffle'],
      early_stopping= _params['early_stopping'],
      validation_fraction= _params['validation_fraction'],
      n_jobs= definitions.getNumberOfCore(),
    )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)
