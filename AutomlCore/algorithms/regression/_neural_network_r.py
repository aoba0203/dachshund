import numpy as np
from sklearn.neural_network import MLPRegressor
from hyperopt import hp
from utils import definitions
from .. import model, model_regression

class NeuralNetworkRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'NeuralNetwork'
    self.params_list = {
      'activation': ['identity', 'logistic', 'tanh', 'relu'],
      'solver': ['sgd', 'adam'], 
      'learning_rate': ['constant', 'invscaling', 'adaptive'],
    }

  def getHyperParameterSpace(self):
    return {
      'layer_1': hp.quniform('layer_1', 12, 64, 4),
      'layer_2': hp.quniform('layer_2', 12, 64, 4),
      'layer_3': hp.quniform('layer_3', 12, 64, 4),
      'activation': hp.choice('activation', self.params_list['activation']),
      'solver': hp.choice('solver', self.params_list['solver']),
      'alpha': hp.uniform('alpha', 0, 1),
      'learning_rate': hp.choice('learning_rate', self.params_list['learning_rate']),
      'max_iter': hp.quniform('max_iter', 150, 250, 10),
      'validation_fraction': hp.uniform('validation_fraction', 0, 1),
      'beta_1': hp.uniform('beta_1', 0, 1),
      'beta_2': hp.uniform('beta_2', 0, 1),      
    }

  def getModel(self, _params):
    return MLPRegressor(
      hidden_layer_sizes=(int(_params['layer_1']), int(_params['layer_2']), int(_params['layer_3'])),
      activation=_params['activation'], 
      solver= _params['solver'],
      alpha= _params['alpha'],
      learning_rate= _params['learning_rate'],
      max_iter= int(_params['max_iter']),
      validation_fraction= _params['validation_fraction'],
      beta_1= _params['beta_1'],
      beta_2= _params['beta_2'],
    )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    pred = self.model.predict(x)
    if sum(np.isnan(pred)) > 0:
      pred = np.nan_to_num(pred)
    return pred
