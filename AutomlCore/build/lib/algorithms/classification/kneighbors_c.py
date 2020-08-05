import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from hyperopt import hp
from utils import definitions
from .. import model, model_classification

class KneighborsClassifier(model.Model, model_classification.ModelClassification):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'KNeighborsClassifier'
    self.params_list = {
      'n_neighbors': np.arange(1, 10),
      'weights': ['uniform', 'distance'],
      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
      'p': np.arange(1, 6),
    }

  def getHyperParameterSpace(self):
    return {
      'n_neighbors': hp.choice('n_neighbors', self.params_list['n_neighbors']),
      'weights': hp.choice('weights', self.params_list['weights']),
      'algorithm': hp.choice('algorithm', self.params_list['algorithm']),
      'leaf_size': hp.quniform('leaf_size', 20, 40, 1),
      'p': hp.choice('p', self.params_list['p']),
    }

  def getModel(self, _params):
    return KNeighborsClassifier(
      n_neighbors= _params['n_neighbors'],
      weights= _params['weights'],
      algorithm= _params['algorithm'],
      leaf_size= _params['leaf_size'],
      p= _params['p'],
      n_jobs= definitions.getNumberOfCore(),
      )
  
  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)

  def getPredictProbaResult(self, x):
    return self.model.predict_proba(x)
    