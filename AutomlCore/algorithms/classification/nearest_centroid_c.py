import numpy as np
from sklearn.neighbors import NearestCentroid
from hyperopt import hp
from utils import definitions
from .. import model, model_classification

class NearestCentroidClassifier(model.Model, model_classification.ModelClassification):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'NearestCentroidClassifier'
    self.params_list = {}

  def getHyperParameterSpace(self):
    return {
      'shrink_threshold': hp.uniform('shrink_threshold', 0, 1),
    }

  def getModel(self, _params):
    return NearestCentroid(
      shrink_threshold= _params['shrink_threshold'],
      # n_jobs= definitions.getNumberOfCore(),
      )
    
  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)

  def getMaxIterCount(self):    
    return 2 ** 2