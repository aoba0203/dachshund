import numpy as np
from sklearn.naive_bayes import GaussianNB
from hyperopt import hp
from utils import definitions
from .. import model, model_classification

class GaussianNbClassifier(model.Model, model_classification.ModelClassification):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'NaiveBayesGaussian'
    self.params_list = {}

  def getHyperParameterSpace(self):
    return {
      'var_smoothing': hp.uniform('var_smoothing', 0, 12),
    }

  def getModel(self, _params):
    return GaussianNB(
      var_smoothing= _params['var_smoothing'],
    )
  
  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)

  def getPredictProbaResult(self, x):
    return self.model.predict_proba(x)
    
  def getMaxIterCount(self):    
    return 2 ** 2