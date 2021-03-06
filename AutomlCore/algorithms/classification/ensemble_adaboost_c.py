import numpy as np
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from hyperopt import hp
from utils import definitions
from .. import model, model_classification

class AdaBoostClassifier(model.Model, model_classification.ModelClassification):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'AdaBoostClassifier'
    self.params_list = {
      'algorithm': ['SAMME', 'SAMME.R'],
    }

  def getHyperParameterSpace(self):
    return {
      'n_estimators': hp.quniform('n_estimators', 50, 200, 5),
      'learning_rate': hp.uniform('learning_rate', 0, 1),
      'algorithm': hp.choice('algorithm', self.params_list['algorithm']),
    }

  def getModel(self, _params):
    return AdaBoost(
      n_estimators= int(_params['n_estimators']),
      learning_rate= _params['learning_rate'],
      algorithm= _params['algorithm']
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
    return 2 ** 3