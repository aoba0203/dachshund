import numpy as np
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
from hyperopt import hp
from utils import definitions
from .. import model, model_classification

class ComplementNBClassifier(model.Model, model_classification.ModelClassification):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'NaiveBayesComplement'
    self.params_list = {}

  def getHyperParameterSpace(self):
    return {
      'alpha': hp.uniform('alpha', 0, 100),
      'norm': [False, True],
    }

  def getModel(self, _params):
    return ComplementNB(
      alpha= _params['alpha'],
      norm= _params['norm'],
    )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)

  def getPredictProbaResult(self, x):
    return self.model.predict_proba(x)
    