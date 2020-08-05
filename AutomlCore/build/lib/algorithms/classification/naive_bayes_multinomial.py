import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from hyperopt import hp
from utils import definitions
from .. import model, model_classification

class MultinomialNbClassifier(model.Model, model_classification.ModelClassification):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'NaiveBayesMultinomial'
    self.params_list = {}

  def getHyperParameterSpace(self):
    return {
      'alpha': hp.uniform('alpha', 0, 100),
    }

  def getModel(self, _params):
    return MultinomialNB(
      alpha= _params['alpha'],
    )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)

  def getPredictProbaResult(self, x):
    return self.model.predict_proba(x)
    