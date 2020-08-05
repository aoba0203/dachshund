import numpy as np
from sklearn.svm import LinearSVC
from hyperopt import hp
from utils import definitions
from .. import model, model_classification

class LinearSvcClassifier(model.Model, model_classification.ModelClassification):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'LinearSupportVectorClassifier'
    self.params_list = {
      'penalty': ['l1', 'l2'],
      'class_weight': ['balanced', None],
    }

  def getHyperParameterSpace(self):
    return {
      'penalty': hp.choice('penalty', self.params_list['penalty']),
      'C': hp.uniform('C', 0, 1), 
      'fit_intercept': hp.choice('fit_intercept', [False, True]),
      'intercept_scaling': hp.uniform('intercept_scaling', 0, 1),
      'class_weight': hp.choice('class_weight', self.params_list['class_weight']),
      'max_iter': hp.quniform('max_iter', 800, 1600, 10),
    }

  def getModel(self, _params):
    return LinearSVC(
      penalty= _params['penalty'],
      C= _params['C'],
      fit_intercept= _params['fit_intercept'],
      class_weight= _params['class_weight'],
      max_iter= int(_params['max_iter']),
      # n_jobs= definitions.getNumberOfCore(),
    )
  
  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)

  def getPredictProbaResult(self, x):
    return self.model.predict_proba(x)
    