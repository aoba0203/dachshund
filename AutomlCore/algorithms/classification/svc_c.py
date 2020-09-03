import numpy as np
from sklearn.svm import SVC
from hyperopt import hp
from utils import definitions
from .. import model, model_classification

class SvcClassifier(model.Model, model_classification.ModelClassification):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'C-SupportVectorClassifier'
    self.params_list = {
      'kernel': ['linear', 'rbf', 'sigmoid'],
      'degree': np.arange(1, 6),
      'class_weight': ['balanced', None],
    }

  def getHyperParameterSpace(self):
    return {
      'C': hp.uniform('C', 0, 1), 
      # 'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']),
      'kernel': hp.choice('kernel', self.params_list['kernel']),
      'degree': hp.choice('degree', self.params_list['degree']),
      'gamma': hp.uniform('gamma', 0, 1),
      'coef0': hp.uniform('coef0', 0, 1),
      'shrinking': hp.choice('shrinking', [False, True]),
      'class_weight': hp.choice('class_weight', self.params_list['class_weight'])
    }

  def getModel(self, _params):
    if _params['class_weight'] == definitions.JSON_NONE:
      _params['class_weight'] = None
    return SVC(
      C= _params['C'],
      kernel= _params['kernel'],
      degree= _params['degree'],
      gamma= _params['gamma'],
      coef0= _params['coef0'],
      shrinking= bool(_params['shrinking']),
      class_weight= _params['class_weight'],
      cache_size= 2000,
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
    
    