from sklearn.linear_model import SGDClassifier
from hyperopt import hp
from utils import definitions
from .. import model, model_classification

class SgdClassifier(model.Model, model_classification.ModelClassification):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'SGDClassifier'
    self.params_list = {
      'penalty': ['l2', 'l1', 'elasticnet'],
      'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
      'class_weight': ['balanced', None],
    }

  def getHyperParameterSpace(self):
    return {
      'penalty': hp.choice('penalty', self.params_list['penalty']),
      'alpha': hp.uniform('alpha', 0, 0.1), 
      'l1_ratio': hp.uniform('l1_ratio', 0, 1),
      'fit_intercept': hp.choice('fit_intercept', [False, True]),
      'max_iter': hp.quniform('max_iter', 800, 1600, 10),
      'shuffle': hp.choice('shuffle', [False, True]),
      'learning_rate': hp.choice('learning_rate', self.params_list['learning_rate']),
      'eta0': hp.uniform('eta0', 0, 1),
      'early_stopping': hp.choice('early_stopping', [False, True]),
      'class_weight': hp.choice('class_weight', self.params_list['class_weight'])
    }

  def getModel(self, _params):
    if _params['class_weight'] == definitions.JSON_NONE:
      _params['class_weight'] = None
    return SGDClassifier(
      penalty= _params['penalty'],
      alpha= _params['alpha'],
      l1_ratio= _params['l1_ratio'],
      fit_intercept= bool(_params['fit_intercept']),
      max_iter= int(_params['max_iter']),
      shuffle= bool(_params['shuffle']),
      learning_rate= _params['learning_rate'],
      eta0 = _params['eta0'],
      early_stopping= bool(_params['early_stopping']),
      class_weight= _params['class_weight'],
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
  
  def getMaxIterCount(self):    
    return 2 ** 6