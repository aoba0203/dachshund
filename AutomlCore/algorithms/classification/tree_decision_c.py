import numpy as np
from sklearn.tree import DecisionTreeClassifier as DecisionTree
from hyperopt import hp
from utils import definitions
from .. import model, model_classification

class DecisionTreeClassifier(model.Model, model_classification.ModelClassification):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'DecisionTreeClassifier'
    self.params_list = {
      'criterion': ['gini', 'entropy'],
      'splitter': ['best', 'random'],
      'max_features': [None, 'auto', 'sqrt','log2'],
    }

  def getHyperParameterSpace(self):
    return {
      'criterion': hp.choice('criterion', self.params_list['criterion']),
      'splitter': hp.choice('splitter', self.params_list['splitter']),
      # 'max_depth': hp.quniform('max_depth', 2, 24, 2),
      # 'min_samples_split': hp.uniform('min_samples_split', 0, 1),
      # 'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
      'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0, 0.5),
      'max_features': hp.choice('max_features', self.params_list['max_features']),
      # 'max_leaf_nodes': hp.quniform('max_leaf_nodes', 2, 48, 2),
      # 'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 1),
      # 'ccp_alpha': hp.uniform('ccp_alpha', 0, 2),
    }

  def getModel(self, _params):
    if _params['max_features'] == definitions.JSON_NONE:
      _params['max_features'] = None
    return DecisionTree(
      criterion= _params['criterion'],
      splitter= _params['splitter'],
      # max_depth= int(_params['max_depth']),
      # min_samples_split= _params['min_samples_split'],
      # min_samples_leaf= _params['min_samples_leaf'],
      min_weight_fraction_leaf= _params['min_weight_fraction_leaf'],
      max_features= _params['max_features'],
      # max_leaf_nodes= int(_params['max_leaf_nodes']),
      # min_impurity_decrease= _params['min_impurity_decrease'],
      # ccp_alpha= _params['ccp_alpha'],
    )

  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)

  def getPredictProbaResult(self, x):
    return self.model.predict_proba(x)
    