import utils
from xgboost import XGBClassifier
from hyperopt import hp
from utils import definitions
from .. import model, model_classification

class XgboostClassifier(model.Model, model_classification.ModelClassification):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'XGBClassifier'
    self.params_list = {
      'booster': ['gbtree', 'gblinear', 'dart'],
    }

  def getHyperParameterSpace(self):
    return {
      'booster': hp.choice('booster', self.params_list['booster']),
      'eta': hp.uniform('eta', 0, 1),
      'gamma': hp.uniform('gamma', 0, 24),      
      'max_depth': hp.quniform('max_depth', 3, 12, 1),
      'min_child_weight': hp.uniform('min_child_weight', 0, 24),
      'max_delta_step': hp.uniform('max_delta_step', 0, 12),
      'subsample': hp.uniform('subsample', 0.5, 1),
      'colsample_bytree': hp.uniform('colsample_bytree', 0, 1),
      'colsample_bylevel': hp.uniform('colsample_bylevel', 0, 1),
      'colsample_bynode': hp.uniform('colsample_bynode', 0, 1),
      'reg_lambda': hp.uniform('reg_lambda', 0, 3),
      'alpha': hp.uniform('alpha', 0, 3),
    }

  def getModel(self, _params):
    return XGBClassifier(
      booster= _params['booster'],
      eta= _params['eta'],
      gamma= _params['gamma'],
      max_depth= int(_params['max_depth']),
      min_child_weight= int(_params['min_child_weight']),
      max_delta_step= int(_params['max_delta_step']),      
      subsample= _params['subsample'],      
      colsample_bytree= _params['colsample_bytree'],
      colsample_bylevel= _params['colsample_bylevel'],
      colsample_bynode= _params['colsample_bynode'],
      reg_lambda= _params['reg_lambda'],
      alpha= _params['alpha'],
      verbosity = 0,
      n_jobs= definitions.getNumberOfCore(),
    )
    
  def trainModel(self, x, y, _params):
    self.model = self.getModel(_params)
    self.model.fit(x, y)
    self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)

  def getMaxIterCount(self):    
    return 2 ** 6