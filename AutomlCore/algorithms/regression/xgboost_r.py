from xgboost import XGBRegressor
from hyperopt import hp
from utils import definitions
from .. import model, model_regression

class XgboostRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'XGBRegressor'
    self.params_list = {
      'booster': ['gbtree', 'gblinear', 'dart'],
    }

  def getHyperParameterSpace(self):
    return {
      'max_depth': hp.quniform('max_depth', -1, 24, 2),
      'booster': hp.choice('booster', self.params_list['booster']),
      'gamma': hp.uniform('gamma', 0, 1),
      'min_child_weight': hp.quniform('min_child_weight', 1, 24, 2),
      'subsample': hp.uniform('subsample', 0, 1),
      'colsample_bytree': hp.uniform('colsample_bytree', 0, 1),
      'eta': hp.uniform('eta', 0, 0.5),
    }

  def getModel(self, _params):
    return XGBRegressor(
      max_depth= int(_params['max_depth']),
      booster= _params['booster'],
      gamma= _params['gamma'],
      min_child_weight= int(_params['min_child_weight']),
      subsample= _params['subsample'],
      colsample_bytree= _params['colsample_bytree'],
      eta= _params['eta'],
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