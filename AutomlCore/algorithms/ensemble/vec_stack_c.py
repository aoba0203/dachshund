from vecstack import stacking
from utils import definitions
from .. import model, model_regression
from hyperopt import hp
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
# https://github.com/vecxoz/vecstack/blob/master/examples/04_sklearn_api_regression_pipeline.ipynb

class VecStackingClassifier(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'VecStackingClassifier'
    # self.cantidate_job_list = _candidate_job_list
    self.params_list = {
      'mode': ['oof_pred', 'oof_pred_bag'],
    }

  def setCandidateJobList(self, _jobs):
    self.cantidate_job_list = _jobs

  def getHyperParameterSpace(self):
    return {
      'max_estimator': hp.quniform('max_estimator', 3, 7, 1),
      'mode': hp.quniform('mode', self.params_list['mode']),
      'stratified': hp.choice('stratified', [False, True]),
      'shuffle': hp.choice('shuffle', [False, True]),
    }

  def getModel(self, _params, _x, _y, _x_eval):
    estimator_list = []
    for idx, job in enumerate(self.cantidate_job_list):
      if idx == _params['max_estimator']:
        break
      estimator_list.append(job.model.getModel(job.best_params))
    s_train, s_test = stacking(
      estimator_list,
      _x, _y, _x_eval,
      regression=False,
      metric=accuracy_score,
      stratified=_params['stratified'],
      shuffle=_params['shuffle'],
      random_state=0,
      n_jobs= definitions.getNumberOfCore(),
    )


    
  # def trainModel(self, x, y, _params):
  #   self.model = self.getModel(_params)
  #   self.model.fit(x, y)
  #   self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)
