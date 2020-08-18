from sklearn.ensemble import StackingClassifier as Stacking
from utils import definitions
from .. import model, model_regression
from hyperopt import hp

class StackingClassifier(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'StackingClassifier'
    # self.cantidate_job_list = _candidate_job_list
    self.params_list = {}

  def setCandidateJobList(self, _jobs):
    self.cantidate_job_list = _jobs

  def getHyperParameterSpace(self):
    return {
      'max_estimator': hp.quniform('max_estimator', 3, 7, 1),
      'passthrough': hp.choice('passthrough', [False, True]),
    }

  def getModel(self, _params):
    estimator_list = []
    for idx, job in enumerate(self.cantidate_job_list):
      if idx == _params['max_estimator']:
        break
      estimator_list.append((job.getJobName(), job.trained_model))
    return Stacking(
      estimators=estimator_list,
      # verbosity = 0,
      passthrough= _params['passthrough'],
      n_jobs= definitions.getNumberOfCore(),
    )
    
  # def trainModel(self, x, y, _params):
  #   self.model = self.getModel(_params)
  #   self.model.fit(x, y)
  #   self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)
