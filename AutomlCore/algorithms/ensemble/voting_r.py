from sklearn.ensemble import VotingRegressor as Voting
from utils import definitions
from .. import model, model_regression
from hyperopt import hp

class VotingRegressor(model.Model, model_regression.ModelRegression):
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'VotingRegressor'
    # self.cantidate_job_list = _candidate_job_list
    self.params_list = {}

  def setCandidateJobList(self, _jobs):
    self.cantidate_job_list = _jobs

  def getHyperParameterSpace(self):
    return {
      'max_estimator': hp.quniform('max_estimator', 3, 7, 1),
    }

  def getModel(self, _params):
    estimator_list = []
    for idx, job in enumerate(self.cantidate_job_list):
      if idx == _params['max_estimator']:
        break
      estimator_list.append((job.getJobName(), job.trained_model))
    return Voting(
      estimators=estimator_list,
      # verbosity = 0,
      n_jobs= definitions.getNumberOfCore(),
    )
    
  # def trainModel(self, x, y, _params):
  #   self.model = self.getModel(_params)
  #   self.model.fit(x, y)
  #   self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)
