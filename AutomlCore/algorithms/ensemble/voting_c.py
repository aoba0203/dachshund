from sklearn.ensemble import VotingClassifier as Voting
from utils import definitions
from .. import model, model_classification
from hyperopt import hp

class VotingClassifier(model.Model, model_classification.ModelClassification):
  # def __init__(self, _candidate_job_list):
  #   super(VotingClassifier, self).__init__()
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.model_name = 'VotingClassifier'
    # self.cantidate_job_list = _candidate_job_list
    self.params_list = {}

  def setCandidateJobList(self, _jobs):
    self.cantidate_job_list = _jobs

  def getHyperParameterSpace(self):
    return {
      'max_estimator': hp.quniform('max_estimator', 3, 7, 1),
      'voting': hp.choice('voting', ['hard', 'soft']),
      'flatten_transform': hp.choice('flatten_transform', [False, True]),
    }

  def getModel(self, _params):
    estimator_list = []
    for idx, job in enumerate(self.cantidate_job_list):
      if idx == _params['max_estimator']:
        break
      estimator_list.append(job.model.model_name, job.trained_model)
    return Voting(
      estimators=estimator_list,
      voting=_params['voting'],
      flatten_transform= bool(_params['flatten_transform']),
      verbosity = 0,
      n_jobs= definitions.getNumberOfCore(),
    )
    
  # def trainModel(self, x, y, _params):
  #   self.model = self.getModel(_params)
  #   self.model.fit(x, y)
  #   self.saveModel()
  
  def getPredictResult(self, x):
    return self.model.predict(x)
