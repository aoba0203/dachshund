from sklearn.metrics import accuracy_score
from dask.distributed import Client
from utils import definitions
import joblib

class ModelClassification:
  def __init__(self, _project_name):
    super().__init__(_project_name)
    self.metrics_name = 'Accucary'
  
  def getTrainResults(self, _x, _y, _x_eval, _y_eval, _params, _for_optimize=True, scorer=accuracy_score):
    _params['verbose'] = 0
    model = self.getModel(_params)
    # client = Client(processes=False)
    # with joblib.parallel_backend('dask', n_jobs=definitions.getNumberOfCore()):
    model.fit(_x, _y)
    pred = model.predict(_x_eval)
    if _for_optimize:
      return (accuracy_score(_y_eval, pred) * -1), model
    else:
      # return accuracy_score(_y_eval, pred), model
      return (accuracy_score(_y_eval, pred) * -1), model

  def getTrainedModelScore(self, _model, _x_eval, _y_eval, _for_optimize=True, scorer=accuracy_score):    
    pred = _model.predict(_x_eval)
    if _for_optimize:
      return (accuracy_score(_y_eval, pred) * -1)
    else:
      # return accuracy_score(_y_eval, pred)
      return (accuracy_score(_y_eval, pred) * -1)
