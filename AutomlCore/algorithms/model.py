from utils import definitions
import numpy as np
import joblib
import os

class Model:
  def __init__(self, _project_name):
    self.project_name = _project_name
    self.results_path = definitions.getProjectResultsPath(_project_name)
    self.model_name = 'model'
    self.model = None
    self.max_eval_step = 30
    self.params_list = {}
    self.params_best = {}
  
  def __getModelFile(self):
    return os.path.join(self.results_path, ('model_' + self.model_name + '.joblib'))

  def saveModel(self):
    if self.model:
      model_file = self.__getModelFile()
      joblib.dump(self.model, model_file)
    else:
      print('Model is None, should train the model')
    return
  
  def loadModel(self):
    model_file = self.__getModelFile()
    if os.path.exists(model_file):
      self.model = joblib.load(model_file)
    else:
      print('Model file is None, should save the model')
    return None

  def setBestParams(self, _bestParamsDic):
    for key, value in zip(self.params_list.keys(), self.params_list.values()):
      idx_params = _bestParamsDic[key]
      _bestParamsDic[key] = value[idx_params]
    self.params_best = _bestParamsDic
  
  def getMaxIterCount(self):
    params_count = len(self.getHyperParameterSpace())
    return 2 ** np.minimum(9, (params_count + 2))
