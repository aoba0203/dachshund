import os
import numpy as np
from pathlib import Path
import multiprocessing

def getProjectRootPath():
  file = Path(os.path.abspath('utils.py'))
  parent= file.parent
  # return os.path.dirname(os.path.abspath(__file__))
  return parent

def getEdaPath():
  rootPath = getProjectRootPath()
  return os.path.join(rootPath, 'eda')

def getPreprocessPath():
  rootPath = getProjectRootPath()
  return os.path.join(rootPath, 'preprocess')

def getProjectResultsPath(_project_name):
  rootPath = getProjectRootPath()
  resultsPath = os.path.join(rootPath, 'results')
  project_path = os.path.join(resultsPath, _project_name)
  if not os.path.exists(project_path):
    os.makedirs(project_path)
  return project_path

def __getBestModelFilePath(_project_name, _model_name, _data_ratio, _extends):
  project_path = getProjectResultsPath(_project_name)
  filename = _model_name + '_' + str(_data_ratio) + _extends
  return os.path.join(project_path, filename)

def getBestModelParamsFilePath(_project_name, _model_name, _data_ratio):
  return __getBestModelFilePath(_project_name, _model_name, _data_ratio, '.json')

def getBestModelFilePath(_project_name, _model_name, _data_ratio):
  return __getBestModelFilePath(_project_name, _model_name, _data_ratio, '.joblib')

def getNumberOfCore():
  cpuCount = multiprocessing.cpu_count()
  return np.minimum(4, cpuCount)
