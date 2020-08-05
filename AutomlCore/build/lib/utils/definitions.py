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

def getBestModelParamsFilePath(_project_name, _model_name, _data_ratio):
  project_path = getProjectResultsPath(_project_name)
  filename = _model_name + '_' + _data_ratio + '.json'
  return os.path.join(project_path, filename)

def getNumberOfCore():
  cpuCount = multiprocessing.cpu_count()
  return np.minimum(4, cpuCount)
