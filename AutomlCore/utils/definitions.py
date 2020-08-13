import os
import numpy as np
from pathlib import Path
import multiprocessing

PROBLEM_TYPE_CLASSIFICATION = 0
PROBLEM_TYPE_REGRESSION = 1

KEY_FEATURE_ADD_NAME = 'feature_add'
KEY_FEATURE_MIS_NAME = 'feature_missing'
KEY_FEATURE_OUT_NAME = 'feature_outlier'
KEY_FEATURE_SCA_NAME = 'feature_scaler'
KEY_FEATURE_SEL_NAME = 'feature_selection'
KEY_FEATURE_SEL_RATE_NAME = 'feature_selection_rate'
KEY_FEATURE_ADD_NAME_LIST = 'feature_add_name'
KEY_FEATURE_MIS_NAME_LIST = 'feature_missing_name'
KEY_FEATURE_OUT_NAME_LIST = 'feature_outlier_name'
KEY_FEATURE_SCA_NAME_LIST = 'feature_scaler_name'
KEY_FEATURE_SEL_NAME_LIST = 'feature_selection_name'
KEY_FEATURE_SEL_COL_LIST = 'feature_selection_columns'

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

def getResultsFilePath(_project_name):
  project_path = getProjectResultsPath(_project_name)
  filename = 'automl_' + _project_name + '.csv'
  return os.path.join(project_path, filename)

def getNumberOfCore():
  cpuCount = multiprocessing.cpu_count()
  return np.minimum(4, cpuCount)
