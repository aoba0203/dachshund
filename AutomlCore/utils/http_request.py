import requests
import json
from utils import definitions

PREFIX_INFO = 'info/'
PREFIX_DETAIL = 'detail/'

URL = 'http://127.0.0.1:8000/pjt/'

def getHttp(_prefix, _project_name=''):  
  url = URL + _prefix + '?project=' + _project_name
  # print('getHttp: ', url)
  response = requests.get(url)
  # print('getHttp - response: ', response)
  return response.status_code, response.text

def postHttp(_prefix, _data):
  url = URL + _prefix
  # print('postHttp: ', url, ', data: ', _data)
  response = requests.post(url, _data)
  # print('postHttp - response: ', response)
  return response.status_code, response.text

def putHttp(_prefix, _id, _data):
  url = URL + _prefix + str(_id)
  # print('putHttp: ', url, ', data: ', _data)
  response = requests.put(url, _data)
  # print('putHttp - response: ', response)
  return response.status_code, response.text

def getProjectIdnData(_project_name):
  code, text = getHttp(PREFIX_INFO, _project_name)
  if len(json.loads(text)) == 0:
    return -1, ''
  return json.loads(text)[0]['id'], json.loads(text)[0]

def getProjectDetailIdnData(_project_name):
  code, text = getHttp(PREFIX_DETAIL, _project_name)
  if len(json.loads(text)) == 0:
    return -1, ''
  return json.loads(text)[0]['id'], json.loads(text)[0]

def makeProjectDetailData(_project_name, _column_list, _column_target, _eda_path, _out_path, _train_results):
  return {
    'project_name': _project_name,
    'column_list': str(_column_list),
    'column_target': _column_target,
    'eda_path': _eda_path,
    'out_path': _out_path,
    'train_results': str(_train_results)
  }
    

def makeProjectInfoData(_project_name, _problem_type, _loss):
  metrics_name = 'Accuracy' if _problem_type == definitions.PROBLEM_TYPE_CLASSIFICATION else 'MeanAbsoluteError'
  return {
    'project_name': _project_name,
    'problem_type': _problem_type,
    'metrics_name': metrics_name,
    'best_loss': _loss
  }