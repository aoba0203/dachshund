import pandas as pd
import numpy as np
import datetime
import json
import os
from . import definitions

def isDateColumn(_df, _column):
  try:
    if len(_df[_df[_column] < 0]) > 0:
      return False
    df_local = pd.to_datetime(_df[_column])
    len_unique_day = len(pd.unique(df_local.dt.day))
    len_unique_hour = len(pd.unique(df_local.dt.hour))
    # if (len_unique_day > 1) | (len_unique_hour > 1):
    if (len_unique_day > 7):
      return True
  except:
    return False
  return False

def getDateColumnList(_df):
  date_column_list = []
  for column in _df.columns:
    if isDateColumn(_df, column):
      date_column_list.append(column)
  return date_column_list

def splitDateColumns(_df):
  date_column_list = getDateColumnList(_df)
  for column in date_column_list:
    df_local = pd.to_datetime(_df[column])
    # if len(pd.unique(df_local.dt.year)) > 1:
    _df[column + '_year'] = df_local.dt.year
    # if len(pd.unique(df_local.dt.month)) > 1:
    _df[column + '_month'] = df_local.dt.month
    if len(pd.unique(df_local.dt.day)) > 1:
      _df[column + '_day'] = df_local.dt.day
    if len(pd.unique(df_local.dt.hour)) > 1:
      _df[column + '_hour'] = df_local.dt.hour
    if len(pd.unique(df_local.dt.minute)) > 1:
      _df[column + '_minute'] = df_local.dt.minute
    _df = _df.drop(column, axis=1)
  return _df

def convertObjectType(_df):  
  for dtype, column in zip(_df.dtypes, _df.columns):
    if dtype == 'object':
      _df[column + '_convert'] = pd.factorize(_df[column])[0]
      _df = _df.drop(column, axis=1)
  return _df

def __jsonConverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()

def writeJsonToFile(_json, _filepath):
  with open(_filepath, 'w') as result_file:
    json.dump(_json, result_file, default=__jsonConverter)

def getJsonFromFile(_filepath):
  if os.path.exists(_filepath):
    result_file = open(_filepath, 'r')
    best = json.load(result_file)
    result_file.close()
    return best
  return {}
