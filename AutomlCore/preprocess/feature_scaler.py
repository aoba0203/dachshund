# Scaler Algorithm
# MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler

#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

#%%

import pandas as pd
import numpy as np
from hyperopt import hp
from utils import definitions
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler
#%%
class FeatureScaler:
  def __getDroppedColumnDf(self, _df):
    drop_column_list = []
    for column, dtype in zip(_df.columns, _df.dtypes):
      if (dtype != 'int64') and (dtype != 'float64'):
        drop_column_list.append(column)
    _df = _df.drop(drop_column_list, axis=1)
    return _df

  def __setScaledDataColumns(self, _df_ori, _df_scaled):
    for column in _df_scaled.columns:
      _df_ori[[column]] = _df_scaled[[column]]
    return _df_ori

  def __getScaledDf(self, _df, _method):
    # df_ori = _df.copy()
    df = self.__getDroppedColumnDf(_df.copy())
    scaled = _method.fit_transform(df)
    df_scale = pd.DataFrame(scaled, columns=df.columns)
    # df_ori = self.__setScaledDataColumns(df_ori, df_scale)
    return df_scale
  
  def __getScaledNoneDf(self, _df):
    return _df

  def __getScaledMinMaxDf(self, _df):
    return self.__getScaledDf(_df, MinMaxScaler())

  def __getScaledMaxAbsDf(self, _df):
    return self.__getScaledDf(_df, MaxAbsScaler())
  
  def __getScaledNormalizerDf(self, _df):
    return self.__getScaledDf(_df, Normalizer())
  
  def __getScaledRobustDf(self, _df):
    return self.__getScaledDf(_df, RobustScaler())

  def getFeatureScalerMethodList(self):    
    return {
      'None': self.__getScaledNoneDf,
      'MinMax': self.__getScaledMinMaxDf,
      'MaxAbs': self.__getScaledMaxAbsDf,
      'Normalizer': self.__getScaledNormalizerDf,
      'Robust': self.__getScaledRobustDf
    }

#%%
if __name__ == '__main__':
  rootPath = definitions.getProjectRootPath()
  df = pd.read_csv(os.path.join(rootPath, 'sample.csv'))

  f_scaler = FeatureScaler()
  hyperparams_space = {
    'feature_scaler': hp.choice('feature_scaler', f_scaler.getFeatureScalerMethodList().values())
  }
