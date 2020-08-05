# Missing Algorithm
# mean, median, most frequency, previous value, next value, fill 0

#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

#%%
import pandas as pd
import numpy as np
from hyperopt import hp
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from utils import definitions
#%%
class MissingData:
  def __setMissingDataColumns(self, _df_ori, _df_fillin):
    # for column in _df_fillin.columns:
    #   _df_ori[[column]] = _df_fillin[[column]]
    # return _df_ori
    return pd.merge(_df_ori, _df_fillin)

  def __getFillMissingData(self, _df, _strategy, _fill_value=None):
    df_ori = _df.copy()
    df_fillin = _df.copy()
    imputer = SimpleImputer(strategy=_strategy, fill_value=_fill_value)
    fillin_data = imputer.fit_transform(df_fillin)
    df_fillin = pd.DataFrame(data=fillin_data, columns=df_fillin.columns)
    df_ori = self.__setMissingDataColumns(df_ori, df_fillin)
    return df_ori

  def __getFillnaDataframe(self, _df, _strategy):
    df_ori = _df.copy()
    df_fillin = _df.copy()
    df_fillin = df_fillin.fillna(method=_strategy)
    df_ori = self.__setMissingDataColumns(df_ori, df_fillin)
    return df_ori

  def __fillIterativeImputer(self, _df):
    df_ori = _df.copy()
    df_fillin = _df.copy()
    imputer = IterativeImputer()
    fillin_data = imputer.fit_transform(df_fillin)
    df_fillin = pd.DataFrame(data=fillin_data, columns=df_fillin.columns)
    df_ori = self.__setMissingDataColumns(df_ori, df_fillin)
    return df_ori

  def __getFillnaNoneDf(self, _df):
    return _df

  def __getFillna0Df(self, _df):
    return self.__getFillMissingData(_df, 'constant', 0)

  def __getFillnaPrevDf(self, _df):
    return self.__getFillnaDataframe(_df, 'ffill')

  def __getFillnaNextDf(self, _df):
    return self.__getFillnaDataframe(_df, 'bfill')

  def __getFillnaMean(self, _df):
    return self.__getFillMissingData(_df, 'mean')
  
  def __getFillnaMedian(self, _df):
    return self.__getFillMissingData(_df, 'median')
  
  def __getFillnaFrequent(self, _df):
    return self.__getFillMissingData(_df, 'most_frequent')
  
  def __getFillnaIterative(self, _df):
    return self.__fillIterativeImputer(_df)

  def getMissingDataMethodList(self):
    return {
      # 'None': self.__getFillnaNoneDf, 
      'Fill_0': self.__getFillna0Df, 
      'Fill_Previous': self.__getFillnaPrevDf, 
      'Fill_Next': self.__getFillnaNextDf,
      'Fill_Mean': self.__getFillnaMean,
      'Fill_Median': self.__getFillnaMedian,
      'Fill_Frequent': self.__getFillnaFrequent,
      'Fill_Iterative': self.__getFillnaIterative
    }

#%%
if __name__ == '__main__':
  rootPath = definitions.getProjectRootPath()
  csv_path = os.path.join(rootPath, 'sample_small.csv')
  df = pd.read_csv(csv_path)
  missing_data = MissingData(df)  
  hyperparams_space = {
    'missing_data': hp.choice('feature_scaler', missing_data.getMissingDataMethodList().values())
  }
