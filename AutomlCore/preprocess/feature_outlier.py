#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

#%%
import pandas as pd
# import data_frame
from utils import definitions
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from dask.distributed import Client
from utils import definitions
import joblib
# from algorithms import outlier

OUTLIER_FRACTION = 0.01
#%%
class FeatureOutlier():
  def __init__(self, _data_ratio):    
    self.data_ratio = _data_ratio
    self.column_name_outlier = 'outlier'
    self.column_name_robust = 'outlier_robust'
    self.column_name_ocsvm = 'outlier_ocsvm'
    self.column_name_iforest = 'outlier_iforest'
    self.column_name_localfactor = 'outlier_local'
    self.path_preprocessing = definitions.getPreprocessPath()

  def __getDroppedColumnDf(self, _df):
    drop_column_list = []
    for column, dtype in zip(_df.columns, _df.dtypes):
      if (dtype != 'int64') and (dtype != 'float64'):
        drop_column_list.append(column)
    _df = _df.drop(drop_column_list, axis=1)
    return _df

  def __getRemovedOutlierNoneDf(self, _df):
    return _df

  def __getRemoveOutlierDf(self, _df, _algorithm, _columns_name, _column_drop=True):        
    file_path = self.path_preprocessing + '/' + str(_columns_name) + '_' + str(self.data_ratio) +'.csv'
    if os.path.exists(file_path):
      df = pd.read_csv(file_path)
      if _column_drop:
        return df.drop(self.column_name_outlier, axis=1)
      else:
        return df
    else:    
      df = self.__getDroppedColumnDf(_df.copy())
      pred = _algorithm.fit_predict(df)
      df[self.column_name_outlier] = pred
      df.to_csv(file_path, index=False)
      df = df[df[self.column_name_outlier] == 1]      
      if _column_drop:
        df = df.drop(self.column_name_outlier, axis=1)      
      return df
  
  def __getRemovedOutlierRobustDf(self, _df, _column_drop=True):    
    robust = EllipticEnvelope(contamination=OUTLIER_FRACTION)
    return self.__getRemoveOutlierDf(_df, robust, self.column_name_robust, _column_drop)

  # def __getRemovedOutlierOneClassSvmDf(self):
  #   filename = 'preprocess_outlier_oneclass_svm.csv'
  #   ocsvm = svm.OneClassSVM(nu=OUTLIER_FRACTION)
  #   return self.__getRemoveOutlierDf(filename, ocsvm)

  def __getRemovedOutlierIsolationForestDf(self, _df, _column_drop=True):
    iForest = IsolationForest(contamination=OUTLIER_FRACTION, n_jobs=definitions.getNumberOfCore())
    return self.__getRemoveOutlierDf(_df, iForest, self.column_name_iforest, _column_drop)
  
  def __getRemovedOutlierLocalFactorDf(self, _df, _column_drop=True):
    localFactor = LocalOutlierFactor(contamination=OUTLIER_FRACTION, n_jobs=definitions.getNumberOfCore())
    return self.__getRemoveOutlierDf(_df, localFactor, self.column_name_localfactor, _column_drop)
  
  def __getRemovedOutlierIntersectionDf(self, _df):
    df = _df.copy()
    df[self.column_name_robust] = self.__getRemovedOutlierRobustDf(_df, _column_drop=False)[self.column_name_outlier]
    # df[self.column_name_ocsvm] = self.__getRemovedOutlierOneClassSvmDf()[self.column_name_outlier]
    df[self.column_name_iforest] = self.__getRemovedOutlierIsolationForestDf(_df, _column_drop=False)[self.column_name_outlier]
    df[self.column_name_localfactor] = self.__getRemovedOutlierLocalFactorDf(_df, _column_drop=False)[self.column_name_outlier]
    df_outlier = df[
      (df[self.column_name_robust] == 1) & 
      # (df[self.column_name_ocsvm] == 1) & 
      (df[self.column_name_iforest] == 1) &
      (df[self.column_name_localfactor] == 1)
    ]
    df_outlier = df_outlier.drop([self.column_name_robust, self.column_name_iforest, self.column_name_localfactor], axis=1)
    return df_outlier

  def getRemovedOutlierMethodList(self):
    return {
      'None': self.__getRemovedOutlierNoneDf, 
      'Robust': self.__getRemovedOutlierRobustDf, 
      # 'OneClassSVM': self.__getRemovedOutlierOneClassSvmDf, 
      'IsolationForest': self.__getRemovedOutlierIsolationForestDf,
      'LocalFactor': self.__getRemovedOutlierLocalFactorDf,
      'Intersection': self.__getRemovedOutlierIntersectionDf
      }
    