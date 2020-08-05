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
# from algorithms import outlier

OUTLIER_FRACTION = 0.01
#%%
class FeatureOutlier():
  def __init__(self):    
    self.column_name_outlier = 'outlier'
    self.column_name_robust = 'outlier_robust'
    self.column_name_ocsvm = 'outlier_ocsvm'
    self.column_name_iforest = 'outlier_iforest'
    self.column_name_localfactor = 'outlier_local'

  def __getDroppedColumnDf(self, _df):
    drop_column_list = []
    for column, dtype in zip(_df.columns, _df.dtypes):
      if (dtype != 'int64') and (dtype != 'float64'):
        drop_column_list.append(column)
    _df = _df.drop(drop_column_list, axis=1)
    return _df

  def __getRemovedOutlierNoneDf(self, _df):
    return _df

  def __getRemoveOutlierDf(self, _df, _algorithm):    
    df = self.__getDroppedColumnDf(_df.copy())
    pred = _algorithm.fit_predict(df)
    df[self.column_name_outlier] = pred
    df = df[df[self.column_name_outlier] == 1]
    return df
  
  def __getRemovedOutlierRobustDf(self, _df):    
    robust = EllipticEnvelope(contamination=OUTLIER_FRACTION)
    return self.__getRemoveOutlierDf(_df, robust)

  # def __getRemovedOutlierOneClassSvmDf(self):
  #   filename = 'preprocess_outlier_oneclass_svm.csv'
  #   ocsvm = svm.OneClassSVM(nu=OUTLIER_FRACTION)
  #   return self.__getRemoveOutlierDf(filename, ocsvm)

  def __getRemovedOutlierIsolationForestDf(self, _df):
    iForest = IsolationForest(contamination=OUTLIER_FRACTION, n_jobs=definitions.getNumberOfCore())
    return self.__getRemoveOutlierDf(_df, iForest)
  
  def __getRemovedOutlierLocalFactorDf(self, _df):
    localFactor = LocalOutlierFactor(contamination=OUTLIER_FRACTION, n_jobs=definitions.getNumberOfCore())
    return self.__getRemoveOutlierDf(_df, localFactor)
  
  def __getRemovedOutlierIntersectionDf(self, _df):
    df = _df.copy()
    df[self.column_name_robust] = self.__getRemovedOutlierRobustDf(_df)[self.column_name_outlier]
    # df[self.column_name_ocsvm] = self.__getRemovedOutlierOneClassSvmDf()[self.column_name_outlier]
    df[self.column_name_iforest] = self.__getRemovedOutlierIsolationForestDf(_df)[self.column_name_outlier]
    df[self.column_name_localfactor] = self.__getRemovedOutlierLocalFactorDf(_df)[self.column_name_outlier]
    df_outlier = df[
      (df[self.column_name_robust] == 1) & 
      # (df[self.column_name_ocsvm] == 1) & 
      (df[self.column_name_iforest] == 1) &
      (df[self.column_name_localfactor] == 1)
    ]
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
    