# clustering algorithm
# k-means, affinity propagation, optics, DBSCAN

#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

#%%
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
from hyperopt import hp
from utils import definitions
#%%
class FeatureAdd:
  def __init__(self):    
    self.n_class = 12

  def __getDroppedColumnDf(self, _df):
    drop_column_list = []
    for column, dtype in zip(_df.columns, _df.dtypes):
      if (dtype != 'int64') and (dtype != 'float64'):
        drop_column_list.append(column)
    _df = _df.drop(drop_column_list, axis=1)
    return _df
  
  def __getAddNoneDf(self, _df):
    return _df

  def __getAddKmeansDf(self, _df):
    df = self.__getDroppedColumnDf(_df.copy())
    kmeans = MiniBatchKMeans(n_clusters=self.n_class)
    pred = kmeans.fit_predict(df)
    df['added_cluster'] = pred
    return df

  def __getAddDbscanDf(self, _df):
    df = self.__getDroppedColumnDf(_df.copy())
    dbscan = DBSCAN(n_jobs=definitions.getNumberOfCore())
    pred = dbscan.fit_predict(df)
    df['added_cluster'] = pred
    return df

  def __getAddGaussianDf(self, _df):
    df = self.__getDroppedColumnDf(_df.copy())
    gmm = GaussianMixture(n_components=self.n_class)
    pred = gmm.fit_predict(df)
    df['added_cluster'] = pred
    return df

  def getFeatureAddMethodList(self):
    return {
      'None': self.__getAddNoneDf, 
      'K-Means': self.__getAddKmeansDf, 
      # 'DBSCAN': self.__getAddDbscanDf, 
      'GausianMixture': self.__getAddGaussianDf
      }

#%%
if __name__ == '__main__':
  rootPath = definitions.getProjectRootPath()
  df = pd.read_csv(os.path.join(rootPath, 'sample.csv'))

  f_add = FeatureAdd('testPjt', df)
  hyperparams_space = {
    'feature_add': hp.choice('feature_add', f_add.getFeatureAddMethodList().values())
  }