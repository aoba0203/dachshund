# Scaler Algorithm
# MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler

#%%
import pandas as pd
import numpy as np
from hyperopt import hp
from utils import definitions
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor, XGBClassifier
import operator
#%%
class FeatureSelection:
  def __init__(self, _problem_type):
    self.problem_type = _problem_type
  
  def __getDroppedColumnDf(self, _df):
    drop_column_list = []
    for column, dtype in zip(_df.columns, _df.dtypes):
      if (dtype != 'int64') and (dtype != 'float64'):
        drop_column_list.append(column)
    _df = _df.drop(drop_column_list, axis=1)
    return _df
  
  def __getSelectedNoneDf(self, _df, _y, _ratio):
    return _df

  def __getSelectedVarianceThreshold(self, _df, _y, _ratio):
    df = self.__getDroppedColumnDf(_df.copy())
    select = VarianceThreshold(threshold=(.8 * (1 - .8)))
    df = select.fit_transform(df)
    column_count = np.shape(df)[1]
    dic = {}
    for n, v in zip(_df.columns.values, select.variances_):
        dic[n] = v
    new_columns = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)    
    return df

  def __getSelectedUnivariate(self, _df, _y, _ratio):
    df = self.__getDroppedColumnDf(_df.copy())
    column_count = int(len(df.columns) * _ratio)
    scorer = f_classif
    if self.problem_type == definitions.PROBLEM_TYPE_CLASSIFICATION:
      scorer = f_classif
    elif self.problem_type == definitions.PROBLEM_TYPE_REGRESSION:
      scorer = f_regression
    x_new = SelectKBest(scorer, k=column_count).fit_transform(df)
    return x_new
  
  def __getSelectedRecursiveFeatureElimination(self, _df, _y, _ratio):
    df = self.__getDroppedColumnDf(_df.copy())
    column_count = int(len(df.columns) * _ratio)
    estimator = XGBClassifier()
    if self.problem_type == definitions.PROBLEM_TYPE_CLASSIFICATION:
      estimator = XGBClassifier()
    elif self.problem_type == definitions.PROBLEM_TYPE_REGRESSION:
      estimator = XGBRegressor()
    selector = RFE(estimator, n_features_to_select=column_count).fit(df, _y)
    column = np.array(_df.columns)
    new_columns = column[np.where(selector.ranking_ == 1, True, False)]
    new_x = selector.transform(df)
    return new_x

  def getFeatureSelectionMethodList(self):    
    return {
      'None': self.__getSelectedNoneDf,
      'VarianceThreshold': self.__getSelectedVarianceThreshold,
      'Univariate': self.__getSelectedUnivariate,
      'RFE': self.__getSelectedRecursiveFeatureElimination,
    }

#%%
if __name__ == '__main__':
  rootPath = definitions.getProjectRootPath()
  df = pd.read_csv(os.path.join(rootPath, 'sample.csv'))

  # f_scaler = FeatureScaler()
  # hyperparams_space = {
  #   'feature_scaler': hp.choice('feature_scaler', f_scaler.getFeatureScalerMethodList().values())
  # }
