#%%%
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from utils import definitions
OUTLIER_FRACTION = 0.01
# %%
def getRobustCovairance(_df):
  clf = EllipticEnvelope(contamination=OUTLIER_FRACTION)
  return clf.fit_predict(_df)

#%%
def getOneClassSvm(_df):
  clf = svm.OneClassSVM(nu=OUTLIER_FRACTION)
  return clf.fit_predict(_df)

#%%
def getIsolationForest(_df):
  clf = IsolationForest(
    contamination=OUTLIER_FRACTION,
    n_jobs=definitions.getNumberOfCore()
    )
  return clf.fit_predict(_df)
# %%
def getLocalFactor(_df):
  clf = LocalOutlierFactor(
    contamination=OUTLIER_FRACTION,
    n_jobs=definitions.getNumberOfCore()
  )
  return clf.fit_predict(_df)
