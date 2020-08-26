#%%
from math import pi
from multiprocessing.spawn import freeze_support
import pandas as pd
import numpy as np
from sklearn import linear_model
from utils import utils, definitions
from preprocess import feature_outlier
from preprocess import feature_add, feature_missing, feature_outlier, feature_scaler
from eda import de_pdprofiling, df_outlier

from training.train_manager import TrainManager
#%%
import warnings
import project

from file_system_observer import FileSystemWatcher
 
warnings.filterwarnings("ignore")
#%%
if __name__ == '__main__':
  # path_train_file = 'sample.csv'
  # project_name = 'rossman_sales'
  # profiler = de_pdprofiling.PdProfiling(project_name, path_train_file)
  # profiler.makeVisualizerHtmlFile()

  # outliers = df_outlier.DfOutlier(project_name, path_train_file)
  # outliers.makeDataframeHtmlFile()

  # df = pd.read_csv(path_train_file)
  # # df = df.sample(int(len(df) * 0.5))
  # t = TrainManager(definitions.PROBLEM_TYPE_REGRESSION, 'testPjt', df, 'Sales')
  # t.startWorkerAdmin()
  print('run main')
  w = FileSystemWatcher()
  w.run()