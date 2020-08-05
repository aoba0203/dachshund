#%%
from math import pi
from multiprocessing.spawn import freeze_support
import pandas as pd
import numpy as np
from sklearn import linear_model
from utils import utils, definitions
from preprocess import feature_outlier

from training.train_manager import TrainManager
#%%
import warnings
 
warnings.filterwarnings("ignore")
# print(len(glob.glob('algorithms/regression/*.py')))
# print(len(glob.glob('algorithms/regression/_*.py')))

# print(len(glob.glob('algorithms/classification/*.py')))
# print(len(glob.glob('algorithms/classification/_*.py')))

#%%
if __name__ == '__main__':
  df = pd.read_csv('sample_small.csv')
  t = TrainManager(0, 'testPjt', df, 'Sales')
  t.startWorkerAdmin()
#%%
# dic = {'copy_X': 1, 'eps': 0.34864529240750924, 'feature_add': 0, 'feature_missing': 4, 'feature_outlier': 4, 'feature_scaler': 2, 'fit_intercept': 0, 'normalize': 0}
# import json
# from utils import utils, definitions

# filepath = definitions.getBestModelParamsFilePath('test', 'model', '50%')
# result_file = open((filepath), 'w')
# json.dump(dic, result_file)
# result_file.close()
#%%
# df = pd.read_csv('sample_small.csv')
# t_manager = train_manager.TrainManager('pjt_manager', df, 'Sales')
# t_manager.optimizeModel()

# t_manager.loadModel()

# job = t_manager.getJob()
# print(job.model.model_name, job.model.getMaxIterCount())
# job = t_manager.getJob()
# print(job.model.model_name, job.model.getMaxIterCount())