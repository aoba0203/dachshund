#%%
from math import pi
from multiprocessing.spawn import freeze_support
import pandas as pd
import numpy as np
from sklearn import linear_model
from utils import utils, definitions, http_request
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

#%%
# data = {'project_name': 'r_Sales_sample-small', 'project_type': 'Regression', 'project_metrics': 'Accucary', 'project_loss': 1990.4905800560748}
# http_request.putHttp(http_request.PREFIX_INFO, 17, data)
# data =  {'project_name': 'r_Sales_sample-small', 'column_list': "['Store', 'DayOfWeek', 'Sales', 'Open', 'Promo', 'SchoolHoliday', 'Date_convert', 'StateHoliday_convert']", 'column_target': 'Sales', 'eda_path': 'D:\\workspace\\dachshund\\media\\results\\r_Sales_sample-small\\r_Sales_sample-small-pandas_profiling.html', 'out_path': 'D:\\workspace\\dachshund\\media\\results\\r_Sales_sample-small\\r_Sales_sample-small-outlier.html', 'train_results': "[{'alpha_1': 0.05788850982133198, 'alpha_2': 0.0004982593143396108, 'compute_score': 1, 'copy_X': 1, 'feature_add': 1, 'feature_missing': 0, 'feature_outlier': 4, 'feature_scaler': 2, 'feature_selection': 0, 'feature_selection_rate': 0.5, 'fit_intercept': 1, 'lambda_1': 0.0001904503007263171, 'lambda_2': 0.9109193931104238, 'n_iter': 288.0, 'normalize': 1, 'feature_add_name': ['None', 'K-Means', 'GausianMixture'], 'feature_missing_name': ['Fill_0', 'Fill_Previous', 'Fill_Next', 'Fill_Mean', 'Fill_Median', 'Fill_Frequent', 'Fill_Iterative'], 'feature_outlier_name': ['None', 'Robust', 'IsolationForest', 'LocalFactor', 'Intersection'], 'feature_scaler_name': ['None', 'MinMax', 'MaxAbs', 'Normalizer', 'Robust'], 'feature_selection_name': ['None', 'RFE'], 'feature_selection_columns': [], 'score': 1990.4905800560748}]"}
# http_request.postHttp(http_request.PREFIX_DETAIL, data)

# %%
