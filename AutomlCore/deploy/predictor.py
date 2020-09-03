from utils import definitions, utils
from preprocess import feature_add, feature_missing, feature_outlier, feature_scaler, feature_selection
from utils.definitions import KEY_FEATURE_ADD_NAME, KEY_FEATURE_MIS_NAME, KEY_FEATURE_OUT_NAME, KEY_FEATURE_SCA_NAME, KEY_FEATURE_SEL_NAME, KEY_FEATURE_SEL_RATE_NAME
from utils.definitions import KEY_FEATURE_ADD_NAME_LIST, KEY_FEATURE_MIS_NAME_LIST, KEY_FEATURE_OUT_NAME_LIST, KEY_FEATURE_SCA_NAME_LIST, KEY_FEATURE_SEL_NAME_LIST, KEY_FEATURE_SEL_COL_LIST
import joblib
import json

class Predictor:
  def __init__(self, _problem_type):
    self.f_add = feature_add.FeatureAdd().getFeatureAddMethodList()
    self.f_missing = feature_missing.MissingData().getMissingDataMethodList()    
    self.f_scaler = feature_scaler.FeatureScaler().getFeatureScalerMethodList()
    self.f_selection = feature_selection.FeatureSelection(_problem_type).getFeatureSelectionMethodList()
    self.feature_selection_list = [0.5, 0.7, 0.8, 0.9]
  
  def getPredictResults(self, _x, _project_name, _model_name, _data_ratio):
    self.f_outlier = feature_outlier.FeatureOutlier(_data_ratio).getRemovedOutlierMethodList()
    x = self.__initDataProcess(_x)
    path_model = definitions.getBestModelFilePath(_project_name, _model_name, _data_ratio)
    path_params = definitions.getBestModelParamsFilePath(_project_name, _model_name, _data_ratio)
    
    model = joblib.load(path_model)
    
    result_file = open(path_params, 'r')
    best = json.load(result_file)
    result_file.close()

    x = self.__getPreprocessedDf(x, best)
    x.to_csv('test.csv')
    return model.predict(x)

  def __initDataProcess(self, _df):
    df = utils.splitDateColumns(_df)
    return utils.convertObjectType(df)   

  def __getPreprocessedDf(self, _df, _params):
    df = _df.copy()
    df = (list(self.f_missing.values())[_params['feature_missing']])(df)
    # df = (list(self.f_outlier.values())[_params['feature_outlier']])(df)
    # df = (list(self.f_add.values())[_params['feature_add']])(df)    
    # x = (list(self.f_scaler.values())[_params['feature_scaler']])(df)
    # x, y = self.__splitXy(df)
    x = (list(self.f_scaler.values())[_params[KEY_FEATURE_SCA_NAME]])(df)
    # x = (list(self.f_selection.values())[_params[KEY_FEATURE_SEL_NAME]])(x, y, _params[KEY_FEATURE_SEL_RATE_NAME])
    # feature_list = _params[KEY_FEATURE_SEL_COL_LIST]
    # x = x[feature_list]
    return x

